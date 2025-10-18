# guarded_extractor.py
from __future__ import annotations

import collections
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------- PDF libs (we'll try multiple strategies) ----------
import pypdf

# Optional fallbacks (imported lazily where used)
_have_pdfminer = True
try:
    # We'll import more precisely inside the function to keep import errors soft
    import pdfminer  # noqa: F401
except Exception:
    _have_pdfminer = False

_have_ocr = True
try:
    import pytesseract  # type: ignore
    from pdf2image import convert_from_path  # type: ignore
    from PIL import Image  # noqa: F401
except Exception:
    _have_ocr = False

# ---------- OpenAI client ----------
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# ---------------- Global toggles ----------------
# If True, merged_ir.json MUST be produced by the model (no local fallback).
STRICT_MERGE = True

# ---------------- Pydantic IR Schema ----------------
try:
    from pydantic import BaseModel, Field, ValidationError, conlist  # noqa: F401
    PydanticAvailable = True
except Exception:  # pragma: no cover
    PydanticAvailable = False

if PydanticAvailable:
    Op = Union[str]

    class Variable(BaseModel):
        name: str
        type: str  # "number" | "boolean" | "string"
        unit: Optional[str] = None
        allowed: Optional[List[str]] = None
        synonyms: List[str] = []
        prompt: Optional[str] = None
        refs: List[str] = []

    class ObsCond(BaseModel):
        obs: str
        op: str  # "lt|le|gt|ge|eq|ne"
        value: Union[float, int, bool, str]

    class SymCond(BaseModel):
        sym: str
        eq: bool

    class AllOf(BaseModel):
        all_of: conlist(Union[ObsCond, SymCond], min_length=1)

    class AnyOf(BaseModel):
        any_of: conlist(Union[ObsCond, SymCond], min_length=1)

    Condition = Union[ObsCond, SymCond, AllOf, AnyOf]

    class ThenBlock(BaseModel):
        triage: Optional[str] = None  # "hospital|clinic|home"
        flags: List[str] = []
        reasons: List[str] = []
        actions: List[dict] = []
        guideline_ref: Optional[str] = None
        priority: int = 0
        advice: List[str] = []  # MUST remain empty (no treatment text)

    class Rule(BaseModel):
        rule_id: str
        when: conlist(Condition, min_length=1)
        then: ThenBlock

    class QA(BaseModel):
        notes: List[str] = []
        unmapped_vars: List[str] = []
        dedup_dropped: int = 0
        overlap_fixed: List[str] = []

    class IR(BaseModel):
        variables: List[Variable] = []
        rules: List[Rule] = []
        canonical_map: dict = {}
        qa: QA = QA()

# ---------------- JSON repair & retries ----------------

def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()

def _loose_to_json(s: str) -> Any:
    s = _strip_code_fences(s)
    # strip comments
    lines: List[str] = []
    in_block = False
    for ln in s.splitlines():
        t = ln
        if in_block:
            if "*/" in t:
                t = t.split("*/", 1)[1]
                in_block = False
            else:
                continue
        if "/*" in t:
            t = t.split("/*", 1)[0]
            in_block = True
        if "//" in t:
            t = t.split("//", 1)[0]
        lines.append(t)
    s = "\n".join(lines)
    # remove trailing commas
    s = re.sub(r",\s*(\]|\})", r"\1", s)
    # crop to first obj/array
    for opener, closer in [('{', '}'), ('[', ']')]:
        a, b = s.find(opener), s.rfind(closer)
        if a != -1 and b != -1 and b > a:
            cand = s[a : b + 1]
            try:
                return json.loads(cand)
            except Exception:
                pass
    return json.loads(s)

def _call_json_with_retries(call_fn, schema_model=None, retries=3, sleep=1.5):
    """Ensure this helper is in global scope (fixes NameError)."""
    last_exc = None
    for i in range(retries):
        try:
            raw = call_fn()
            obj = _loose_to_json(raw)
            if schema_model is not None and PydanticAvailable:
                obj = schema_model.model_validate(obj).model_dump()
            return obj
        except Exception as e:
            last_exc = e
            time.sleep(sleep * (2 ** i))
    raise last_exc

def _extract_fenced_blocks(text: str) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    i = 0
    while True:
        start = text.find("```", i)
        if start == -1:
            break
        nl = text.find("\n", start)
        if nl == -1:
            break
        lang = text[start + 3 : nl].strip() or ""
        end = text.find("```", nl + 1)
        if end == -1:
            break
        body = text[nl + 1 : end].strip()
        blocks.append((lang, body))
        i = end + 3
    return blocks

# --- BPMN helpers -------------------------------------------------------------
def _extract_xml_tag(raw: str, tag: str) -> Optional[str]:
    """Pull <tag>...</tag> out of an unstructured string."""
    start = raw.find(f"<{tag}")
    end = raw.find(f"</{tag}>")
    if start != -1 and end != -1 and end > start:
        end += len(f"</{tag}>")
        return raw[start:end].strip()
    return None

def _sanitize_bpmn(xml: str) -> str:
    """Ensure minimal namespaces and a closing </bpmn:definitions>."""
    s = (xml or "").strip()
    if not s:
        return s
    if 'xmlns:bpmn="' not in s:
        s = s.replace(
            "<bpmn:definitions",
            '<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            1
        )
    if "</bpmn:definitions>" not in s and s.count("<bpmn:definitions") == 1:
        s += "\n</bpmn:definitions>"
    return s

# ---------------- Normalization helpers ----------------
_CANON_TRIAGE_KEYS = ("triage", "propose_triage")
_CANON_FLAG_KEYS = ("flags", "set_flags")
_DERIVED_BLOCKLIST = {"danger_sign", "clinic_referral", "triage"}

def _to_snake(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "var"

def _normalize_rule_schema(rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    r = dict(rule or {})
    th = dict(r.get("then") or {})
    triage = None
    for k in _CANON_TRIAGE_KEYS:
        if th.get(k) is not None:
            triage = th[k]; break
    if triage is not None:
        th["triage"] = triage
    flags = None
    for k in _CANON_FLAG_KEYS:
        if th.get(k) is not None:
            flags = th[k]; break
    th["flags"] = list(flags or [])
    th.setdefault("reasons", [])
    th.setdefault("actions", [])
    th.setdefault("advice", [])  # keep empty
    th["priority"] = int(th.get("priority") or 0)
    r["then"] = th

    when = r.get("when")
    if not isinstance(when, list) or len(when) == 0:
        return None
    def _has_bad_sym(c: Dict[str, Any]) -> bool:
        return "sym" in c and str(c.get("sym")).strip().lower() in _DERIVED_BLOCKLIST
    for cond in when:
        if not isinstance(cond, dict): return None
        if _has_bad_sym(cond): return None
        if "all_of" in cond or "any_of" in cond:
            seq = cond.get("all_of") or cond.get("any_of") or []
            for sub in seq:
                if _has_bad_sym(sub): return None
    return r

def _dedupe_rule_ids(rules: List[Dict[str, Any]], prefix: str = "r") -> List[Dict[str, Any]]:
    seen = {}
    out = []
    for i, r in enumerate(rules, 1):
        rid = str(r.get("rule_id") or f"{prefix}_{i}").strip()
        rid = re.sub(r"\s+", "_", rid.lower())
        if rid in seen:
            k = 2
            new_id = f"{rid}__{k}"
            while new_id in seen:
                k += 1
                new_id = f"{rid}__{k}"
            rid = new_id
        seen[rid] = True
        r["rule_id"] = rid
        out.append(r)
    return out

def _build_canonical_map(config: Dict[str, Any], variables: List[Dict[str, Any]]) -> Dict[str, str]:
    canon_set = set(config.get("canonical_variables", []))
    by_name = { (v.get("name") or "").strip().lower(): v for v in variables if isinstance(v, dict) }
    idx: Dict[str, str] = {}
    for name, v in by_name.items():
        syns = set([name]) | set((v.get("synonyms") or []))
        target = next((s for s in syns if s in canon_set), name)
        for s in syns:
            idx[str(s).strip().lower()] = target
    return idx

def _rewrite_rule_to_canon(rule: Dict[str, Any], canon: Dict[str, str]) -> Dict[str, Any]:
    r = dict(rule)
    def map_var(x: str) -> str:
        return canon.get(str(x).strip().lower(), str(x).strip().lower())
    new_when = []
    for c in r.get("when", []):
        c = dict(c)
        if "sym" in c:
            c["sym"] = map_var(c["sym"])
        elif "obs" in c:
            c["obs"] = map_var(c["obs"])
        elif "all_of" in c or "any_of" in c:
            k = "all_of" if "all_of" in c else "any_of"
            seq = []
            for s in c.get(k) or []:
                s = dict(s)
                if "sym" in s: s["sym"] = map_var(s["sym"])
                if "obs" in s: s["obs"] = map_var(s["obs"])
                seq.append(s)
            c[k] = seq
        new_when.append(c)
    r["when"] = new_when
    return r

def _flatten_rule_conditions(rule: Dict[str, Any]) -> List[str]:
    out = []
    def one(c):
        if "sym" in c:
            return f"{c['sym']}=={str(c.get('eq')).lower()}"
        if "obs" in c:
            return f"{c['obs']} {c['op']} {json.dumps(c['value'])}"
        return None
    for c in rule.get("when", []):
        if "all_of" in c or "any_of" in c:
            k = "all_of" if "all_of" in c else "any_of"
            parts = [one(s) for s in (c.get(k) or []) if one(s)]
            out.append(f"{k}:" + " && ".join(parts))
        else:
            s = one(c)
            if s: out.append(s)
    return out

def _rules_flattened(ir: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{
        "rule_id": r.get("rule_id"),
        "triage": (r.get("then") or {}).get("triage"),
        "conds": _flatten_rule_conditions(r)
    } for r in ir.get("rules", [])]

def _preflight_ir(ir: Dict[str, Any]) -> None:
    names = [ (v.get("name") or "").strip().lower() for v in ir.get("variables", []) if isinstance(v, dict) ]
    dup_var_names = [n for n, cnt in collections.Counter(names).items() if cnt > 1]
    if dup_var_names:
        raise RuntimeError(f"Duplicate variable names after merge: {dup_var_names}")
    ids = [ (r.get("rule_id") or "").strip().lower() for r in ir.get("rules", []) if isinstance(r, dict) ]
    dups = [i for i,c in collections.Counter(ids).items() if c>1]
    if dups:
        raise RuntimeError(f"Duplicate rule_ids after dedupe pass: {dups}")
    empties = [r.get("rule_id") for r in ir.get("rules", []) if not (r.get("when") or [])]
    if empties:
        raise RuntimeError(f"Empty WHEN in rules: {empties}")

def _enforce_ask_ownership(ask_plan: List[Dict[str, Any]], priority_order=("module_a","module_b","module_c","module_d","module_e")):
    owner: Dict[str, str] = {}
    order = {m:i for i,m in enumerate(priority_order)}
    for blk in sorted([b for b in ask_plan if isinstance(b, dict)], key=lambda b: order.get(b.get("module","zzz"), 999)):
        module = blk.get("module")
        kept_ask = []
        for q in blk.get("ask", []) or []:
            if q not in owner:
                owner[q] = module
                kept_ask.append(q)
        blk["ask"] = kept_ask
        fuw = {}
        for cond, qs in (blk.get("followups_if") or {}).items():
            new_qs = []
            for q in qs:
                if q not in owner:
                    owner[q] = module
                    new_qs.append(q)
            fuw[cond] = new_qs
        blk["followups_if"] = fuw
    return ask_plan

# ---------- Duplicate resolution + snake-case, with rule rewriting ----------

def _norm_name(s: Optional[str]) -> str:
    return (s or "").strip()

def _pages_from_ref_string(s: str) -> List[int]:
    if not s:
        return []
    out = []
    for tok in re.findall(r"p(\d{1,4})(?:-(\d{1,4}))?", s, flags=re.IGNORECASE):
        a = int(tok[0]); b = int(tok[1]) if tok[1] else None
        if b is None:
            out.append(a)
        else:
            lo, hi = sorted((a, b))
            out.extend(list(range(lo, hi + 1))[:500])
    return out

def _pages_from_refs(refs: Optional[List[str]]) -> set:
    pages = set()
    for r in (refs or []):
        for p in _pages_from_ref_string(str(r)):
            pages.add(p)
    return pages

def _var_sig(v: Dict[str, Any]) -> tuple:
    t = v.get("type")
    u = v.get("unit")
    allowed = tuple(sorted([str(x) for x in (v.get("allowed") or [])]))
    return (t, u, allowed)

def _coalesce_group(vars_with_same_sig: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = dict(vars_with_same_sig[0])
    syn = set(base.get("synonyms") or [])
    refs = set(base.get("refs") or [])
    allowed = set(base.get("allowed") or [])
    for v in vars_with_same_sig[1:]:
        syn |= set(v.get("synonyms") or [])
        refs |= set(v.get("refs") or [])
        allowed |= set(v.get("allowed") or [])
        if not base.get("prompt") and v.get("prompt"):
            base["prompt"] = v["prompt"]
    base["synonyms"] = sorted(list(syn))
    base["refs"] = sorted(list(refs))
    base["allowed"] = sorted(list(allowed)) if allowed else []
    return base

def _resolve_variables_snake_and_rewrite_rules(merged: Dict[str, Any]) -> Dict[str, Any]:
    """
    1) Group variables by snake_case(name) so near-duplicates collide.
    2) Within each group, partition by signature (type/unit/allowed).
       - Same signature → coalesce (merge synonyms/refs).
       - Different signatures → keep as distinct variants with suffixes __2, __3...
    3) Assign final variable names (snake case + optional suffix).
    4) Rewrite rules to refer to the chosen final names using guideline_ref page overlap.
    5) Preserve all original spellings in synonyms.
    """
    qa_notes = merged.setdefault("qa", {}).setdefault("notes", [])
    vars_in = [v for v in (merged.get("variables") or []) if isinstance(v, dict) and _norm_name(v.get("name"))]
    # Build groups by snake key
    groups: Dict[str, List[Dict[str, Any]]] = {}
    names_seen_per_group: Dict[str, set] = {}
    for v in vars_in:
        orig = _norm_name(v.get("name"))
        key = _to_snake(orig)
        vv = dict(v)
        # keep the original spelling in synonyms so we don't lose semantics
        syns = set(vv.get("synonyms") or [])
        syns.add(orig)
        vv["synonyms"] = sorted(list(syns))
        groups.setdefault(key, []).append(vv)
        names_seen_per_group.setdefault(key, set()).update({orig.lower(), *[str(s).strip().lower() for s in syns]})

    new_vars: List[Dict[str, Any]] = []
    # map any known name (lower) -> variant list for this group
    name_to_variants: Dict[str, List[Dict[str, Any]]] = {}
    rename_events: Dict[str, List[str]] = {}

    for base, gvars in groups.items():
        # Partition by signature
        sig_bins: Dict[tuple, List[Dict[str, Any]]] = {}
        for v in gvars:
            sig_bins.setdefault(_var_sig(v), []).append(v)

        variants_meta = []
        if len(sig_bins) > 1:
            qa_notes.append(f"variable_homonyms_split:{base}:{len(sig_bins)}")

        # Coalesce each bin and assign final names
        idx = 1
        for _sig, same_sig_vars in sig_bins.items():
            merged_v = _coalesce_group(same_sig_vars)
            final_name = base if idx == 1 else f"{base}__{idx}"
            idx += 1
            # capture rename info: all originals in this bin
            originals = { _norm_name(x.get("name")) for x in same_sig_vars if _norm_name(x.get("name")) }
            for o in originals:
                rename_events.setdefault(o, []).append(final_name)
            # finalize
            merged_v["name"] = final_name
            pages = _pages_from_refs(merged_v.get("refs") or [])
            variants_meta.append({"new_name": final_name, "pages": pages})
            new_vars.append(merged_v)

        # Wire name → variants map (for rule rewriting)
        for n in names_seen_per_group.get(base, set()):
            name_to_variants[n] = variants_meta

    # Heuristic picker
    def pick_variant(seen_name: str, rule_ref: Optional[str]) -> str:
        variants = name_to_variants.get((seen_name or "").strip().lower())
        if not variants:
            # If we don't know, keep its snake base
            return _to_snake(seen_name)
        if not rule_ref:
            return variants[0]["new_name"]
        r_pages = set(_pages_from_ref_string(rule_ref))
        best = variants[0]; best_sc = -1
        for v in variants:
            sc = len(r_pages & set(v["pages"]))
            if sc > best_sc:
                best_sc = sc; best = v
        return best["new_name"]

    # Rewrite rules
    rules_out: List[Dict[str, Any]] = []
    for r in (merged.get("rules") or []):
        rr = dict(r)
        rule_ref = _norm_name(((rr.get("then") or {}).get("guideline_ref")))
        new_when = []
        for c in (rr.get("when") or []):
            c = dict(c)
            if "sym" in c and c["sym"]:
                sym = _norm_name(c["sym"])
                key = sym.strip().lower()
                if key in name_to_variants or _to_snake(sym) in groups:
                    c["sym"] = pick_variant(sym, rule_ref)
                else:
                    # still snake-case it for consistency
                    c["sym"] = _to_snake(sym)
            elif "obs" in c and c["obs"]:
                obs = _norm_name(c["obs"])
                key = obs.strip().lower()
                if key in name_to_variants or _to_snake(obs) in groups:
                    c["obs"] = pick_variant(obs, rule_ref)
                else:
                    c["obs"] = _to_snake(obs)
            elif "all_of" in c or "any_of" in c:
                k = "all_of" if "all_of" in c else "any_of"
                seq = []
                for s in (c.get(k) or []):
                    s = dict(s)
                    if "sym" in s and s["sym"]:
                        sym = _norm_name(s["sym"])
                        key = sym.strip().lower()
                        if key in name_to_variants or _to_snake(sym) in groups:
                            s["sym"] = pick_variant(sym, rule_ref)
                        else:
                            s["sym"] = _to_snake(sym)
                    if "obs" in s and s["obs"]:
                        obs = _norm_name(s["obs"])
                        key = obs.strip().lower()
                        if key in name_to_variants or _to_snake(obs) in groups:
                            s["obs"] = pick_variant(obs, rule_ref)
                        else:
                            s["obs"] = _to_snake(obs)
                    seq.append(s)
                c[k] = seq
            new_when.append(c)
        rr["when"] = new_when
        rules_out.append(rr)

    # QA note for visibility
    if rename_events:
        # collapse to simple mapping: original -> list of finals
        merged.setdefault("qa", {}).setdefault("notes", []).append(f"snake_case_renamed:{rename_events}")

    merged["variables"] = new_vars
    merged["rules"] = rules_out
    return merged

# ---------- Fact sheet helpers (compact input for merging) ----------
def _fact_sheet_from_sections(section_objs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a compact, deduplicated fact sheet:
      - variables: minimal fields only (name/type/unit/allowed/synonyms/refs)
      - rules: normalized 'when' + minimal 'then' (triage/flags/reasons/actions/guideline_ref/priority)
    """
    var_ix: Dict[str, Dict[str, Any]] = {}
    facts_rules: List[Dict[str, Any]] = []
    for sec in (section_objs or []):
        for v in (sec.get("variables") or []):
            if not isinstance(v, dict): continue
            name = (v.get("name") or "").strip()
            if not name: continue
            key = name.lower()
            cur = var_ix.get(key)
            if cur is None:
                cur = dict(name=name,
                           type=v.get("type"),
                           unit=v.get("unit"),
                           allowed=v.get("allowed"),
                           synonyms=list(v.get("synonyms") or []),
                           refs=list(v.get("refs") or []))
                var_ix[key] = cur
            else:
                # merge synonyms/refs
                cur["synonyms"] = sorted(list(set((cur.get("synonyms") or []) + (v.get("synonyms") or []))))
                cur["refs"] = sorted(list(set((cur.get("refs") or []) + (v.get("refs") or []))))
        for r in (sec.get("rules") or []):
            if not isinstance(r, dict): continue
            nr = _normalize_rule_schema(r)
            if not nr: continue
            th = nr.get("then") or {}
            facts_rules.append({
                "rule_id": r.get("rule_id"),
                "when": nr.get("when") or [],
                "then": {
                    "triage": th.get("triage"),
                    "flags": th.get("flags") or [],
                    "reasons": th.get("reasons") or [],
                    "actions": th.get("actions") or [],
                    "guideline_ref": th.get("guideline_ref"),
                    "priority": int(th.get("priority") or 0),
                    "advice": []  # enforce empty
                }
            })
    return {
        "variables": list(var_ix.values()),
        "rules": _dedupe_rule_ids(facts_rules, prefix="fs"),
        "qa": {"notes": ["from_fact_sheet_builder"]}
    }

# ---------------- Text extraction helpers ----------------

def _extract_pypdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages, start=1):
            try:
                t = (page.extract_text() or "").strip()
            except Exception:
                t = ""
            pages.append((i, t))
    return pages

def _extract_pdfminer_pages(pdf_path: str) -> List[Tuple[int, str]]:
    if not _have_pdfminer:
        return []
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception:
        return []
    # get page count via pypdf (still works even if text empty)
    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            n = len(reader.pages)
    except Exception:
        n = 0
    if n <= 0:
        # last resort: whole doc (will chunk later)
        try:
            full = extract_text(pdf_path) or ""
            if not full.strip():
                return []
            return [(1, full)]
        except Exception:
            return []
    out: List[Tuple[int, str]] = []
    for i in range(n):
        try:
            txt = extract_text(pdf_path, page_numbers=[i]) or ""
        except Exception:
            txt = ""
        out.append((i+1, txt.strip()))
    return out

# OCR extractor (optional)
def _extract_ocr_pages(pdf_path: str) -> List[Tuple[int, str]]:
    if not _have_ocr:
        return []
    pages: List[Tuple[int, str]] = []
    try:
        images = convert_from_path(pdf_path, dpi=200)
    except Exception:
        return []
    for idx, img in enumerate(images, start=1):
        try:
            text = pytesseract.image_to_string(img) or ""
        except Exception:
            text = ""
        pages.append((idx, text.strip()))
    return pages

# Smarter normalization and heading-aware splitting

_heading_rx = re.compile(
    r"^(annex|appendix|module|chapter|section|part|lesson|unit|task|topic)\b"
    r"|\b(assessment|treatment|referral|follow\s*up|training)\b"
    r"|^[A-Z][A-Z0-9 ,:/\-]{6,}$",
    flags=re.IGNORECASE
)

def _normalize_whitespace(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    return s.strip()

def _dehyphenate(s: str) -> str:
    return re.sub(r"(?<=\w)-\n(?=\w)", "", s)

def _normalize_bullets(s: str) -> str:
    return re.sub(r"^[•·◦▪➤→]\s*", "- ", s, flags=re.MULTILINE)

def _strip_repeating_headers_footers(pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    tops, bottoms = [], []
    for _, t in pages:
        lines = [ln.strip() for ln in (t.splitlines() if t else []) if ln.strip()]
        if not lines:
            tops.append(""); bottoms.append(""); continue
        tops.append(lines[0]); bottoms.append(lines[-1])
    def common(items):
        ctr = collections.Counter(items)
        if not ctr: return None
        item, cnt = ctr.most_common(1)[0]
        return item if cnt >= max(3, int(0.4 * len(items))) and len(item) >= 6 else None
    top_c = common(tops)
    bot_c = common(bottoms)
    cleaned: List[Tuple[int, str]] = []
    for i, t in pages:
        if not t:
            cleaned.append((i, t)); continue
        lines = t.splitlines()
        if top_c and lines and lines[0].strip() == top_c: lines = lines[1:]
        if bot_c and lines and lines[-1].strip() == bot_c: lines = lines[:-1]
        cleaned.append((i, "\n".join(lines)))
    return cleaned

def _postprocess_page_text(pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for i, t in pages:
        s = t or ""
        s = _normalize_whitespace(s)
        s = _dehyphenate(s)
        s = _normalize_bullets(s)
        out.append((i, s))
    return _strip_repeating_headers_footers(out)

def _extract_pdfminer_layout_pages(pdf_path: str) -> List[Tuple[int, str]]:
    if not _have_pdfminer:
        return []
    try:
        from pdfminer.high_level import extract_pages  # type: ignore
        from pdfminer.layout import LTTextBox, LTTextLine, LAParams  # type: ignore
    except Exception:
        return []
    laparams = LAParams(line_overlap=0.5, char_margin=2.0, line_margin=0.4, word_margin=0.1, boxes_flow=None)
    pages: List[Tuple[int, str]] = []
    try:
        for pno, page_layout in enumerate(extract_pages(pdf_path, laparams=laparams), start=1):
            items = []
            for el in page_layout:
                if isinstance(el, (LTTextBox, LTTextLine)):
                    x0, y0, x1, y1 = el.bbox
                    items.append((y1, x0, el.get_text()))
            items.sort(key=lambda t: (-t[0], t[1]))
            text = "".join([txt for _, __, txt in items])
            pages.append((pno, text))
    except Exception:
        return []
    return pages

def _split_into_sections_by_headings(pages: List[Tuple[int, str]], max_chars: int = 4000) -> List[Tuple[str, str]]:
    chunks: List[Tuple[str, str]] = []
    cur_buf = ""
    start_page: Optional[int] = None

    def flush(end_page: int):
        nonlocal cur_buf, start_page
        if cur_buf.strip():
            chunks.append((f"p{start_page}-{end_page}", cur_buf))
        cur_buf = ""
        start_page = None

    for pno, txt in pages:
        lines = txt.splitlines()
        i = 0
        while i < len(lines):
            ln = lines[i]
            is_heading = bool(_heading_rx.search(ln.strip()))
            if is_heading and cur_buf and len(cur_buf) > int(0.5 * max_chars):
                flush(pno - 1)
            if start_page is None:
                start_page = pno
            cur_buf += f"[PAGE {pno}]\n{ln}\n"
            if len(cur_buf) >= max_chars:
                flush(pno)
            i += 1
    if cur_buf:
        flush(pno)
    if len(chunks) >= 2 and len(chunks[-1][1]) < int(0.25 * max_chars):
        sid, body = chunks.pop()
        pid, prev = chunks.pop()
        a = pid.split("-")[0]
        b = sid.split("-")[-1]
        chunks.append((f"{a}-{b}", prev + "\n" + body))
    return chunks

def _chunk_pages_len_only(pages: List[Tuple[int, str]], max_chars: int = 4000) -> List[Tuple[str, str]]:
    chunks: List[Tuple[str, str]] = []
    buf = ""; ids: List[int] = []
    for i, t in pages:
        if not t:
            continue
        add = f"[PAGE {i}]\n{t}\n"
        if len(buf) + len(add) > max_chars and buf:
            chunks.append((f"p{ids[0]}-{ids[-1]}", buf))
            buf = ""; ids = []
        buf += add; ids.append(i)
    if buf:
        chunks.append((f"p{ids[0]}-{ids[-1]}", buf))
    return chunks

# ---------------- Guarded Extractor ----------------
class OpenAIGuardedExtractor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_section: str = "gpt-5",
        model_merge: str = "gpt-5",
        model_dmn: str = "gpt-5",
        model_bpmn: str = "gpt-5",
        model_audit: str = "gpt-5",
        seed: Optional[int] = 42,
        canonical_config_path: Optional[str] = "chatchw/config/canonical_config.json",
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required")
        if OpenAI is None:
            raise RuntimeError("openai package not available")

        def _norm(m: str, default: str) -> str:
            if m and m.startswith("auto:"):
                m = m.split("auto:", 1)[1] or default
            if m in (None, "", "auto"):
                return default
            return m

        self.client = OpenAI(api_key=key)
        self.seed = seed
        self.models = dict(
            section=_norm(model_section, "gpt-5"),
            merge=_norm(model_merge,  "gpt-5"),
            dmn=_norm(model_dmn,      "gpt-5"),
            bpmn=_norm(model_bpmn,    "gpt-5"),
            audit=_norm(model_audit,  "gpt-5"),
        )

        self.debug_dir = Path(".chatchw_debug"); self.debug_dir.mkdir(exist_ok=True)

        # --- Load neutral canonical config (external, optional) ---
        default_canon_vars = [
            "symptom1","symptom1_duration_days",
            "symptom2","symptom2_duration_days",
            "sign1_present","sign2_present","sign3_present",
            "measurement1_rate","measurement2_temp_c","measurement3_mm",
            "age_months","area1_flag"
        ]
        try:
            cfg_path = Path(canonical_config_path)
            if cfg_path.exists():
                self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
            else:
                self.config = {
                    "canonical_variables": default_canon_vars,
                    "priority_tiers": {"hospital_min": 90, "clinic_min": 50},
                }
        except Exception:
            self.config = {
                "canonical_variables": default_canon_vars,
                "priority_tiers": {"hospital_min": 90, "clinic_min": 50},
            }

        canon = ", ".join(self.config.get("canonical_variables", default_canon_vars))

        # ---------------- PROMPTS (keep examples generic, OUTPUT may use medical terms) ----------------
        self.section_system = (
            "You extract structured decision rules from a technical manual. Return ONLY JSON.\n\n"
            "IMPORTANT OUTPUT POLICY\n"
            "- Preserve original clinical/medical terms (symptoms, signs, diseases, measurements, treatments) EXACTLY as written in the source text wherever they appear (variable names, allowed values, reasons, actions, guideline_ref). Do NOT anonymize or replace with placeholders.\n"
            "- Any examples in this prompt are GENERIC and illustrative of structure only; do NOT copy their wording or their placeholder variable names into the output.\n"
            "- Do not answer questions, do not add free-text clinical advice, and do not include any prose outside the JSON object.\n\n"
            "SCHEMA\n{\n  \"variables\":[{\"name\":snake,\"type\":\"number|boolean|string\",\"unit\":null|unit,\"allowed\":null|[...],"
            "                \"synonyms\":[snake...],\"prompt\":short,\"refs\":[page_or_section]}],\n"
            "  \"rules\":[{\"rule_id\":str,\n"
            "            \"when\":[ {\"obs\":var,\"op\":\"lt|le|gt|ge|eq|ne\",\"value\":num|bool|string} |\n"
            "                     {\"sym\":var,\"eq\":true|false} |\n"
            "                     {\"all_of\":[COND...]} | {\"any_of\":[COND...]} ],\n"
            "            \"then\":{\"triage\":\"hospital|clinic|home\",\n"
            "                    \"flags\":[snake...],\n"
            "                    \"reasons\":[snake...],\n"
            "                    \"actions\":[{\"id\":snake,\"if_available\":bool}],\n"
            "                    \"advice\":[],\n"
            "                    \"guideline_ref\":str,\"priority\":int}}],\n"
            "  \"canonical_map\":{},\n"
            "  \"qa\":{\"notes\":[]}\n}\n\n"
            f"RULES\n- Use canonical names if present: {canon} (these are EXAMPLES; prefer the source's actual medical terms).\n"
            "- Do NOT invent thresholds; encode only literal thresholds from the text.\n"
            "- No derived outputs in conditions (ban keys: danger_sign, clinic_referral, triage).\n"
            "- Severity policy (abstract): explicit critical_sign→triage:\"hospital\"; otherwise prefer clinic over home when uncertain.\n"
            "- Priority tiers: hospital≥90, clinic 50–89, home<50.\n"
            "- Every rule gets guideline_ref like \"p41\" or a section id.\n\n"
            "OUTPUT\nOnly the JSON object for THIS section."
        )

        # NOTE: this prompt now expects a compact FACT_SHEET instead of raw sections
        self.merge_system = (
            "Merge this FACT_SHEET (pre-extracted variables and rules) into one comprehensive IR. "
            "Return ONLY JSON with the same schema as step 1 plus:\n"
            "- canonical_map filled for all variables\n- qa.unmapped_vars\n- qa.dedup_dropped\n- qa.overlap_fixed\n"
            "RULES\n"
            "- Preserve original clinical/medical terms from inputs; do NOT replace with generic placeholders.\n"
            "- Normalize labels to snake_case where needed, but keep the medical meaning (e.g., \"acute_respiratory_distress\").\n"
            "- Rewrite all rules to canonical names (prefer existing variable names from inputs; do not invent placeholders).\n"
            "- Force then.advice to [] (empty).\n"
            "- If two rules conflict on the same conditions, tighten bounds or split any_of.\n"
            "- Keep every literal threshold.\n"
            "INPUT\nFACT_SHEET:\n<COMPACT VARIABLES + RULES>"
        )

        # NEW: Rules consolidation system prompt (fact sheet → unified RULES)
        self.rules_system = (
            "You are given a FACT_SHEET extracted from a clinical manual. "
            "Consolidate overlapping or duplicate facts into an organized list of RULES. "
            "Return ONLY JSON with:\n"
            "{ \"rules\": [ {\"rule_id\": str,\n"
            "               \"when\": [ {\"obs\": var, \"op\":\"lt|le|gt|ge|eq|ne\", \"value\": num|bool|string} |\n"
            "                         {\"sym\": var, \"eq\": true|false} |\n"
            "                         {\"all_of\":[COND...] } | {\"any_of\":[COND...]} ],\n"
            "               \"then\": {\"triage\":\"hospital|clinic|home\",\n"
            "                         \"flags\":[snake...],\n"
            "                         \"reasons\":[snake...],\n"
            "                         \"actions\":[{\"id\":snake,\"if_available\":bool}],\n"
            "                         \"advice\":[],\n"
            "                         \"guideline_ref\": str, \"priority\": int } } ] }\n\n"
            "HARD RULES\n"
            "1) Preserve the manual’s clinical terms exactly. Do not invent placeholders.\n"
            "2) Resolve overlaps: merge duplicate conditions, split any_of when needed, keep literal thresholds.\n"
            "3) Ban derived fields in conditions: danger_sign, clinic_referral, triage.\n"
            "4) advice must be [].\n"
            "5) Every rule must have guideline_ref like \"p41\" or a section id.\n\n"
            "INPUT\nFACT_SHEET:\n<variables + per-section rules>"
        )

        self.dmn_system = (
            "Convert RULES_JSON into modular DMN 1.4 using the DMN 1.4 MODEL namespace (2019-11-11). Return exactly TWO fenced blocks:\n"
            "1) ```xml <dmn:definitions>…```  2) ```json ASK_PLAN```\n\n"
            "HARD CONSTRAINTS\n"
            "- Use the variable names EXACTLY as they appear in RULES_JSON (they may be medical terms). Do not invent placeholders.\n"
            "- Any examples shown here are GENERIC and illustrative only; your output MUST use the actual variable names from RULES_JSON.\n"
            "- Advice must be an empty array in IR, empty string cells in DMN.\n"
            "- Use ONLY variables present in RULES_JSON.\n"
            "- Severity policy (abstract): critical_sign→hospital; prolonged_or_threshold_only→clinic; ambiguous→prefer clinic over home.\n\n"
            "DMN REQUIREMENTS\n"
            "- Root tag MUST be <dmn:definitions xmlns:dmn=\"https://www.omg.org/spec/DMN/20191111/MODEL/\">.\n"
            "- Decisions: decide_module_a, decide_module_b, decide_module_c, decide_module_d, decide_module_e, aggregate_final\n"
            "- Each module uses <dmn:decisionTable hitPolicy=\"FIRST\"> with outputs: triage:string, danger_sign:boolean, clinic_referral:boolean, reason:string, ref:string, advice:string\n"
            "- Ensure <dmn:inputData> exists for every variable present in RULES_JSON.\n"
            "- Advice column MUST be \"\" in outputEntry cells.\n"
            "- aggregate_final MUST use module boolean outputs directly (no string parsing):\n"
            "  Inputs: decide_module_b.danger_sign, decide_module_c.clinic_referral, decide_module_d.clinic_referral, decide_module_e.clinic_referral\n"
            "  Rows (FIRST):\n"
            "    1) decide_module_b.danger_sign = true → hospital (danger_sign=true, clinic_referral=true)\n"
            "    2) else if any clinic_referral = true → clinic (danger_sign=false, clinic_referral=true)\n"
            "    3) else → home (danger_sign=false, clinic_referral=false)\n\n"
            "ASK_PLAN (example is GENERIC; your output MUST reference the real variables from RULES_JSON):\n"
            "[\n"
            "  {\"module\":\"module_a\",\"ask\":[\"sign1_present\",\"sign2_present\",\"sign3_present\"]},\n"
            "  {\"module\":\"module_b\",\"ask\":[\"symptom1\"],\"followups_if\":{\"symptom1==true\":[\"measurement1_rate\",\"symptom1_duration_days\",\"age_months\"]}},\n"
            "  {\"module\":\"module_c\",\"ask\":[\"symptom2\"],\"followups_if\":{\"symptom2==true\":[\"symptom2_duration_days\"]}},\n"
            "  {\"module\":\"module_d\",\"ask\":[\"measurement2_temp_c\"],\"followups_if\":{}},\n"
            "  {\"module\":\"module_e\",\"ask\":[\"measurement3_mm\",\"area1_flag\"]}\n"
            "]\n"
        )

        self.bpmn_system = (
            "Produce one BPMN 2.0 <bpmn:definitions> only. Use xmlns:bpmn=\"http://www.omg.org/spec/BPMN/20100524/MODEL\" and xmlns:xsi.\n"
            "Process id=\"chatchw_flow\" isExecutable=\"false\"\n"
            "Start → userTask \"Ask module_b\" → XOR \"module_b present?\"\n true → userTask \"Collect module_b details\" → userTask \"Ask module_c\"\n false → userTask \"Ask module_c\"\n→ userTask \"Ask module_b followups\" → userTask \"Ask module_e\"\n→ businessRuleTask \"Evaluate decisions\" decisionRef=\"aggregate_final\"\n→ XOR \"Main triage\"\n"
            "  flow to \"Hospital\" with <conditionExpression xsi:type=\"tFormalExpression\">danger_sign == true</conditionExpression>\n"
            "  flow to \"Clinic\" with <conditionExpression xsi:type=\"tFormalExpression\">clinic_referral == true</conditionExpression>\n"
            "  default flow to \"Home\"\n"
            "Use variable names exactly as in RULES_JSON/DMN (these may be medical terms). Valid namespaces. No extra prose.\n"
        )

        self.coverage_system = (
            "Given RULES_JSON and the DMN XML, return ONLY JSON:\n"
            "{\"unmapped_rule_ids\":[...], \"module_counts\":{...}, \"notes\":[...]}\n"
            "Use RULES_JSON.ir_flat.conds to match DMN inputEntry texts literally when possible.\n"
            "A rule is mapped if its literal conditions appear as inputEntry cells in some module row with the same triage tier.\n"
            "Ignore text fields like reason/advice; match on conditions and triage only.\n"
            "INPUT\nRULES_JSON: <merged IR + ir_flat>\nDMN: <xml>"
        )

    # ---------------- PDF sectioning with multi-strategy fallbacks -----------------
    def extract_sections_from_pdf(self, pdf_path: str, max_chars: int = 4000) -> List[Tuple[str, str]]:
        """
        Return a list of (section_id, text) with page markers retained.
        Strategy:
          1) pypdf extraction
          2) pdfminer layout-aware per-page (if available)
          3) pdfminer basic per-page
          4) OCR (pytesseract + pdf2image) per-page (if available)
        """
        logp = self.debug_dir / "sectioning.log"

        # 1) pypdf
        pages = _extract_pypdf_pages(pdf_path)
        pages = _postprocess_page_text(pages)
        nonempty = sum(1 for _, t in pages if (t or "").strip())
        if nonempty > 0:
            chunks = _split_into_sections_by_headings(pages, max_chars=max_chars)
            if not chunks:
                chunks = _chunk_pages_len_only(pages, max_chars=max_chars)
            try: logp.write_text("Used pypdf extraction\n", encoding="utf-8")
            except Exception: pass
            return chunks

        # 2) pdfminer layout-aware
        if _have_pdfminer:
            pm_layout = _extract_pdfminer_layout_pages(pdf_path)
            pm_layout = _postprocess_page_text(pm_layout)
            if sum(1 for _, t in pm_layout if (t or "").strip()) > 0:
                chunks = _split_into_sections_by_headings(pm_layout, max_chars=max_chars)
                if not chunks:
                    chunks = _chunk_pages_len_only(pm_layout, max_chars=max_chars)
                try: logp.write_text("Used pdfminer layout extraction\n", encoding="utf-8")
                except Exception: pass
                return chunks

        # 3) pdfminer basic
        if _have_pdfminer:
            pm_pages = _extract_pdfminer_pages(pdf_path)
            pm_pages = _postprocess_page_text(pm_pages)
            if sum(1 for _, t in pm_pages if (t or "").strip()) > 0:
                chunks = _split_into_sections_by_headings(pm_pages, max_chars=max_chars)
                if not chunks:
                    chunks = _chunk_pages_len_only(pm_pages, max_chars=max_chars)
                try: logp.write_text("Used pdfminer basic extraction\n", encoding="utf-8")
                except Exception: pass
                return chunks

        # 4) OCR
        if _have_ocr:
            ocr_pages = _extract_ocr_pages(pdf_path)
            ocr_pages = _postprocess_page_text(ocr_pages)
            if sum(1 for _, t in ocr_pages if (t or "").strip()) > 0:
                chunks = _split_into_sections_by_headings(ocr_pages, max_chars=max_chars)
                if not chunks:
                    chunks = _chunk_pages_len_only(ocr_pages, max_chars=max_chars)
                try: logp.write_text("Used OCR extraction (pytesseract + pdf2image)\n", encoding="utf-8")
                except Exception: pass
                return chunks

        # Nothing worked
        try:
            logp.write_text("No text extracted by any strategy\n", encoding="utf-8")
        except Exception:
            pass
        return []

    # ---------------- OpenAI helpers (GPT-5 safe) -----------------
    def _complete(self, model_name: str, messages: list, max_out: int):
        # Build a base payload that is compatible with both GPT-5 and older models.
        base = dict(model=model_name, messages=messages)

        # For GPT-5 models: do NOT force temperature/top_p (some 5-series reject overrides).
        # For older models: we keep deterministic knobs.
        is_gpt5 = isinstance(model_name, str) and model_name.startswith("gpt-5")
        if not is_gpt5:
            base.update(dict(temperature=0.0, top_p=1))

        # Try GPT-5 token param first; gracefully fall back to legacy.
        extras = {}
        if is_gpt5:
            extras = {"reasoning_effort": "minimal", "verbosity": "low"}
        # Include seed only initially; if rejected, we drop it in fallbacks.
        base_seed = dict(base); base_seed["seed"] = getattr(self, "seed", None)

        # Try: gpt-5 style with max_completion_tokens
        try:
            return self.client.chat.completions.create(
                max_completion_tokens=max_out,
                **base_seed, **extras
            )
        except Exception as e1:
            msg1 = str(e1).lower()

            # Retry: remove optional extras if they are not supported
            if "verbosity" in msg1 or "reasoning" in msg1 or "unsupported parameter" in msg1:
                try:
                    return self.client.chat.completions.create(
                        max_completion_tokens=max_out,
                        **base_seed
                    )
                except Exception as e2:
                    msg2 = str(e2).lower()
                    # Fallback to legacy max_tokens if max_completion_tokens unsupported
                    if "max_completion_tokens" in msg2 or "unsupported parameter" in msg2:
                        try:
                            return self.client.chat.completions.create(
                                max_tokens=max_out,
                                **base_seed
                            )
                        except Exception as e3:
                            msg3 = str(e3).lower()
                            # Last attempt: drop seed if it's rejected
                            if "seed" in msg3 or "unsupported parameter" in msg3:
                                base_noseed = dict(base)
                                return self.client.chat.completions.create(
                                    max_tokens=max_out,
                                    **base_noseed
                                )
                            raise
                    raise

            # If the first error complained about max_completion_tokens directly, try legacy
            if "max_completion_tokens" in msg1:
                try:
                    return self.client.chat.completions.create(
                        max_tokens=max_out,
                        **base_seed, **extras
                    )
                except Exception as e4:
                    msg4 = str(e4).lower()
                    if "seed" in msg4 or "unsupported parameter" in msg4:
                        base_noseed = dict(base)
                        return self.client.chat.completions.create(
                            max_tokens=max_out,
                            **base_noseed, **extras
                        )
                    raise
            raise

    def _chat_json(self, system: str, user: str, max_out: int = 6000, model_key: str = "section", schema_model=None) -> Dict[str, Any]:
        model_name = self.models.get(model_key, self.models["section"])
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def _call():
            resp = self._complete(model_name, messages, max_out)
            return (resp.choices[0].message.content or "").strip()

        try:
            return _call_json_with_retries(_call, schema_model=schema_model, retries=3)
        except Exception as e:
            # truncate huge debug blobs so we can actually read failures
            dbg_user = user
            max_head = 12000
            max_tail = 4000
            if len(dbg_user) > (max_head + max_tail):
                dbg_user = dbg_user[:max_head] + "\n...<TRUNCATED>...\n" + dbg_user[-max_tail:]
            Path(self.debug_dir / "guarded_debug_last.txt").write_text(
                f"SYSTEM:\n{system}\n\nUSER (len={len(user)}):\n{dbg_user}\n\nERR:{e}", encoding="utf-8"
            )
            raise

    def _chat_text(self, system: str, user: str, max_out: int = 12000, model_key: str = "dmn") -> str:
        model_name = self.models.get(model_key, self.models["dmn"])
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        resp = self._complete(model_name, messages, max_out)
        return (resp.choices[0].message.content or "").strip()

    # ---------------- Step 1: per-section -----------------
    def extract_rules_per_section(self, pdf_path: str) -> List[Dict[str, Any]]:
        sections = self.extract_sections_from_pdf(pdf_path)
        results: List[Dict[str, Any]] = []
        for sec_id, text in sections:
            user = f"SECTION_ID: {sec_id}\n\nTEXT:\n{text}"
            try:
                obj = self._chat_json(
                    self.section_system, user,
                    max_out=6000, model_key="section",
                    schema_model=IR if PydanticAvailable else None
                )
            except Exception:
                continue
            cleaned = []
            for r in obj.get("rules", []) or []:
                nr = _normalize_rule_schema(r)
                if nr: cleaned.append(nr)
                else: obj.setdefault("qa", {}).setdefault("notes", []).append(f"dropped_rule:{r.get('rule_id')}")
            obj["rules"] = _dedupe_rule_ids(cleaned, prefix=f"{sec_id}")
            results.append(obj)
        return results

    # ---------------- NEW: Step 2a — build unified RULES from FACT_SHEET -----------------
    def generate_rules_from_fact_sheet(self, fact_sheet: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Ask GPT to consolidate overlaps and produce a single RULES list from the compact FACT_SHEET.
        Normalizes with _normalize_rule_schema and de-dupes ids. Falls back to FACT_SHEET rules if model fails.
        """
        payload = json.dumps(fact_sheet, ensure_ascii=False)
        user = f"FACT_SHEET\n{payload}"

        try:
            obj = self._chat_json(
                self.rules_system, user,
                max_out=8000,
                model_key="merge",  # reuse merge allocation
                schema_model=None
            )
            rules = obj.get("rules") if isinstance(obj, dict) else obj
            if not isinstance(rules, list):
                raise ValueError("rules consolidation returned unexpected shape")
            cleaned = []
            for r in rules:
                nr = _normalize_rule_schema(r)
                if nr:
                    cleaned.append(nr)
            rules_out = _dedupe_rule_ids(cleaned, prefix="rl")
        except Exception as e:
            # Safe fallback so pipeline still runs
            rules_out = []
            for r in (fact_sheet.get("rules") or []):
                nr = _normalize_rule_schema(r)
                if nr:
                    rules_out.append(nr)
            rules_out = _dedupe_rule_ids(rules_out, prefix="fs")
            try:
                self.debug_dir.joinpath("rules_from_facts_error.txt").write_text(str(e), encoding="utf-8")
            except Exception:
                pass

        # Persist for inspection
        try:
            self.debug_dir.joinpath("rules_from_facts.json").write_text(
                json.dumps({"rules": rules_out}, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

        return rules_out

    # ---------------- Step 2: merge (now drives off FACT_SHEET) -----------------
    def merge_sections(self, section_objs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 0) Build compact fact sheet and save for debug
        fact_sheet = _fact_sheet_from_sections(section_objs)
        try:
            self.debug_dir.joinpath("fact_sheet.json").write_text(
                json.dumps(fact_sheet, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

        # 1) NEW: Ask GPT to consolidate facts into a canonical RULES list
        consolidated_rules = self.generate_rules_from_fact_sheet(fact_sheet)

        # 2) Merge variables/QA with model (or fallback), but force rules to the consolidated list
        payload = json.dumps(fact_sheet, ensure_ascii=False)
        user = f"FACT_SHEET\n{payload}"

        if STRICT_MERGE:
            merged = self._chat_json(
                self.merge_system, user,
                max_out=8000,  # allow a bit more room here
                model_key="merge",
                schema_model=IR if PydanticAvailable else None
            )
            # Overwrite rules with the consolidated ones from fact sheet
            merged["rules"] = consolidated_rules
            # If model didn't emit variables, take from fact sheet
            if not merged.get("variables"):
                merged["variables"] = fact_sheet.get("variables", [])
        else:
            # Deterministic local fallback using the compact fact sheet
            variables: Dict[str, Dict[str, Any]] = {}
            qa_notes: List[str] = ["fallback_merge_used", "source=fact_sheet"]
            for v in (fact_sheet.get("variables") or []):
                if not isinstance(v, dict): continue
                name = str(v.get("name", "")).strip()
                if not name: continue
                cur = variables.get(name.lower())
                if cur is None:
                    vv = dict(v)
                    if not isinstance(vv.get("synonyms"), list): vv["synonyms"] = []
                    if not isinstance(vv.get("refs"), list): vv["refs"] = []
                    variables[name.lower()] = vv
                else:
                    syn = set(cur.get("synonyms") or []) | set(v.get("synonyms") or [])
                    refs = set(cur.get("refs") or []) | set(v.get("refs") or [])
                    cur["synonyms"] = sorted(list(syn))
                    cur["refs"] = sorted(list(refs))
            merged = {
                "variables": list(variables.values()),
                "rules": consolidated_rules,
                "canonical_map": {},
                "qa": {"notes": qa_notes, "unmapped_vars": [], "dedup_dropped": 0, "overlap_fixed": []},
            }

        # ---- Resolve duplicates & snake-case consistently, then rewrite rules ----
        merged = _resolve_variables_snake_and_rewrite_rules(merged)

        # Canonical map + rewrite to canonical names (still keeps medical wording)
        canon = _build_canonical_map(self.config, merged.get("variables", []))
        merged["canonical_map"] = canon
        merged["rules"] = [_rewrite_rule_to_canon(r, canon) for r in (merged.get("rules") or [])]
        merged["rules"] = _dedupe_rule_ids(merged["rules"], prefix="merged")

        # Validate
        if PydanticAvailable:
            merged = IR.model_validate(merged).model_dump()
        _preflight_ir(merged)
        return merged

    # ---------------- Step 3: DMN + ASK_PLAN -----------------
    def generate_dmn_and_ask_plan(self, merged_ir: Dict[str, Any]) -> Tuple[str, Any]:
        user = "RULES_JSON:\n" + json.dumps(merged_ir, ensure_ascii=False)
        text = self._chat_text(self.dmn_system, user, max_out=12000, model_key="dmn")

        try:
            Path(self.debug_dir / "dmn_ask_debug_last.txt").write_text(text, encoding="utf-8")
        except Exception:
            pass

        blocks = _extract_fenced_blocks(text)
        dmn_xml: Optional[str] = None
        ask_plan: Optional[Any] = None

        def _parse_json_loose(s: str) -> Optional[Any]:
            try:
                return _loose_to_json(s)
            except Exception:
                return None

        for _lang, body in blocks:
            b = body.strip()
            if dmn_xml is None and "<dmn:definitions" in b and "</dmn:definitions>" in b:
                dmn_xml = b; continue
            if ask_plan is None:
                parsed = _parse_json_loose(b)
                if isinstance(parsed, (list, dict)):
                    ask_plan = parsed

        if dmn_xml is None:
            s = text
            start = s.find("<dmn:definitions"); end = s.find("</dmn:definitions>")
            if start != -1 and end != -1 and end > start:
                end += len("</dmn:definitions>")
                dmn_xml = s[start:end].strip()

        if ask_plan is None:
            parsed = _parse_json_loose(text)
            if isinstance(parsed, (list, dict)):
                ask_plan = parsed

        def _sanitize_dmn(xml: str) -> str:
            s = xml
            # Ensure closing tags on entries
            s = re.sub(r"(<dmn:outputEntry>\s*<dmn:text>)([^<]*?)(</dmn:outputEntry>)", r"\1\2</dmn:text>\3", s, flags=re.DOTALL)
            s = s.replace("<dmn:outputEntry><dmn:text></dmn:text></dmn:outputEntry>", "<dmn:outputEntry><dmn:text>\"\"</dmn:text></dmn:outputEntry>")
            s = s.replace("<dmn:outputEntry><dmn:text></dmn:outputEntry>", "<dmn:outputEntry><dmn:text>\"\"</dmn:text></dmn:outputEntry>")
            s = re.sub(r"(<dmn:inputEntry>\s*<dmn:text>)([^<]*?)(</dmn:inputEntry>)", r"\1\2</dmn:text>\3", s, flags=re.DOTALL)
            s = s.replace("<dmn:inputEntry><dmn:text></dmn:text></dmn:inputEntry>", "<dmn:inputEntry><dmn:text>-</dmn:text></dmn:inputEntry>")
            s = s.replace("<dmn:inputEntry><dmn:text></dmn:inputEntry>", "<dmn:inputEntry><dmn:text>-</dmn:text></dmn:inputEntry>")
            # Convert non-FEEL 'otherwise' to wildcard '-'
            s = re.sub(r'(<dmn:inputEntry>\s*<dmn:text>)\s*otherwise\s*(</dmn:text>\s*</dmn:inputEntry>)', r'\1-\2', s, flags=re.IGNORECASE)
            # Namespace
            if 'xmlns:dmn="' not in s:
                s = s.replace("<dmn:definitions", '<dmn:definitions xmlns:dmn="https://www.omg.org/spec/DMN/20191111/MODEL/"', 1)
            return s

        if dmn_xml:
            dmn_xml = _sanitize_dmn(dmn_xml)

        if not dmn_xml or ask_plan is None:
            try:
                Path(self.debug_dir / "dmn_ask_debug_last.txt").write_text(text, encoding="utf-8")
            except Exception:
                pass
            raise RuntimeError("Failed to get DMN and ASK_PLAN from model")

        # Normalize ask_plan to a list of blocks; drop unknown variables safely
        try:
            known_vars = {v["name"] for v in merged_ir.get("variables", []) if isinstance(v, dict) and v.get("name")}
            mutated: List[Dict[str, Any]] = []
            unknowns = set()
            # Accept either a list or an object with ASK_PLAN
            if isinstance(ask_plan, dict) and "ASK_PLAN" in ask_plan:
                ask_plan = ask_plan.get("ASK_PLAN") or []
            if isinstance(ask_plan, list):
                for blk in ask_plan:
                    if not isinstance(blk, dict): continue
                    ask = [q for q in (blk.get("ask") or []) if q in known_vars]
                    fuw: Dict[str, List[str]] = {}
                    for cond, qs in (blk.get("followups_if", {}) or {}).items():
                        kept = [q for q in (qs or []) if q in known_vars]
                        fuw[cond] = kept
                        for q in (qs or []):
                            if q not in known_vars: unknowns.add(q)
                    for q in (blk.get("ask") or []):
                        if q not in known_vars: unknowns.add(q)
                    mutated.append({"module": blk.get("module"), "ask": ask, "followups_if": fuw})
                ask_plan = mutated
            if unknowns:
                merged_ir.setdefault("qa", {}).setdefault("notes", []).append(f"ask_plan_unknowns_dropped:{sorted(list(unknowns))}")
            ask_plan = _enforce_ask_ownership(ask_plan)
        except Exception:
            pass

        return dmn_xml, ask_plan

    # ---------------- Step 4: BPMN -----------------
    def generate_bpmn(self, dmn_xml: str, ask_plan: Any) -> str:
        user = "DMN:\n```xml\n" + dmn_xml + "\n```\n\n" + "ASK_PLAN:\n" + json.dumps(ask_plan, ensure_ascii=False)
        text = self._chat_text(self.bpmn_system, user, max_out=12000, model_key="bpmn")

        # 1) Preferred: fenced block containing <bpmn:definitions>…</bpmn:definitions>
        blocks = _extract_fenced_blocks(text)
        for _lang, body in blocks:
            if "<bpmn:definitions" in body:
                return _sanitize_bpmn(body)

        # 2) Raw scrape: namespaced tag
        xml = _extract_xml_tag(text, "bpmn:definitions")
        if xml:
            return _sanitize_bpmn(xml)

        # 3) Raw scrape: non-namespaced <definitions>…</definitions> → normalize to bpmn:
        xml = _extract_xml_tag(text, "definitions")
        if xml:
            xml = re.sub(r"<\s*definitions\b", "<bpmn:definitions", xml, count=1)
            xml = re.sub(r"</\s*definitions\s*>", "</bpmn:definitions>", xml, count=1)
            return _sanitize_bpmn(xml)

        # 4) Debug-friendly dump (truncated if huge), then fail
        try:
            dbg = text
            if len(dbg) > 24000:
                dbg = dbg[:16000] + "\n...<TRUNCATED>...\n" + dbg[-6000:]
            Path(self.debug_dir / "bpmn_debug_last.txt").write_text(dbg, encoding="utf-8")
        except Exception:
            pass
        raise RuntimeError("Failed to get BPMN from model")

    # ---------------- Step 5: coverage -----------------
    def audit_coverage(self, merged_ir: Dict[str, Any], dmn_xml: str) -> Dict[str, Any]:
        payload = {"ir": merged_ir, "ir_flat": _rules_flattened(merged_ir)}
        user = "RULES_JSON:\n" + json.dumps(payload, ensure_ascii=False) + "\nDMN:\n```xml\n" + dmn_xml + "\n```"
        return self._chat_json(self.coverage_system, user, max_out=6000, model_key="audit", schema_model=None)
