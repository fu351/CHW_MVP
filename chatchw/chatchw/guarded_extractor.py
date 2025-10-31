from __future__ import annotations

import collections
import csv
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------- PDF libs (we'll try multiple strategies) ----------
import pypdf

# Optional fallbacks (imported lazily where used)
_have_pdfminer = True
try:
    # import softly so the module can still run without it
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

# ---------------- JSON repair and retries ----------------
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
    s = re.sub(r",\s*(]|})", r"\1", s)
    # crop to first obj or array
    for opener, closer in [("{", "}"), ("[", "]")]:
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
            1,
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
            triage = th[k]
            break
    if triage is not None:
        th["triage"] = triage

    flags = None
    for k in _CANON_FLAG_KEYS:
        if th.get(k) is not None:
            flags = th[k]
            break
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
        if not isinstance(cond, dict):
            return None
        if _has_bad_sym(cond):
            return None
        if "all_of" in cond or "any_of" in cond:
            seq = cond.get("all_of") or cond.get("any_of") or []
            for sub in seq:
                if _has_bad_sym(sub):
                    return None
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
    by_name = {(v.get("name") or "").strip().lower(): v for v in variables if isinstance(v, dict)}
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
                if "sym" in s:
                    s["sym"] = map_var(s["sym"])
                if "obs" in s:
                    s["obs"] = map_var(s["obs"])
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
            if s:
                out.append(s)
    return out


def _rules_flattened(ir: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "rule_id": r.get("rule_id"),
            "triage": (r.get("then") or {}).get("triage"),
            "conds": _flatten_rule_conditions(r),
        }
        for r in ir.get("rules", [])
    ]


def _preflight_ir(ir: Dict[str, Any]) -> None:
    names = [(v.get("name") or "").strip().lower() for v in ir.get("variables", []) if isinstance(v, dict)]
    dup_var_names = [n for n, cnt in collections.Counter(names).items() if cnt > 1]
    if dup_var_names:
        raise RuntimeError(f"Duplicate variable names after merge: {dup_var_names}")
    ids = [(r.get("rule_id") or "").strip().lower() for r in ir.get("rules", []) if isinstance(r, dict)]
    dups = [i for i, c in collections.Counter(ids).items() if c > 1]
    if dups:
        raise RuntimeError(f"Duplicate rule_ids after dedupe pass: {dups}")
    empties = [r.get("rule_id") for r in ir.get("rules", []) if not (r.get("when") or [])]
    if empties:
        raise RuntimeError(f"Empty WHEN in rules: {empties}")


def _enforce_ask_ownership(
    ask_plan: List[Dict[str, Any]],
    priority_order=("module_a", "module_b", "module_c", "module_d", "module_e"),
):
    owner: Dict[str, str] = {}
    order = {m: i for i, m in enumerate(priority_order)}
    for blk in sorted([b for b in ask_plan if isinstance(b, dict)], key=lambda b: order.get(b.get("module", "zzz"), 999)):
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

# ---------- Duplicate resolution plus snake case, with rule rewriting ----------
def _norm_name(s: Optional[str]) -> str:
    return (s or "").strip()


def _pages_from_ref_string(s: str) -> List[int]:
    if not s:
        return []
    out = []
    for tok in re.findall(r"p(\d{1,4})(?:-(\d{1,4}))?", s, flags=re.IGNORECASE):
        a = int(tok[0])
        b = int(tok[1]) if tok[1] else None
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
    1) Group variables by snake_case(name) so near duplicates collide.
    2) Within each group, partition by signature (type/unit/allowed).
    3) Assign final names, then rewrite rules consistently.
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
        syns = set(vv.get("synonyms") or [])
        syns.add(orig)
        vv["synonyms"] = sorted(list(syns))
        groups.setdefault(key, []).append(vv)
        names_seen_per_group.setdefault(key, set()).update({orig.lower(), *[str(s).strip().lower() for s in syns]})

    new_vars: List[Dict[str, Any]] = []
    name_to_variants: Dict[str, List[Dict[str, Any]]] = {}
    rename_events: Dict[str, List[str]] = {}

    for base, gvars in groups.items():
        sig_bins: Dict[tuple, List[Dict[str, Any]]] = {}
        for v in gvars:
            sig_bins.setdefault(_var_sig(v), []).append(v)

        variants_meta = []
        if len(sig_bins) > 1:
            qa_notes.append(f"variable_homonyms_split:{base}:{len(sig_bins)}")

        idx = 1
        for _sig, same_sig_vars in sig_bins.items():
            merged_v = _coalesce_group(same_sig_vars)
            final_name = base if idx == 1 else f"{base}__{idx}"
            idx += 1
            originals = {_norm_name(x.get("name")) for x in same_sig_vars if _norm_name(x.get("name"))}
            for o in originals:
                rename_events.setdefault(o, []).append(final_name)
            merged_v["name"] = final_name
            pages = _pages_from_refs(merged_v.get("refs") or [])
            variants_meta.append({"new_name": final_name, "pages": pages})
            new_vars.append(merged_v)

        for n in names_seen_per_group.get(base, set()):
            name_to_variants[n] = variants_meta

    def pick_variant(seen_name: str, rule_ref: Optional[str]) -> str:
        variants = name_to_variants.get((seen_name or "").strip().lower())
        if not variants:
            return _to_snake(seen_name)
        if not rule_ref:
            return variants[0]["new_name"]
        r_pages = set(_pages_from_ref_string(rule_ref))
        best = variants[0]
        best_sc = -1
        for v in variants:
            sc = len(r_pages & set(v["pages"]))
            if sc > best_sc:
                best_sc = sc
                best = v
        return best["new_name"]

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

    if rename_events:
        merged.setdefault("qa", {}).setdefault("notes", []).append(f"snake_case_renamed:{rename_events}")

    merged["variables"] = new_vars
    merged["rules"] = rules_out
    return merged

# ---------- Fact sheet helpers ----------
def _fact_sheet_from_sections(section_objs: List[Dict[str, Any]]) -> Dict[str, Any]:
    var_ix: Dict[str, Dict[str, Any]] = {}
    facts_rules: List[Dict[str, Any]] = []
    for sec in (section_objs or []):
        for v in (sec.get("variables") or []):
            if not isinstance(v, dict):
                continue
            name = (v.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            cur = var_ix.get(key)
            if cur is None:
                cur = dict(
                    name=name,
                    type=v.get("type"),
                    unit=v.get("unit"),
                    allowed=v.get("allowed"),
                    synonyms=list(v.get("synonyms") or []),
                    refs=list(v.get("refs") or []),
                )
                var_ix[key] = cur
            else:
                cur["synonyms"] = sorted(list(set((cur.get("synonyms") or []) + (v.get("synonyms") or []))))
                cur["refs"] = sorted(list(set((cur.get("refs") or []) + (v.get("refs") or []))))
        for r in (sec.get("rules") or []):
            if not isinstance(r, dict):
                continue
            nr = _normalize_rule_schema(r)
            if not nr:
                continue
            th = nr.get("then") or {}
            facts_rules.append(
                {
                    "rule_id": r.get("rule_id"),
                    "when": nr.get("when") or [],
                    "then": {
                        "triage": th.get("triage"),
                        "flags": th.get("flags") or [],
                        "reasons": th.get("reasons") or [],
                        "actions": th.get("actions") or [],
                        "guideline_ref": th.get("guideline_ref"),
                        "priority": int(th.get("priority") or 0),
                        "advice": [],  # enforce empty
                    },
                }
            )
    return {
        "variables": list(var_ix.values()),
        "rules": _dedupe_rule_ids(facts_rules, prefix="fs"),
        "qa": {"notes": ["from_fact_sheet_builder"]},
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
    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            n = len(reader.pages)
    except Exception:
        n = 0
    if n <= 0:
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
        out.append((i + 1, txt.strip()))
    return out


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


_heading_rx = re.compile(
    r"^(annex|appendix|module|chapter|section|part|lesson|unit|task|topic)\b"
    r"|\b(assessment|treatment|referral|follow\s*up|training)\b"
    r"|^[A-Z][A-Z0-9 ,:/-]{6,}$",
    flags=re.IGNORECASE,
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
            tops.append("")
            bottoms.append("")
            continue
        tops.append(lines[0])
        bottoms.append(lines[-1])

    def common(items):
        ctr = collections.Counter(items)
        if not ctr:
            return None
        item, cnt = ctr.most_common(1)[0]
        return item if cnt >= max(3, int(0.4 * len(items))) and len(item) >= 6 else None

    top_c = common(tops)
    bot_c = common(bottoms)
    cleaned: List[Tuple[int, str]] = []
    for i, t in pages:
        if not t:
            cleaned.append((i, t))
            continue
        lines = t.splitlines()
        if top_c and lines and lines[0].strip() == top_c:
            lines = lines[1:]
        if bot_c and lines and lines[-1].strip() == bot_c:
            lines = lines[:-1]
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
    buf = ""
    ids: List[int] = []
    for i, t in pages:
        if not t:
            continue
        add = f"[PAGE {i}]\n{t}\n"
        if len(buf) + len(add) > max_chars and buf:
            chunks.append((f"p{ids[0]}-{ids[-1]}", buf))
            buf = ""
            ids = []
        buf += add
        ids.append(i)
    if buf:
        chunks.append((f"p{ids[0]}-{ids[-1]}", buf))
    return chunks

# ---------------- DMN parsing plus CSV and XLS wiring helpers ----------------
_DMNN = {"dmn": "https://www.omg.org/spec/DMN/20191111/MODEL/"}


def _text_or(el):
    return (el.text or "").strip() if el is not None and el.text else ""


def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def _feel_atom_to_xpath(val: str, var_type: str) -> Optional[str]:
    t = (var_type or "").lower()
    v = (val or "").strip()
    if v.lower() in ("-", "otherwise", ""):
        return None
    if v.lower() in ("true", "false"):
        # For user entered booleans we use select_one yes_no -> 'yes' or 'no'
        return "'yes'" if v.lower() == "true" else "'no'"
    if re.match(r"^-?\d+(\.\d+)?$", v):
        return v
    return f"'{_strip_quotes(v)}'"


def _and_join(parts: List[str]) -> str:
    parts = [p for p in parts if p and p.strip()]
    if not parts:
        return "true()"
    if len(parts) == 1:
        return parts[0]
    # Join multiple parts with logical AND inside parentheses
    return "(" + ") and (".join(parts) + ")"

# ---------- DMN output validator (fail fast on blanks or invalids) ----------
def _validate_dmn_outputs_or_die(xml: str) -> None:
    root = ET.fromstring(xml)
    required = ["triage", "danger_sign", "clinic_referral", "reason", "ref", "advice"]
    for dec in root.findall(".//dmn:decision", _DMNN):
        name = (dec.get("name") or dec.get("id") or "decision").strip()
        table = dec.find(".//dmn:decisionTable", _DMNN)
        if table is None:
            raise RuntimeError(f"{name}: missing dmn:decisionTable")
        outs = table.findall("./dmn:output", _DMNN)
        have = [(o.get("name") or "").strip() for o in outs]
        missing = [r for r in required if r not in have]
        if missing:
            raise RuntimeError(f"{name}: missing required outputs {missing}")
        rules = table.findall("./dmn:rule", _DMNN)
        if not rules:
            raise RuntimeError(f"{name}: has zero rules")
        for i, r in enumerate(rules, start=1):
            vals = [_text_or(e.find("./dmn:text", _DMNN)) for e in r.findall("./dmn:outputEntry", _DMNN)]
            if len(vals) < len(outs):
                vals += [""] * (len(outs) - len(vals))
            triage, danger, clinic, reason, ref, advice = (vals + [""] * 6)[:6]
            if triage.strip().strip('"') not in ("hospital", "clinic", "home"):
                raise RuntimeError(f"{name}: rule {i} invalid triage: {triage!r}")
            if danger.strip().lower() not in ("true", "false"):
                raise RuntimeError(f"{name}: rule {i} danger_sign not boolean: {danger!r}")
            if clinic.strip().lower() not in ("true", "false"):
                raise RuntimeError(f"{name}: rule {i} clinic_referral not boolean: {clinic!r}")
            if not ref.strip():
                raise RuntimeError(f"{name}: rule {i} ref is empty (need a page or section ref)")


def _ensure_dmn_inputs(dmn_xml: str, ask_plan: Any, merged_ir: Dict[str, Any]) -> str:
    """
    Ensure that every decision table in the DMN has <dmn:input> definitions.
    The DMN generated by the model may omit these elements, which prevents
    the XLSForm wiring logic from matching conditions to variables.  This
    helper creates input elements for each decision table, using the ask_plan
    ordering to choose variable names when available, or by extracting
    variable names from FEEL expressions.  It also sets the typeRef
    (number|boolean|string) based on merged_ir variable definitions.
    """
    try:
        root = ET.fromstring(dmn_xml)
    except Exception:
        return dmn_xml

    # Build type lookup
    var_types: Dict[str, str] = {}
    for v in (merged_ir.get("variables") or []):
        if not isinstance(v, dict):
            continue
        name = (v.get("name") or "").strip()
        if not name:
            continue
        typ = (v.get("type") or "string").strip().lower()
        var_types[name] = typ

    # Normalize ask_plan by module
    plan_map: Dict[str, List[str]] = {}
    if isinstance(ask_plan, list):
        for blk in ask_plan:
            if not isinstance(blk, dict):
                continue
            mod = (blk.get("module") or "").strip()
            if not mod:
                continue
            ask_list = [q for q in (blk.get("ask") or []) if q]
            plan_map[mod] = ask_list

    DMN_NS = _DMNN.get("dmn")
    if not DMN_NS:
        return dmn_xml

    for dec in root.findall(f".//{{{DMN_NS}}}decision"):
        name = (dec.get("name") or dec.get("id") or "").strip()
        mod  = re.sub(r"^decide[_\\-:]+", "", name) or "module"
        table = dec.find(f".//{{{DMN_NS}}}decisionTable")
        if table is None:
            continue

        # Determine the maximum number of input conditions across rules
        rules = table.findall(f"{{{DMN_NS}}}rule")
        max_conds = 0
        for r in rules:
            conds = r.findall(f"{{{DMN_NS}}}inputEntry")
            max_conds = max(max_conds, len(conds))

        # Choose input names: prefer ask_plan; fallback to extracting from FEEL
        input_names: List[str] = []
        ask_list = plan_map.get(mod)
        if ask_list:
            input_names = ask_list[:max_conds]
        if not input_names:
            # Extract variable names from the first rule's conditions
            if rules:
                first_rule = rules[0]
                for idx in range(max_conds):
                    try:
                        inp_entry = first_rule.findall(f"{{{DMN_NS}}}inputEntry")[idx]
                    except Exception:
                        continue
                    text_node = inp_entry.find(f"{{{DMN_NS}}}text")
                    expr = (text_node.text or "").strip() if text_node is not None else ""
                    var_name = None
                    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_\.:-]*)\s*(?:[=!<>]=|[<>])", expr)
                    if m:
                        var_tok = m.group(1).strip()
                        # remove prefix if dot
                        if "." in var_tok:
                            var_tok = var_tok.split(".")[-1]
                        var_name = var_tok
                    if not var_name:
                        var_name = f"input_{idx+1}"
                    input_names.append(var_name)
            # fallback: generic names if still empty
            if not input_names and max_conds > 0:
                input_names = [f"input_{i+1}" for i in range(max_conds)]

        # Rebuild <dmn:input> definitions if the counts mismatch
        existing_inputs = table.findall(f"{{{DMN_NS}}}input")
        if len(existing_inputs) != max_conds:
            for inp_el in existing_inputs:
                table.remove(inp_el)
            for idx, in_name in enumerate(input_names):
                inp_el = ET.Element(f"{{{DMN_NS}}}input")
                typ = var_types.get(in_name, "string")
                inp_expr = ET.SubElement(inp_el, f"{{{DMN_NS}}}inputExpression", {"typeRef": typ})
                text_el = ET.SubElement(inp_expr, f"{{{DMN_NS}}}text")
                text_el.text = in_name
                table.insert(idx, inp_el)

    return ET.tostring(root, encoding="unicode")

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
        # Allow running with --dmn/--merged only, no API needed
        if not key and not os.getenv("SKIP_OPENAI_CHECK"):
            pass
        if OpenAI is None and key:
            raise RuntimeError("openai package not available")

        def _norm(m: str, default: str) -> str:
            if m and m.startswith("auto:"):
                m = m.split("auto:", 1)[1] or default
            if m in (None, "", "auto"):
                return default
            return m

        self.client = OpenAI(api_key=key) if key and OpenAI else None
        self.seed = seed
        self.models = dict(
            section=_norm(model_section, "gpt-5"),
            merge=_norm(model_merge, "gpt-5"),
            dmn=_norm(model_dmn, "gpt-5"),
            bpmn=_norm(model_bpmn, "gpt-5"),
            audit=_norm(model_audit, "gpt-5"),
        )

        # usage counters
        self.usage = {"prompt": 0, "completion": 0, "total": 0}
        self._usage_events: List[Dict[str, int]] = []

        self.debug_dir = Path(".chatchw_debug")
        self.debug_dir.mkdir(exist_ok=True)

        # canonical config
        default_canon_vars = [
            "symptom1",
            "symptom1_duration_days",
            "symptom2",
            "symptom2_duration_days",
            "sign1_present",
            "sign2_present",
            "sign3_present",
            "measurement1_rate",
            "measurement2_temp_c",
            "measurement3_mm",
            "age_months",
            "area1_flag",
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

        # ---------------- prompts ----------------
        self.section_system = (
            "You extract structured decision rules from a technical manual. Return ONLY JSON.\n\n"
            "IMPORTANT OUTPUT POLICY\n"
            "- Preserve original clinical or medical terms EXACTLY as written in the source text wherever they appear.\n"
            "- Any examples below are generic; do NOT copy their wording.\n"
            "- No prose outside the JSON object.\n\n"
            "SCHEMA\n{\n  \"variables\":[{\"name\":snake,\"type\":\"number|boolean|string\",\"unit\":null|unit,\"allowed\":null|[...] ,"
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
            f"RULES\n- Use canonical names if present: {canon} (examples; prefer the source's medical terms).\n"
            "- Do NOT invent thresholds; encode only literal thresholds.\n"
            "- No derived outputs in conditions (ban keys: danger_sign, clinic_referral, triage).\n"
            "- Every rule gets guideline_ref like \"p41\" or a section id.\n"
            "OUTPUT: only JSON for THIS section."
        )

        self.merge_system = (
            "Merge this FACT_SHEET (pre-extracted variables and rules) into one comprehensive IR. "
            "Return ONLY JSON with the same schema as step 1 plus:\n"
            "- canonical_map filled for all variables\n- qa.unmapped_vars\n- qa.dedup_dropped\n- qa.overlap_fixed\n"
            "RULES\n"
            "- Preserve original clinical terms; do NOT replace with generic placeholders.\n"
            "- Normalize labels to snake_case where needed; keep medical meaning.\n"
            "- Rewrite rules to canonical names.\n"
            "- Force then.advice to [].\n"
            "- Keep every literal threshold.\n"
            "INPUT\nFACT_SHEET:\n<COMPACT VARIABLES + RULES>"
        )

        self.rules_system = (
            "Given a FACT_SHEET (variables + rules), consolidate overlaps into RULES.\n"
            "Return ONLY JSON: { \"rules\": [ {\"rule_id\": str, \"when\": [...], \"then\": {\"triage\": \"hospital|clinic|home\", "
            "\"flags\":[], \"reasons\":[], \"actions\":[{\"id\":snake,\"if_available\":bool}], \"advice\":[], \"guideline_ref\": str, \"priority\": int } } ] }\n"
            "HARD RULES: preserve terms, resolve overlaps, ban derived fields in conditions, advice must be [], each rule has guideline_ref.\n"
        )

        # strict DMN prompt to ensure all outputs populated
        self.dmn_system = (
            "Convert RULES_JSON into modular DMN 1.4 using the DMN 1.4 MODEL namespace (2019-11-11). Return exactly TWO fenced blocks:\n"
            "1) ```xml <dmn:definitions>…```  2) ```json ASK_PLAN```\n\n"
            "HARD CONSTRAINTS\n"
            "- Use the variable names EXACTLY as they appear in RULES_JSON.\n"
            "- Advice must be an empty array in IR, empty string cells in DMN.\n"
            "- Decisions: decide_module_a, decide_module_b, decide_module_c, decide_module_d, decide_module_e, aggregate_final\n"
            "- Each module uses <dmn:decisionTable hitPolicy=\"FIRST\"> with outputs: triage:string, danger_sign:boolean, clinic_referral:boolean, reason:string, ref:string, advice:string\n"
            "- aggregate_final must use boolean outputs directly (no string parsing).\n"
            "- Populate EVERY <dmn:outputEntry><dmn:text> with a literal value (no blanks):\n"
            "  • triage: \"hospital\" | \"clinic\" | \"home\"\n"
            "  • danger_sign: true | false\n"
            "  • clinic_referral: true | false\n"
            "  • reason: snake_case string (e.g., severe_dehydration)\n"
            "  • ref: \"pNN\" or a section id string\n"
            "  • advice: \"\" (empty string)\n"
            "- Do not leave any output cells blank.\n"
            "- Provide an ASK_PLAN array describing input collection order and followups.\n"
        )

        self.bpmn_system = (
            "Produce one BPMN 2.0 <bpmn:definitions> only. Use xmlns:bpmn and xmlns:xsi. Build a simple ask -> evaluate -> route flow.\n"
        )

        self.coverage_system = (
            "Given RULES_JSON and the DMN XML, return ONLY JSON:\n"
            "{\"unmapped_rule_ids\":[...], \"module_counts\":{...}, \"notes\":[...]}\n"
            "Use RULES_JSON.ir_flat.conds to match DMN inputEntry texts literally when possible.\n"
            "Match on conditions and triage only.\n"
        )

    # ---------------- OpenAI helpers -----------------
    def _accumulate_usage(self, resp) -> None:
        try:
            u = getattr(resp, "usage", None)
            if not u and isinstance(resp, dict):
                u = resp.get("usage")
            if not u:
                return
            p = int(getattr(u, "prompt_tokens", 0) or (u.get("prompt_tokens") if isinstance(u, dict) else 0) or 0)
            c = int(getattr(u, "completion_tokens", 0) or (u.get("completion_tokens") if isinstance(u, dict) else 0) or 0)
            t = int(getattr(u, "total_tokens", 0) or (u.get("total_tokens") if isinstance(u, dict) else (p + c)))
            self.usage["prompt"] += p
            self.usage["completion"] += c
            self.usage["total"] += t
            self._usage_events.append({"prompt": p, "completion": c, "total": t})
        except Exception:
            pass

    def get_usage_summary(self) -> Dict[str, int]:
        return dict(self.usage)

    def _complete(self, model_name: str, messages: list, max_out: int):
        if not self.client:
            raise RuntimeError("OpenAI client not initialized (no API key).")
        base = dict(model=model_name, messages=messages)
        is_gpt5 = isinstance(model_name, str) and model_name.startswith("gpt-5")
        if not is_gpt5:
            base.update(dict(temperature=0.0, top_p=1))
        extras = {}
        if is_gpt5:
            extras = {"reasoning_effort": "minimal", "verbosity": "low"}
        base_seed = dict(base)
        base_seed["seed"] = getattr(self, "seed", None)
        try:
            resp = self.client.chat.completions.create(max_completion_tokens=max_out, **base_seed, **extras)
            self._accumulate_usage(resp)
            return resp
        except Exception as e1:
            msg1 = str(e1).lower()
            if "verbosity" in msg1 or "reasoning" in msg1 or "unsupported parameter" in msg1:
                try:
                    resp = self.client.chat.completions.create(max_completion_tokens=max_out, **base_seed)
                    self._accumulate_usage(resp)
                    return resp
                except Exception as e2:
                    msg2 = str(e2).lower()
                    if "max_completion_tokens" in msg2 or "unsupported parameter" in msg2:
                        try:
                            resp = self.client.chat.completions.create(max_tokens=max_out, **base_seed)
                            self._accumulate_usage(resp)
                            return resp
                        except Exception as e3:
                            msg3 = str(e3).lower()
                            if "seed" in msg3 or "unsupported parameter" in msg3:
                                base_noseed = dict(base)
                                resp = self.client.chat.completions.create(max_tokens=max_out, **base_noseed)
                                self._accumulate_usage(resp)
                                return resp
                            raise
                    raise
            if "max_completion_tokens" in msg1:
                try:
                    resp = self.client.chat.completions.create(max_tokens=max_out, **base_seed, **extras)
                    self._accumulate_usage(resp)
                    return resp
                except Exception as e4:
                    msg4 = str(e4).lower()
                    if "seed" in msg4 or "unsupported parameter" in msg4:
                        base_noseed = dict(base)
                        resp = self.client.chat.completions.create(max_tokens=max_out, **base_noseed, **extras)
                        self._accumulate_usage(resp)
                        return resp
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

    # ---------------- Step 1: per section -----------------
    def extract_rules_per_section(self, pdf_path: str) -> List[Dict[str, Any]]:
        sections = self.extract_sections_from_pdf(pdf_path)
        results: List[Dict[str, Any]] = []
        for sec_id, text in sections:
            user = f"SECTION_ID: {sec_id}\n\nTEXT:\n{text}"
            try:
                obj = self._chat_json(
                    self.section_system, user, max_out=6000, model_key="section", schema_model=IR if PydanticAvailable else None
                )
            except Exception:
                continue
            cleaned = []
            for r in obj.get("rules", []) or []:
                nr = _normalize_rule_schema(r)
                if nr:
                    cleaned.append(nr)
                else:
                    obj.setdefault("qa", {}).setdefault("notes", []).append(f"dropped_rule:{r.get('rule_id')}")
            obj["rules"] = _dedupe_rule_ids(cleaned, prefix=f"{sec_id}")
            results.append(obj)
        return results

    # ---------------- Step 2a: consolidate rules from compact fact sheet -----------------
    def generate_rules_from_fact_sheet(self, fact_sheet: Dict[str, Any]) -> List[Dict[str, Any]]:
        payload = json.dumps(fact_sheet, ensure_ascii=False)
        user = f"FACT_SHEET\n{payload}"
        try:
            obj = self._chat_json(self.rules_system, user, max_out=8000, model_key="merge", schema_model=None)
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
        try:
            self.debug_dir.joinpath("rules_from_facts.json").write_text(
                json.dumps({"rules": rules_out}, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
        return rules_out

    # ---------------- Step 2: merge -----------------
    def merge_sections(self, section_objs: List[Dict[str, Any]]) -> Dict[str, Any]:
        fact_sheet = _fact_sheet_from_sections(section_objs)
        try:
            self.debug_dir.joinpath("fact_sheet.json").write_text(json.dumps(fact_sheet, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        consolidated_rules = self.generate_rules_from_fact_sheet(fact_sheet)

        payload = json.dumps(fact_sheet, ensure_ascii=False)
        user = f"FACT_SHEET\n{payload}"

        if STRICT_MERGE:
            merged = self._chat_json(self.merge_system, user, max_out=8000, model_key="merge", schema_model=IR if PydanticAvailable else None)
            merged["rules"] = consolidated_rules
            if not merged.get("variables"):
                merged["variables"] = fact_sheet.get("variables", [])
        else:
            variables: Dict[str, Dict[str, Any]] = {}
            qa_notes: List[str] = ["fallback_merge_used", "source=fact_sheet"]
            for v in (fact_sheet.get("variables") or []):
                if not isinstance(v, dict):
                    continue
                name = str(v.get("name", "")).strip()
                if not name:
                    continue
                cur = variables.get(name.lower())
                if cur is None:
                    vv = dict(v)
                    if not isinstance(vv.get("synonyms"), list):
                        vv["synonyms"] = []
                    if not isinstance(vv.get("refs"), list):
                        vv["refs"] = []
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

        merged = _resolve_variables_snake_and_rewrite_rules(merged)

        canon = _build_canonical_map(self.config, merged.get("variables", []))
        merged["canonical_map"] = canon
        merged["rules"] = [_rewrite_rule_to_canon(r, canon) for r in (merged.get("rules") or [])]
        merged["rules"] = _dedupe_rule_ids(merged["rules"], prefix="merged")

        if PydanticAvailable:
            merged = IR.model_validate(merged).model_dump()
        _preflight_ir(merged)
        return merged

    # ---------------- Step 3: DMN plus ASK_PLAN -----------------
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
                dmn_xml = b
                continue
            if ask_plan is None:
                parsed = _parse_json_loose(b)
                if isinstance(parsed, (list, dict)):
                    ask_plan = parsed

        if dmn_xml is None:
            s = text
            start = s.find("<dmn:definitions")
            end = s.find("</dmn:definitions>")
            if start != -1 and end != -1 and end > start:
                end += len("</dmn:definitions>")
                dmn_xml = s[start:end].strip()

        if ask_plan is None:
            parsed = _parse_json_loose(text)
            if isinstance(parsed, (list, dict)):
                ask_plan = parsed

        def _sanitize_dmn(xml: str) -> str:
            s = xml
            s = re.sub(r"(<dmn:outputEntry>\s*<dmn:text>)([^<]*?)(</dmn:outputEntry>)", r"\1\2</dmn:text>\3", s, flags=re.DOTALL)
            s = s.replace(
                "<dmn:outputEntry><dmn:text></dmn:text></dmn:outputEntry>",
                '<dmn:outputEntry><dmn:text>""</dmn:text></dmn:outputEntry>',
            )
            s = s.replace(
                "<dmn:outputEntry><dmn:text></dmn:outputEntry>",
                '<dmn:outputEntry><dmn:text>""</dmn:text></dmn:outputEntry>',
            )
            s = re.sub(r"(<dmn:inputEntry>\s*<dmn:text>)([^<]*?)(</dmn:inputEntry>)", r"\1\2</dmn:text>\3", s, flags=re.DOTALL)
            s = s.replace(
                "<dmn:inputEntry><dmn:text></dmn:text></dmn:inputEntry>",
                "<dmn:inputEntry><dmn:text>-</dmn:text></dmn:inputEntry>",
            )
            s = s.replace(
                "<dmn:inputEntry><dmn:text></dmn:inputEntry>",
                "<dmn:inputEntry><dmn:text>-</dmn:inputEntry>",
            )
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

        # validate DMN completeness before proceeding
        _validate_dmn_outputs_or_die(dmn_xml)

        try:
            known_vars = {v["name"] for v in merged_ir.get("variables", []) if isinstance(v, dict) and v.get("name")}
            mutated: List[Dict[str, Any]] = []
            unknowns = set()
            if isinstance(ask_plan, dict) and "ASK_PLAN" in ask_plan:
                ask_plan = ask_plan.get("ASK_PLAN") or []
            if isinstance(ask_plan, list):
                for blk in ask_plan:
                    if not isinstance(blk, dict):
                        continue
                    ask = [q for q in (blk.get("ask") or []) if q in known_vars]
                    fuw: Dict[str, List[str]] = {}
                    for cond, qs in (blk.get("followups_if", {}) or {}).items():
                        kept = [q for q in (qs or []) if q in known_vars]
                        fuw[cond] = kept
                        for q in (qs or []):
                            if q not in known_vars:
                                unknowns.add(q)
                    for q in (blk.get("ask") or []):
                        if q not in known_vars:
                            unknowns.add(q)
                    mutated.append({"module": blk.get("module"), "ask": ask, "followups_if": fuw})
                ask_plan = mutated
            if unknowns:
                merged_ir.setdefault("qa", {}).setdefault("notes", []).append(f"ask_plan_unknowns_dropped:{sorted(list(unknowns))}")
            ask_plan = _enforce_ask_ownership(ask_plan)
        except Exception:
            pass

        # ensure inputs are defined for each decision table
        dmn_xml = _ensure_dmn_inputs(dmn_xml, ask_plan, merged_ir)

        return dmn_xml, ask_plan

    # ---------------- Step 4: BPMN -----------------
    def generate_bpmn(self, dmn_xml: str, ask_plan: Any) -> str:
        user = "DMN:\n```xml\n" + dmn_xml + "\n`````\n\n" + "ASK_PLAN:\n" + json.dumps(ask_plan, ensure_ascii=False)
        text = self._chat_text(self.bpmn_system, user, max_out=12000, model_key="bpmn")
        blocks = _extract_fenced_blocks(text)
        for _lang, body in blocks:
            if "<bpmn:definitions" in body:
                return _sanitize_bpmn(body)
        xml = _extract_xml_tag(text, "bpmn:definitions")
        if xml:
            return _sanitize_bpmn(xml)
        xml = _extract_xml_tag(text, "definitions")
        if xml:
            xml = re.sub(r"<\s*definitions\b", "<bpmn:definitions", xml, count=1)
            xml = re.sub(r"</\s*definitions\s*>", "</bpmn:definitions>", xml, count=1)
            return _sanitize_bpmn(xml)
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

    # ---------------- Step 6: DMN or ASK_PLAN to XLSX (XLSForm) -----------------
    def export_xlsx_from_dmn(
        self, merged_ir: Dict[str, Any], ask_plan: List[Dict[str, Any]], out_xlsx_path: str, template_xlsx_path: Optional[str] = None
    ) -> str:
        try:
            from openpyxl import load_workbook, Workbook
        except Exception as e:
            raise RuntimeError("openpyxl is required for XLSX export (pip install openpyxl)") from e

        # base workbook
        if template_xlsx_path and Path(template_xlsx_path).exists():
            wb = load_workbook(template_xlsx_path)
            for sheet in ("survey", "choices"):
                if sheet not in wb.sheetnames:
                    wb.create_sheet(title=sheet)
        else:
            wb = Workbook()
            if "Sheet" in wb.sheetnames:
                wb.remove(wb["Sheet"])
            ws_s = wb.create_sheet("survey")
            ws_c = wb.create_sheet("choices")
            ws_s.append(["type", "name", "label", "hint", "required", "constraint", "relevant", "appearance", "calculation"])
            ws_c.append(["list_name", "name", "label"])

        # SETTINGS
        form_id = Path(out_xlsx_path).stem
        title = form_id.replace("_", " ").title()
        version_val = int(time.time())
        sms_keyword_default = f"J1!{form_id}!"
        sms_separator_default = "!"

        if "settings" in wb.sheetnames:
            del wb["settings"]
        ws_set = wb.create_sheet("settings")
        ws_set.append(["id_string", "title", "default_language", "version", "sms_keyword", "sms_separator"])
        ws_set.append([str(form_id), str(title), "en", int(version_val), str(sms_keyword_default), str(sms_separator_default)])

        ws_survey = wb["survey"]
        ws_choices = wb["choices"]

        def _headers(ws):
            row1 = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None) or []
            return [c or "" for c in row1]

        s_hdr = _headers(ws_survey)
        c_hdr = _headers(ws_choices)

        def _col_ix(ws, hdr_list, name):
            try:
                return hdr_list.index(name)
            except Exception:
                hdr_list.append(name)
                ws.cell(row=1, column=len(hdr_list), value=name)
                return len(hdr_list) - 1

        s_ix = {k: _col_ix(ws_survey, s_hdr, k) for k in ("type", "name", "label", "hint", "required", "constraint", "relevant", "appearance", "calculation")}
        c_ix = {k: _col_ix(ws_choices, c_hdr, k) for k in ("list_name", "name", "label")}

        var_by_name = {(v.get("name") or "").strip(): v for v in (merged_ir.get("variables") or []) if isinstance(v, dict) and v.get("name")}

        def _label_for(name: str) -> str:
            v = var_by_name.get(name)
            if v and v.get("prompt"):
                return str(v["prompt"])
            return re.sub(r"_+", " ", str(name)).strip().capitalize()

        # choices: ensure yes_no once
        existing_choices = set()
        for row in ws_choices.iter_rows(min_row=2, values_only=True):
            if not row:
                continue
            list_name, name, _ = (row + (None, None, None))[:3]
            if list_name and name:
                existing_choices.add((str(list_name), str(name)))

        def ensure_yes_no():
            for val in ("yes", "no"):
                if ("yes_no", val) not in existing_choices:
                    row = [None] * len(c_hdr)
                    row[c_ix["list_name"]] = "yes_no"
                    row[c_ix["name"]] = val
                    row[c_ix["label"]] = val.capitalize()
                    ws_choices.append(row)
                    existing_choices.add(("yes_no", val))

        ensure_yes_no()

        def ensure_list(list_name: str, values: List[str]):
            for v in values or []:
                key = (list_name, str(v))
                if key not in existing_choices:
                    row = [None] * len(c_hdr)
                    row[c_ix["list_name"]] = list_name
                    row[c_ix["name"]] = str(v)
                    row[c_ix["label"]] = re.sub(r"_+", " ", str(v)).strip().capitalize()
                    ws_choices.append(row)
                    existing_choices.add(key)

        def _strip_quotes_local(s: str) -> str:
            s = (s or "").strip()
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                return s[1:-1]
            return s

        def cond_to_relevant(cond: str) -> Optional[str]:
            s = str(cond or "").strip()
            m = re.match(r"^\s*([A-Za-z0-9_]+)\s*([=!<>]=|[<>]|=)\s*(.+?)\s*$", s)
            if not m:
                return None
            var, op, val = m.groups()
            val = val.strip()
            if val.lower() in ("true", "false"):
                val = "'yes'" if val.lower() == "true" else "'no'"
            elif re.match(r"^-?\d+(\.\d+)?$", val):
                pass
            else:
                val = f"'{_strip_quotes_local(val)}'"
            if op == "==":
                op = "="
            return f"${{{var}}} {op} {val}"

        relevant_for: Dict[str, List[str]] = {}
        for blk in (ask_plan or []):
            fuw = (blk or {}).get("followups_if") or {}
            for cond, qs in fuw.items():
                r = cond_to_relevant(cond)
                if not r:
                    continue
                for q in (qs or []):
                    relevant_for.setdefault(q, []).append(r)

        def append_row(row_dict: Dict[str, Any]):
            row = [None] * len(s_hdr)
            for k, v in row_dict.items():
                if k not in s_ix:
                    s_ix[k] = _col_ix(ws_survey, s_hdr, k)
                row[s_ix[k]] = v
            ws_survey.append(row)

        added_questions: set = set()
        for blk in (ask_plan or []):
            mod = (blk or {}).get("module") or "module"
            append_row({"type": "begin group", "name": f"{_to_snake(mod)}_group", "label": re.sub(r"_+", " ", mod).title()})
            for q in (blk or {}).get("ask", []) or []:
                if q in added_questions:
                    continue
                v = var_by_name.get(q, {})
                qtype = str(v.get("type") or "").lower()
                allowed = v.get("allowed") or []
                if qtype == "boolean":
                    q_type_cell = "select_one yes_no"
                elif allowed:
                    list_name = f"list_{q}"
                    ensure_list(list_name, allowed)
                    q_type_cell = f"select_one {list_name}"
                elif qtype == "number":
                    q_type_cell = "decimal"
                else:
                    q_type_cell = "text"
                rel_expr = " or ".join(sorted(set(relevant_for[q]))) if q in relevant_for else None
                append_row({"type": q_type_cell, "name": q, "label": _label_for(q), "required": None, "relevant": rel_expr})
                added_questions.add(q)
            append_row({"type": "end group"})

        outp = Path(out_xlsx_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        wb.save(str(outp))
        return str(outp)

    # ---------------- Step 7: DMN to per module CSVs -----------------
    def parse_dmn_decision_tables(self, dmn_xml: str) -> Dict[str, dict]:
        root = ET.fromstring(dmn_xml)
        out: Dict[str, dict] = {}
        for dec in root.findall(".//dmn:decision", _DMNN):
            name = (dec.get("name") or dec.get("id") or "module").strip()
            mod = re.sub(r"^decide[_\-:]+", "", name).strip() or "module"
            table = dec.find(".//dmn:decisionTable", _DMNN)
            if table is None:
                continue
            inputs = [_text_or(i.find("./dmn:inputExpression/dmn:text", _DMNN)) for i in table.findall("./dmn:input", _DMNN)]
            outputs = [(o.get("name") or "").strip() for o in table.findall("./dmn:output", _DMNN)]
            rows = []
            for r in table.findall("./dmn:rule", _DMNN):
                conds = [_text_or(e.find("./dmn:text", _DMNN)) for e in r.findall("./dmn:inputEntry", _DMNN)]
                outs = [_text_or(e.find("./dmn:text", _DMNN)) for e in r.findall("./dmn:outputEntry", _DMNN)]
                rows.append({"conds": conds, "outs": outs})
            out[mod] = {"inputs": inputs, "outputs": outputs, "rows": rows}
        return out

    def export_csvs_from_dmn(self, dmn_xml: str, out_dir: str) -> Dict[str, str]:
        # normalize refs like 'p11-10, p 7 - 6 ; P15-P17' to 'p10-11, p7-6; p15-17'
        def _normalize_page_ranges(ref: str) -> str:
            s = _strip_quotes(ref or "")

            # add p if a range lacks it entirely
            def _ensure_p_prefix(m):
                a, b = m.group(1), m.group(2)
                return f"p{a}-{b}"

            s = re.sub(r"\b(\d{1,4})\s*-\s*(\d{1,4})\b", _ensure_p_prefix, s)

            # normalize pA - pB variants, preserving low to high
            def _swap_if_needed(m):
                a = int(m.group(1))
                b = int(m.group(2))
                lo, hi = sorted((a, b))
                return f"p{lo}-{hi}"

            s = re.sub(r"[Pp]\s*(\d{1,4})\s*-\s*[Pp]?\s*(\d{1,4})", _swap_if_needed, s)

            # tidy separators
            s = re.sub(r"\s*([,;])\s*", r"\1 ", s)
            return s.strip()

        tables = self.parse_dmn_decision_tables(dmn_xml)
        out_map: Dict[str, str] = {}
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)

        def norm_out_val(col: str, val: str) -> str:
            v = _strip_quotes(val or "")
            if col in ("danger_sign", "clinic_referral"):
                vv = v.strip().lower()
                if vv in ("true", "false"):
                    return vv
            if col == "ref":
                return _normalize_page_ranges(v)
            return v

        DEFAULTS = {
            "triage": "home",
            "danger_sign": "false",
            "clinic_referral": "false",
            "reason": "",
            "ref": "p0",
            "advice": "",
        }

        for mod, t in tables.items():
            outs = [c for c in t["outputs"] if c] or ["triage", "danger_sign", "clinic_referral", "reason", "ref", "advice"]
            fname = f"dmn_{mod}.csv"  # disk name maps to pulldata id dmn_<mod>
            fpath = outp / fname
            with open(fpath, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["key"] + outs)
                for i, row in enumerate(t["rows"], start=1):
                    key = f"r{str(i).zfill(3)}"
                    raw_vals = row["outs"][: len(outs)]
                    raw_vals += [""] * (len(outs) - len(raw_vals))
                    filled = []
                    for col_name, cell in zip(outs, raw_vals):
                        v = cell if (cell is not None and str(cell).strip() != "") else DEFAULTS.get(col_name, "")
                        filled.append(norm_out_val(col_name, v))
                    w.writerow([key] + filled)
            out_map[mod] = str(fpath)

        return out_map

    # ---------------- Step 8: Wire DMN outputs into XLSForm -----------------
    def wire_decisions_into_xlsx(self, xlsx_path: str, dmn_xml: str, merged_ir: Dict[str, Any], media_id_prefix: str = "dmn_") -> None:
        """
        Embed all DMN decision logic directly into the XLSForm.  This implementation
        no longer relies on per-module CSVs or pulldata() calls; instead each
        decision table's outputs are produced using nested if() expressions
        referencing the user's answers.  The media directory is reserved for
        images only.
        """
        from openpyxl import load_workbook

        wb = load_workbook(xlsx_path)
        ws = wb["survey"]

        def hdrs():
            row1 = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None) or []
            return [c or "" for c in row1]

        hdr = hdrs()

        def col(name):
            if name in hdr:
                return hdr.index(name)
            hdr.append(name)
            ws.cell(row=1, column=len(hdr), value=name)
            return len(hdr) - 1

        # column indices for survey sheet
        c_type = col("type")
        c_name = col("name")
        c_calc = col("calculation")
        c_lab = col("label")
        c_rel = col("relevant")

        def append_by_cols(values: dict):
            """Append a row to the survey sheet given a mapping of field names."""
            row = [None] * len(hdr)
            if "type" in values:
                row[c_type] = values["type"]
            if "name" in values:
                row[c_name] = values["name"]
            if "calculation" in values:
                row[c_calc] = values["calculation"]
            if "label" in values:
                row[c_lab] = values["label"]
            if "relevant" in values:
                row[c_rel] = values["relevant"]
            ws.append(row)

        def map_var_token_to_xls(var_token: str) -> str:
            """Translate a DMN variable token to the corresponding XLSForm variable name."""
            tok = (var_token or "").strip()
            # for tokens like decide_module_a.danger_sign → dmn_module_a_danger_sign
            if "." in tok:
                left, right = tok.rsplit(".", 1)
                mod = re.sub(r"^decide[_\-:]+", "", left)
                return f"{media_id_prefix}{_to_snake(mod)}_{_to_snake(right)}"
            return tok

        def feel_to_xls_test(expr: str) -> Optional[str]:
            """Convert a simple FEEL expression into an XLSForm boolean test."""
            s = (expr or "").strip()
            if s in ("", "-", "otherwise"):
                return None
            # basic comparison parsing
            m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_\.\-:]*)\s*(=|==|!=|<=|>=|<|>)\s*(.+?)\s*$", s)
            if not m:
                return None
            var_tok, op, rhs = m.groups()
            if op == "==":
                op = "="
            xls_var = map_var_token_to_xls(var_tok)
            lhs_is_calc = xls_var.startswith(media_id_prefix)
            val = (rhs or "").strip()
            # boolean literal mapping: if left side is a calculate field (prefixed), remain 'true'/'false', else map to yes/no
            if val.lower() in ("true", "false"):
                rhs_x = "'true'" if lhs_is_calc else ("'yes'" if val.lower() == "true" else "'no'")
            elif re.match(r"^-?\d+(\.\d+)?$", val):
                rhs_x = val
            else:
                rhs_x = f"'{_strip_quotes(val)}'"
            return f"${{{xls_var}}} {op} {rhs_x}"

        # parse all decision tables from the DMN
        tables = self.parse_dmn_decision_tables(dmn_xml)
        for mod, t in tables.items():
            # Build DMN row matchers (conjunction of conditions per rule)
            row_tests: List[str] = []
            for i, row in enumerate(t["rows"], start=1):
                conjuncts: List[str] = []
                for feel in row["conds"]:
                    xp = feel_to_xls_test(feel)
                    if xp:
                        conjuncts.append(xp)
                # join all conjuncts with and; if none, true()
                row_tests.append(_and_join(conjuncts))

            # Determine output columns (triage, danger_sign, etc.)
            outs = [c for c in t["outputs"] if c] or ["triage", "danger_sign", "clinic_referral", "reason", "ref", "advice"]

            # For each output column, build nested if expression selecting the row's output value
            for col_idx, out_col in enumerate(outs):
                # Gather the value for this output column from each rule
                values: List[str] = []
                for row in t["rows"]:
                    raw_val = row["outs"][col_idx] if col_idx < len(row["outs"]) else ""
                    # If value is missing or blank, keep as empty string
                    v = (raw_val or "").strip()
                    values.append(v)
                # Build nested ifs from bottom to top
                expr_out = "''"
                # iterate reversed to produce inner-most default first
                for test, val in reversed(list(zip(row_tests, values))):
                    # determine the literal representation
                    if val == "":
                        lit = "''"
                    else:
                        vv = _strip_quotes(val)
                        # booleans remain literal (true/false) for calculate fields
                        if vv.lower() in ("true", "false"):
                            lit = f"'{vv.lower()}'"
                        else:
                            lit = f"'{vv}'"
                    expr_out = f"if({test}, {lit}, {expr_out})"

                # Append calculate row for this output
                append_by_cols({
                    "type": "calculate",
                    "name": f"{media_id_prefix}{mod}_{out_col}",
                    "calculation": expr_out,
                    "label": f"{mod} {out_col}".replace("_", " "),
                    "relevant": None
                })

        wb.save(xlsx_path)

    def build_orchestrator_folder(
        self,
        dmn_xml: str,
        ask_plan: List[Dict[str, Any]],
        merged_ir: Dict[str, Any],
        output_dir: Union[str, Path],
        template_xlsx_path: Optional[str] = None,
        logo_path: Optional[str] = None,
    ) -> str:
        """
        Create a complete orchestrator folder containing an XLSForm with embedded
        decision logic, a placeholder XML file, a properties file, and a media
        directory with a logo image.  This helper wraps the existing XLSX
        creation and wiring functions and adds the additional files expected by
        the Community Health Toolkit (CHT).  The DMN decision tables are
        compiled directly into the XLSForm (no CSVs are produced).

        Parameters
        ----------
        dmn_xml: str
            The DMN XML generated from the model.
        ask_plan: list
            Ask plan describing the ordering of questions for each module.
        merged_ir: dict
            The merged intermediate representation containing variables and rules.
        output_dir: str or Path
            Directory into which the orchestrator folder structure should be
            created.  If the directory does not exist it will be created.
            The final orchestrator files (XLSX, XML, properties) and the media
            folder will reside directly inside this directory.
        template_xlsx_path: Optional[str]
            Optional path to an XLSX file to use as a template for the
            orchestrator.  If provided and exists, the template will be used
            as the base workbook for export_xlsx_from_dmn.
        logo_path: Optional[str]
            Optional path to a PNG image to use as the logo.  If not provided
            or the path does not exist, a default logo will be used when
            available.  If no default is available a blank PNG will be
            generated.

        Returns
        -------
        str
            The path to the orchestrator directory on disk.
        """
        out_root = Path(output_dir)
        # Ensure base orchestrator directory exists
        out_root.mkdir(parents=True, exist_ok=True)

        # Paths for individual files
        xlsx_path = out_root / "orchestrator.xlsx"
        xml_path = out_root / "orchestrator.xml"
        props_path = out_root / "orchestrator.properties.json"
        media_dir = out_root / "media"

        # Create media directory
        media_dir.mkdir(exist_ok=True)

        # Step 1: build xlsx using merged IR and ask plan
        # Use the provided template if supplied
        self.export_xlsx_from_dmn(merged_ir, ask_plan, str(xlsx_path), template_xlsx_path)

        # Step 2: embed DMN logic directly into xlsx
        self.wire_decisions_into_xlsx(str(xlsx_path), dmn_xml, merged_ir)

        # Step 3: generate placeholder XML
        xml_placeholder = (
            "<?xml version=\"1.0\"?>\n"
            "<orchestrator>\n"
            "  <!-- Placeholder XML form: logic embedded in orchestrator.xlsx -->\n"
            "</orchestrator>\n"
        )
        with xml_path.open("w", encoding="utf-8") as f_xml:
            f_xml.write(xml_placeholder)

        # Step 4: handle logo
        logo_dest = media_dir / "logo.png"
        # Use provided logo if available
        from shutil import copyfile
        if logo_path and Path(logo_path).exists():
            try:
                copyfile(logo_path, logo_dest)
            except Exception:
                pass
        else:
            # Try to copy a default logo from the current directory if present
            default_logo = Path(__file__).with_name("205a491d-21bb-46eb-be6e-6e5279e4156b.png")
            if default_logo.exists():
                try:
                    copyfile(default_logo, logo_dest)
                except Exception:
                    pass
            else:
                # Generate a 1x1 transparent PNG as fallback
                try:
                    from PIL import Image

                    img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
                    img.save(logo_dest, "PNG")
                except Exception:
                    # If PIL is not available, write an empty file
                    logo_dest.touch()

        # Step 5: write properties file
        properties = {
            "title": "Orchestrator",
            "icon": "icon-healthcare",
            "context": {
                "person": False,
                "place": False,
            },
            "internalId": "orchestrator",
            "xmlFormId": "orchestrator",
            "media": ["logo.png"],
        }
        with props_path.open("w", encoding="utf-8") as f_props:
            json.dump(properties, f_props, indent=2)

        return str(out_root)

    # ---------------- PDF sectioning wrapper -----------------
    def extract_sections_from_pdf(self, pdf_path: str, max_chars: int = 4000) -> List[Tuple[str, str]]:
        logp = self.debug_dir / "sectioning.log"

        pages = _extract_pypdf_pages(pdf_path)
        pages = _postprocess_page_text(pages)
        nonempty = sum(1 for _, t in pages if (t or "").strip())
        if nonempty > 0:
            chunks = _split_into_sections_by_headings(pages, max_chars=max_chars)
            if not chunks:
                chunks = _chunk_pages_len_only(pages, max_chars=max_chars)
            try:
                logp.write_text("Used pypdf extraction\n", encoding="utf-8")
            except Exception:
                pass
            return chunks

        if _have_pdfminer:
            pm_layout = _extract_pdfminer_layout_pages(pdf_path)
            pm_layout = _postprocess_page_text(pm_layout)
            if sum(1 for _, t in pm_layout if (t or "").strip()) > 0:
                chunks = _split_into_sections_by_headings(pm_layout, max_chars=max_chars)
                if not chunks:
                    chunks = _chunk_pages_len_only(pm_layout, max_chars=max_chars)
                try:
                    logp.write_text("Used pdfminer layout extraction\n", encoding="utf-8")
                except Exception:
                    pass
                return chunks

        if _have_pdfminer:
            pm_pages = _extract_pdfminer_pages(pdf_path)
            pm_pages = _postprocess_page_text(pm_pages)
            if sum(1 for _, t in pm_pages if (t or "").strip()) > 0:
                chunks = _split_into_sections_by_headings(pm_pages, max_chars=max_chars)
                if not chunks:
                    chunks = _chunk_pages_len_only(pm_pages, max_chars=max_chars)
                try:
                    logp.write_text("Used pdfminer basic extraction\n", encoding="utf-8")
                except Exception:
                    pass
                return chunks

        if _have_ocr:
            ocr_pages = _extract_ocr_pages(pdf_path)
            ocr_pages = _postprocess_page_text(ocr_pages)
            if sum(1 for _, t in ocr_pages if (t or "").strip()) > 0:
                chunks = _split_into_sections_by_headings(ocr_pages, max_chars=max_chars)
                if not chunks:
                    chunks = _chunk_pages_len_only(ocr_pages, max_chars=max_chars)
                try:
                    logp.write_text("Used OCR extraction (pytesseract + pdf2image)\n", encoding="utf-8")
                except Exception:
                    pass
                return chunks

        try:
            logp.write_text("No text extracted by any strategy\n", encoding="utf-8")
        except Exception:
            pass
        return []


# ---------------- convenience: minimal CLI usage (optional) ----------------
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Extract -> DMN -> CSVs -> XLSForm -> wire pulldata()")
    parser.add_argument("--pdf", required=False, help="Guideline PDF")
    parser.add_argument("--dmn", required=False, help="Existing DMN XML path (skips model steps)")
    parser.add_argument("--merged", required=False, help="Existing merged_ir.json path (skips model steps)")
    parser.add_argument("--ask", required=False, help="Existing ask_plan.json path (optional)")
    parser.add_argument("--xlsx-out", default="forms/app/orchestrator.xlsx")
    parser.add_argument("--media-dir", default="forms/app/orchestrator-media")
    parser.add_argument("--template", default=None)
    args = parser.parse_args()

    gx = OpenAIGuardedExtractor(api_key=os.environ.get("OPENAI_API_KEY"))

    merged = None
    dmn_xml = None
    ask_plan = None

    if args.pdf:
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        sections = gx.extract_rules_per_section(args.pdf)
        merged = gx.merge_sections(sections)
        dmn_xml, ask_plan = gx.generate_dmn_and_ask_plan(merged)
    elif args.dmn and args.merged:
        dmn_xml = Path(args.dmn).read_text(encoding="utf-8")
        merged = json.loads(Path(args.merged).read_text(encoding="utf-8"))
        if args.ask and Path(args.ask).exists():
            raw = json.loads(Path(args.ask).read_text(encoding="utf-8"))
            ask_plan = raw.get("ASK_PLAN") if isinstance(raw, dict) and "ASK_PLAN" in raw else raw
        else:
            # Minimal plan: ask all known variables in one group
            ask_plan = [
                {
                    "module": "module_a",
                    "ask": [v["name"] for v in (merged.get("variables") or []) if isinstance(v, dict) and v.get("name")],
                    "followups_if": {},
                }
            ]
    else:
        print("Supply either --pdf OR both --dmn and --merged", file=sys.stderr)
        sys.exit(2)

    # Create an orchestrator folder containing the XLSForm with embedded logic,
    # a placeholder XML, a properties file, and a media directory.  The
    # orchestrator folder name is derived from the --xlsx-out argument by
    # removing the .xlsx suffix.  For example, if --xlsx-out is
    # 'forms/app/orchestrator.xlsx', the resulting orchestrator directory
    # will be 'forms/app/orchestrator'.  CSVs are not generated.
    orchestrator_root = Path(args.xlsx_out).with_suffix("")
    orchestrator_root.parent.mkdir(parents=True, exist_ok=True)
    gx.build_orchestrator_folder(
        dmn_xml=dmn_xml,
        ask_plan=ask_plan,
        merged_ir=merged,
        output_dir=orchestrator_root,
        template_xlsx_path=args.template,
        logo_path=None,
    )

    print("Orchestrator directory:", orchestrator_root)