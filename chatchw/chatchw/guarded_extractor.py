"""
OpenAI-only, guarded extractor implementing a 5-step pipeline (optimized):
1) Per-section extraction → JSON IR  (strict schema, retries, smarter PDF sectioning)
2) Global merge + canonicalization → merged IR  (schema-validated)
3) Modular DMN (FIRST) + ASK_PLAN (+ advice column)  (robust parsing/sanitizing)
4) BPMN (from DMN + ASK_PLAN)
5) Coverage audit mapping RULES_JSON to DMN rows

Key improvements:
- Deterministic LLM params, optional per-step models, retry/backoff, loose→strict JSON repair
- Pydantic IR schema to prevent bad shapes entering pipeline (incl. 'advice' for treatment guidance)
- Config-driven canonical variables (fallback to in-file defaults), fewer hardcoded domain bits
- Smarter PDF sectioning: filter non-clinical pages, chunk to target size
- ASK_PLAN validator against IR variables; unknown questions dropped, logged to QA
- DMN sanitizer for common tag/text issues
- Centralized debug writes under ./.chatchw_debug

NEW (bulk update):
- Rule schema normalization (propose_triage → triage, set_flags → flags), drop empty WHEN, ban derived outputs in conditions
- Global rule_id dedupe/normalization
- Canonical variable mapping + rewrite of all conditions
- Fail-fast preflight (duplicates, empties)
- ASK ownership pass (single owner per variable to avoid double-asks)
- Coverage hinting with flattened canonical condition strings
"""

from __future__ import annotations

import collections
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pypdf

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# ---------------------------- Pydantic IR Schema ----------------------------
try:
    from pydantic import BaseModel, Field, ValidationError, conlist
    PydanticAvailable = True
except Exception:  # pragma: no cover
    PydanticAvailable = False

if PydanticAvailable:

    Op = Union[str]  # we’ll validate allowed ops in prompts; pydantic literal optional

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
        advice: List[str] = []  # treatment & advisory text snippets

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

# ---------------------------- JSON repair & retries ----------------------------

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
    """Repair common JSON issues: code fences, line/block comments, trailing commas, and crop to first obj/array."""
    s = _strip_code_fences(s)
    # strip block & line comments
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
    # remove trailing commas before ] or }
    s = re.sub(r",\s*(\]|\})", r"\1", s)
    # try to parse first object/array window
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

# ---------------------------- Normalization helpers ----------------
_CANON_TRIAGE_KEYS = ("triage", "propose_triage")
_CANON_FLAG_KEYS = ("flags", "set_flags")
_DERIVED_BLOCKLIST = {"danger_sign", "clinic_referral", "triage"}

def _normalize_rule_schema(rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Make rule schema consistent; drop if invalid/empty WHEN or derived outputs used as inputs."""
    r = dict(rule or {})
    th = dict(r.get("then") or {})
    # triage
    triage = None
    for k in _CANON_TRIAGE_KEYS:
        if th.get(k) is not None:
            triage = th[k]
            break
    if triage is not None:
        th["triage"] = triage
    # flags
    flags = None
    for k in _CANON_FLAG_KEYS:
        if th.get(k) is not None:
            flags = th[k]
            break
    th["flags"] = list(flags or [])
    # defaults
    th.setdefault("reasons", [])
    th.setdefault("actions", [])
    th.setdefault("advice", [])
    th["priority"] = int(th.get("priority") or 0)
    r["then"] = th

    # Validate WHEN (ban empty lists and bad shapes, ban derived outputs in conditions)
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
    arr = []
    for r in ir.get("rules", []):
        arr.append({
            "rule_id": r.get("rule_id"),
            "triage": (r.get("then") or {}).get("triage"),
            "conds": _flatten_rule_conditions(r)
        })
    return arr

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

def _enforce_ask_ownership(ask_plan: List[Dict[str, Any]], priority_order=("danger_signs","respiratory","diarrhea","fever_malaria","nutrition")):
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

# ---------------------------- Guarded Extractor ----------------------------
class OpenAIGuardedExtractor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_section: str = "gpt-4o-mini",
        model_merge: str = "gpt-4o-mini",
        model_dmn: str = "gpt-4o",
        model_bpmn: str = "gpt-4o",
        model_audit: str = "gpt-4o-mini",
        seed: Optional[int] = 42,
        canonical_config_path: Optional[str] = "chatchw/config/canonical_config.json",
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required")
        if OpenAI is None:
            raise RuntimeError("openai package not available")

        self.client = OpenAI(api_key=key)
        self.seed = seed
        self.models = dict(
            section=model_section, merge=model_merge, dmn=model_dmn, bpmn=model_bpmn, audit=model_audit
        )

        self.debug_dir = Path(".chatchw_debug")
        self.debug_dir.mkdir(exist_ok=True)

        # --- Load config (fallback defaults if file missing) ---
        default_canon_vars = [
            "diarrhea",
            "blood_in_stool",
            "diarrhea_duration_days",
            "fever",
            "temperature_c",
            "fever_duration_days",
            "cough",
            "cough_duration_days",
            "resp_rate",
            "age_months",
            "chest_indrawing",
            "muac_mm",
            "edema_both_feet",
            "convulsions",
            "unconscious",
            "unable_to_drink",
            "vomiting_everything",
            "malaria_area",
        ]
        try:
            cfg_path = Path(canonical_config_path)
            if cfg_path.exists():
                self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
            else:
                self.config = {
                    "canonical_variables": default_canon_vars,
                    "priority_tiers": {"hospital_min": 90, "clinic_min": 50},
                    "resp_rate_cutoffs": {"infant_ge": 50, "child_ge": 40},
                    "muac": {"clinic_min": 115, "home_min": 125},
                }
        except Exception:
            self.config = {
                "canonical_variables": default_canon_vars,
                "priority_tiers": {"hospital_min": 90, "clinic_min": 50},
                "resp_rate_cutoffs": {"infant_ge": 50, "child_ge": 40},
                "muac": {"clinic_min": 115, "home_min": 125},
            }

        canon = ", ".join(self.config.get("canonical_variables", default_canon_vars))

        # ---------------- PROMPTS ----------------
        # Step 1 prompt (per-section)
        self.section_system = (
            "You extract WHO CHW clinical rules. Return ONLY JSON.\n\n"
            "SCHEMA\n{\n  \"variables\":[{\"name\":snake,\"type\":\"number|boolean|string\",\"unit\":null|unit,\"allowed\":null|[...],\n"
            "                \"synonyms\":[snake...],\"prompt\":short,\"refs\":[page_or_section]}],\n  \"rules\":[{\"rule_id\":str,\n"
            "            \"when\":[ {\"obs\":var,\"op\":\"lt|le|gt|ge|eq|ne\",\"value\":num|bool|string} |\n"
            "                     {\"sym\":var,\"eq\":true|false} |\n"
            "                     {\"all_of\":[COND...]} | {\"any_of\":[COND...]} ],\n"
            "            \"then\":{\"triage\":\"hospital|clinic|home\",\n"
            "                    \"flags\":[snake...],\n"
            "                    \"reasons\":[snake...],\n"
            "                    \"actions\":[{\"id\":snake,\"if_available\":bool}],\n"
            "                    \"advice\":[string...],\n"
            "                    \"guideline_ref\":str,\"priority\":int}}],\n"
            "  \"canonical_map\":{},\n"
            "  \"qa\":{\"notes\":[]}\n}\n\n"
            f"RULES\n- Use canonical names if present: {canon}\n"
            "- Add all seen aliases into synonyms and rewrite conditions to the canonical\n"
            "- No derived outputs in conditions (ban danger_sign, clinic_referral, triage)\n"
            "- Encode literal thresholds only. No invented cutoffs. No “..”\n"
            "- Priority tiers: hospital≥90, clinic 50–89, home<50\n"
            "- Every rule gets guideline_ref like \"p41\" or section id\n\n"
            "OUTPUT\nOnly the JSON object for THIS section"
        )

        # Step 2 prompt (merge)
        self.merge_system = (
            "Merge these section JSON objects into one comprehensive IR. Return ONLY JSON with the same schema as step 1 plus:\n"
            "- canonical_map filled for all variables\n- qa.unmapped_vars\n- qa.dedup_dropped\n- qa.overlap_fixed\nRULES\n- Rewrite all rules to canonical names\n"
            "- If two rules conflict on the same condition set tighten bounds or split any_of so both can exist without overlap\n"
            "- Keep every literal threshold\nINPUT\n<PASTE ALL SECTION JSONs>"
        )

        # Step 3 prompt (DMN + ASK_PLAN)
        self.dmn_system = (
            "Convert RULES_JSON into modular DMN 1.4 using the DMN 1.4 MODEL namespace (2019-11-11). Return exactly two fenced blocks:\n"
            "1) ```xml <dmn:definitions>…```  2) ```json ASK_PLAN```\n\n"
            "DMN REQUIREMENTS\n"
            "- Root tag MUST be <dmn:definitions xmlns:dmn=\"https://www.omg.org/spec/DMN/20191111/MODEL/\">. No other DMN namespaces.\n"
            "- Decisions: decide_danger_signs, decide_diarrhea, decide_fever_malaria, decide_respiratory, decide_nutrition, aggregate_final\n"
            "- Each module decision uses <dmn:decisionTable hitPolicy=\"FIRST\">\n  Inputs: only relevant canonical variables\n"
            "  Outputs: triage:string, danger_sign:boolean, clinic_referral:boolean, reason:string, ref:string, advice:string\n"
            "  Rows: use FEEL literals/comparators only; escape &lt; and &gt; in <dmn:text>.\n"
            "  Invariants: hospital → danger_sign=true; clinic → clinic_referral=true.\n"
            "- decide_danger_signs one row per sign (no multi-sign rows)\n"
            "- decide_diarrhea: blood_in_stool==true → clinic; diarrhea_duration_days >= 14 → clinic\n"
            "- decide_fever_malaria: add classic fever/malaria escalations\n"
            "- decide_respiratory: age-based fast breathing thresholds\n"
            "- decide_nutrition: MUAC 115–124 → clinic; >=125 → home\n"
            "- aggregate_final inputs: module booleans; FIRST policy rows → hospital/clinic/home\n"
            "- Add <dmn:informationRequirement> from aggregate_final to each module decision.\n"
            "- Ensure <dmn:inputData> exists for every canonical variable.\n\n"
            "ASK_PLAN (canonical names; parent-first with gated follow-ups):\n"
            "[ {\"module\":\"diarrhea\",\"ask\":[\"diarrhea\"],\"followups_if\":{\"diarrhea==true\":[\"blood_in_stool\",\"diarrhea_duration_days\"]}},\n"
            "  {\"module\":\"fever_malaria\",\"ask\":[\"fever\"],\"followups_if\":{\"fever==true\":[\"temperature_c\",\"fever_duration_days\",\"malaria_area\"]}},\n"
            "  {\"module\":\"respiratory\",\"ask\":[\"cough\"],\"followups_if\":{\"cough==true\":[\"resp_rate\",\"cough_duration_days\",\"chest_indrawing\"]}},\n"
            "  {\"module\":\"nutrition\",\"ask\":[\"muac_mm\",\"edema_both_feet\"]},\n"
            "  {\"module\":\"danger_signs\",\"ask\":[\"convulsions\",\"unconscious\",\"unable_to_drink\",\"vomiting_everything\",\"chest_indrawing\",\"edema_both_feet\"]}]\n"
        )

        # Step 4 prompt (BPMN)
        self.bpmn_system = (
            "Produce one BPMN 2.0 <bpmn:definitions> only. Use xmlns:bpmn=\"http://www.omg.org/spec/BPMN/20100524/MODEL\" and xmlns:xsi. No vendor extensions.\n\n"
            "Process id=\"chatchw_flow\" isExecutable=\"false\"\n"
            "Start → userTask \"Ask diarrhea\" → XOR \"Diarrhea present?\"\n true → userTask \"Collect diarrhea details\" → userTask \"Ask fever/malaria\"\n false → userTask \"Ask fever/malaria\"\n→ userTask \"Ask cough/pneumonia\" → userTask \"Ask malnutrition\"\n→ businessRuleTask \"Evaluate decisions\" decisionRef=\"aggregate_final\"\n→ XOR \"Main triage\"\n"
            "  flow to \"Hospital\" with <conditionExpression xsi:type=\"tFormalExpression\">danger_sign == true</conditionExpression>\n"
            "  flow to \"Clinic\" with <conditionExpression xsi:type=\"tFormalExpression\">clinic_referral == true</conditionExpression>\n"
            "  default flow to \"Home\"\nUse canonical variable names exactly. Valid namespaces. No prose.\nINPUT\nProvide the DMN you just produced and the ASK_PLAN to align variable names"
        )

        # Step 5 prompt (coverage)
        self.coverage_system = (
            "Given RULES_JSON and the DMN XML, return ONLY JSON:\n"
            "{\"unmapped_rule_ids\":[...], \"module_counts\":{...}, \"notes\":[...]}\n"
            "Use RULES_JSON.ir_flat.conds (canonical stringified conditions) to match DMN inputEntry texts literally when possible.\n"
            "A rule is mapped if its literal conditions appear as inputEntry cells in some module row with the same triage tier.\n"
            "INPUT\nRULES_JSON: <merged IR + ir_flat>\nDMN: <xml>"
        )

    # ---------------- PDF sectioning (smarter) -----------------
    def extract_sections_from_pdf(self, pdf_path: str, max_chars: int = 4000) -> List[Tuple[str, str]]:
        """
        Return a list of (section_id, text) with page markers retained.
        Filters to likely-clinical pages and merges adjacent pages into ~max_chars chunks.
        """
        out: List[Tuple[str, str]] = []
        raw: List[Tuple[int, str]] = []
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                try:
                    t = (page.extract_text() or "").strip()
                except Exception:
                    t = ""
                raw.append((i, t))

        def is_clinical(t: str) -> bool:
            keys = ["fever", "diarrhea", "cough", "muac", "convulsion", "vomit", "resp", "edema", "temperature", "age", "danger", "triage"]
            score = sum(k in t.lower() for k in keys)
            return score >= 1 and len(t) > 200

        chunks: List[Tuple[str, str]] = []
        buf = ""
        ids: List[int] = []
        for i, t in raw:
            if not is_clinical(t):
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

    # ---------------- OpenAI helpers -----------------
    def _chat_json(self, system: str, user: str, max_tokens: int = 6000, model_key: str = "section", schema_model=None) -> Dict[str, Any]:
        def _call():
            resp = self.client.chat.completions.create(
                model=self.models.get(model_key, self.models["section"]),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                top_p=1,
                max_tokens=max_tokens,
                seed=self.seed,  # supported in newer SDKs; safe if ignored
            )
            return (resp.choices[0].message.content or "").strip()

        try:
            return _call_json_with_retries(_call, schema_model=schema_model, retries=3)
        except Exception as e:
            Path(self.debug_dir / "guarded_debug_last.txt").write_text(f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nERR:{e}", encoding="utf-8")
            raise

    def _chat_text(self, system: str, user: str, max_tokens: int = 12000, model_key: str = "dmn") -> str:
        resp = self.client.chat.completions.create(
            model=self.models.get(model_key, self.models["dmn"]),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            top_p=1,
            max_tokens=max_tokens,
            seed=self.seed,
        )
        return (resp.choices[0].message.content or "").strip()

    # ---------------- Step 1: per-section -----------------
    def extract_rules_per_section(self, pdf_path: str) -> List[Dict[str, Any]]:
        sections = self.extract_sections_from_pdf(pdf_path)
        results: List[Dict[str, Any]] = []
        for sec_id, text in sections:
            user = f"SECTION_ID: {sec_id}\n\nTEXT:\n{text}"
            try:
                obj = self._chat_json(self.section_system, user, max_tokens=6000, model_key="section", schema_model=IR if PydanticAvailable else None)
            except Exception:
                # Skip bad sections but keep going
                continue

            # Normalize & drop invalid rules early; dedupe rule_ids by-section
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

    # ---------------- Step 2: merge -----------------
    def merge_sections(self, section_objs: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = json.dumps(section_objs, ensure_ascii=False)
        user = f"INPUT\n{payload}"
        try:
            merged = self._chat_json(self.merge_system, user, max_tokens=6000, model_key="merge", schema_model=IR if PydanticAvailable else None)
        except Exception:
            # Fallback: deterministic merge to guarantee JSON
            variables: Dict[str, Dict[str, Any]] = {}
            rules: List[Dict[str, Any]] = []
            qa_notes: List[str] = ["fallback_merge_used"]
            for sec in section_objs:
                for v in (sec.get("variables") or []):
                    if not isinstance(v, dict):
                        continue
                    name = str(v.get("name", "")).strip().lower()
                    if not name:
                        continue
                    cur = variables.get(name)
                    if cur is None:
                        vv = dict(v)
                        if not isinstance(vv.get("synonyms"), list): vv["synonyms"] = []
                        if not isinstance(vv.get("refs"), list): vv["refs"] = []
                        variables[name] = vv
                    else:
                        syn = set(cur.get("synonyms") or []) | set(v.get("synonyms") or [])
                        refs = set(cur.get("refs") or []) | set(v.get("refs") or [])
                        cur["synonyms"] = sorted(list(syn))
                        cur["refs"] = sorted(list(refs))
                for r in (sec.get("rules") or []):
                    if isinstance(r, dict):
                        nr = _normalize_rule_schema(r)
                        if nr:
                            rules.append(nr)
                        else:
                            qa_notes.append(f"dropped_rule:{r.get('rule_id')}")
            rules = _dedupe_rule_ids(rules, prefix="merge")
            merged = {
                "variables": list(variables.values()),
                "rules": rules,
                "canonical_map": {},
                "qa": {"notes": qa_notes, "unmapped_vars": [], "dedup_dropped": 0, "overlap_fixed": []},
            }

        # Canonical map + rewrite rules to canonical; validate & preflight
        canon = _build_canonical_map(self.config, merged.get("variables", []))
        merged["canonical_map"] = canon
        merged["rules"] = [_rewrite_rule_to_canon(r, canon) for r in (merged.get("rules") or [])]
        # dedupe rule_ids again (post-normalization)
        merged["rules"] = _dedupe_rule_ids(merged["rules"], prefix="merged")

        if PydanticAvailable:
            merged = IR.model_validate(merged).model_dump()
        _preflight_ir(merged)
        return merged

    # ---------------- Step 3: DMN + ASK_PLAN -----------------
    def generate_dmn_and_ask_plan(self, merged_ir: Dict[str, Any]) -> Tuple[str, Any]:
        user = "RULES_JSON:\n" + json.dumps(merged_ir, ensure_ascii=False)
        text = self._chat_text(self.dmn_system, user, max_tokens=12000, model_key="dmn")

        # Save raw for troubleshooting
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

        # First pass: use fenced blocks
        for _lang, body in blocks:
            b = body.strip()
            if dmn_xml is None and "<dmn:definitions" in b and "</dmn:definitions>" in b:
                dmn_xml = b
                continue
            if ask_plan is None:
                parsed = _parse_json_loose(b)
                if isinstance(parsed, (list, dict)):
                    ask_plan = parsed

        # Fallbacks
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

        # Minimal DMN sanitizer
        def _sanitize_dmn(xml: str) -> str:
            s = xml
            # Ensure closing text tags
            s = re.sub(r"(<dmn:outputEntry>\s*<dmn:text>)([^<]*?)(</dmn:outputEntry>)", r"\1\2</dmn:text>\3", s, flags=re.DOTALL)
            s = s.replace("<dmn:outputEntry><dmn:text></dmn:text></dmn:outputEntry>", "<dmn:outputEntry><dmn:text>\"\"</dmn:text></dmn:outputEntry>")
            s = s.replace("<dmn:outputEntry><dmn:text></dmn:outputEntry>", "<dmn:outputEntry><dmn:text>\"\"</dmn:text></dmn:outputEntry>")
            s = re.sub(r"(<dmn:inputEntry>\s*<dmn:text>)([^<]*?)(</dmn:inputEntry>)", r"\1\2</dmn:text>\3", s, flags=re.DOTALL)
            s = s.replace("<dmn:inputEntry><dmn:text></dmn:text></dmn:inputEntry>", "<dmn:inputEntry><dmn:text>-</dmn:text></dmn:inputEntry>")
            s = s.replace("<dmn:inputEntry><dmn:text></dmn:inputEntry>", "<dmn:inputEntry><dmn:text>-</dmn:text></dmn:inputEntry>")
            # Force MODEL namespace only
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

        # ASK_PLAN validation against IR variables (drop unknowns, log) + ownership enforcement
        try:
            known_vars = {v["name"] for v in merged_ir.get("variables", []) if isinstance(v, dict) and v.get("name")}
            mutated = []
            unknowns = set()
            if isinstance(ask_plan, list):
                for blk in ask_plan:
                    if not isinstance(blk, dict):
                        continue
                    ask = [q for q in (blk.get("ask") or []) if q in known_vars]
                    fuw = {}
                    for cond, qs in (blk.get("followups_if", {}) or {}).items():
                        fuw[cond] = [q for q in (qs or []) if q in known_vars]
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
            # Ownership pass to prevent double-asks
            ask_plan = _enforce_ask_ownership(ask_plan)
        except Exception:
            pass

        return dmn_xml, ask_plan

    # ---------------- Step 4: BPMN -----------------
    def generate_bpmn(self, dmn_xml: str, ask_plan: Any) -> str:
        user = "DMN:\n```xml\n" + dmn_xml + "\n```\n\n" + "ASK_PLAN:\n" + json.dumps(ask_plan, ensure_ascii=False)
        text = self._chat_text(self.bpmn_system, user, max_tokens=6000, model_key="bpmn")
        blocks = _extract_fenced_blocks(text)
        for _lang, body in blocks:
            if "<bpmn:definitions" in body:
                return body.strip()
        Path(self.debug_dir / "bpmn_debug_last.txt").write_text(text, encoding="utf-8")
        raise RuntimeError("Failed to get BPMN from model")

    # ---------------- Step 5: coverage -----------------
    def audit_coverage(self, merged_ir: Dict[str, Any], dmn_xml: str) -> Dict[str, Any]:
        payload = {
            "ir": merged_ir,
            "ir_flat": _rules_flattened(merged_ir)  # helper for exact literal matching
        }
        user = "RULES_JSON:\n" + json.dumps(payload, ensure_ascii=False) + "\nDMN:\n```xml\n" + dmn_xml + "\n```"
        return self._chat_json(self.coverage_system, user, max_tokens=6000, model_key="audit", schema_model=None)
