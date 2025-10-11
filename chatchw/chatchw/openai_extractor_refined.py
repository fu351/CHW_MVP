"""
OpenAI-powered rule extraction from PDFs and text.
Generalized, standards-aligned, normalized rules IR for downstream DMN/BPMN.
- No hardcoded clinical variables
- Page-aware chunking
- Deterministic JSON outputs
- Normalization, deduplication, and alignment invariants
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import pypdf

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# ------------------------- helpers -------------------------

SNAKE_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def to_snake(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(".", "_").replace("-", "_").replace("/", "_")
    s = SNAKE_NON_ALNUM.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def coerce_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"true", "yes", "y", "1"}:
            return True
        if t in {"false", "no", "n", "0"}:
            return False
    return None


def coerce_number(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception:
            return None
    return None


def stable_rule_signature(rule: Dict[str, Any]) -> Tuple:
    """Build a signature to dedup semantically equivalent rules."""
    when = rule.get("when", [])
    then = rule.get("then", {})
    when_sig: List[Tuple] = []
    for cond in when:
        if not isinstance(cond, dict):
            continue
        if "obs" in cond and "op" in cond and "value" in cond:
            when_sig.append(("obs", to_snake(str(cond["obs"])), str(cond["op"]).lower(), str(cond["value"])))
        elif "sym" in cond and "eq" in cond:
            when_sig.append(("sym", to_snake(str(cond["sym"])), bool(cond["eq"])))
        elif "any_of" in cond:
            inner = tuple(sorted([stable_rule_signature({"when": [c]}) for c in cond.get("any_of", [])]))
            when_sig.append(("any_of", inner))
        elif "all_of" in cond:
            inner = tuple(sorted([stable_rule_signature({"when": [c]}) for c in cond.get("all_of", [])]))
            when_sig.append(("all_of", inner))
    when_sig_sorted = tuple(sorted(when_sig))

    propose = then.get("propose_triage")
    flags = tuple(sorted([to_snake(x) for x in then.get("set_flags", [])])) if isinstance(then.get("set_flags"), list) else ()
    reasons = tuple(sorted([to_snake(x) for x in then.get("reasons", [])])) if isinstance(then.get("reasons"), list) else ()
    return (when_sig_sorted, str(propose), flags, reasons)


# ------------------------- extractor -------------------------

class OpenAIRuleExtractor:
    """Extract clinical rules using OpenAI API with normalization and invariants."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_extraction: str = "gpt-4o",
        chunk_chars: int = 9000,
        overlap_chars: int = 600,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if OpenAI is None:
            raise RuntimeError("openai package not available in this environment")

        self.client = OpenAI(api_key=self.api_key)
        self.model_extraction = model_extraction
        self.chunk_chars = max(2000, chunk_chars)
        self.overlap_chars = max(0, overlap_chars)

        # Generalized, page-aware, standards-aligned system prompt
        self.system_prompt = system_prompt or (
            "You extract deterministic, machine-readable clinical triage rules from WHO CHW guidance.\n\n"
            "Return ONLY JSON (no prose) with:\n"
            "{\n  \"variables\":[\n    {\"name\":snake_case, \"type\":\"number|boolean|string\",\n     \"unit\":null|unit, \"allowed\":null|[...],\n     \"synonyms\":[snake_case...], \"prompt\":short_question, \"refs\":[page_or_section]}\n  ],\n  \"rules\":[\n    {\"rule_id\":str,\n     \"when\":[ COND | {\"any_of\":[COND...] } | {\"all_of\":[COND...]} ],\n     \"then\":{\n       \"triage\":\"hospital|clinic|home\",\n       \"flags\":[snake_case...],\n       \"reasons\":[snake_case...],\n       \"actions\":[{\"id\":snake_case,\"if_available\":bool}],\n       \"guideline_ref\":str, \"priority\":int } }\n  ]\n}\n\n"
            "COND:\n- {\"obs\": var, \"op\":\"lt|le|gt|ge|eq|ne\", \"value\": number|string|boolean}\n- {\"sym\": var, \"eq\": true|false}\n\n"
            "Canonicals you MUST detect if present, with synonyms and units:\n"
            "- temperature_c:number (°C), resp_rate:number (/min), age_months:number, muac_mm:number (mm), muac_color:string\n"
            "- diarrhea:boolean, diarrhea_duration_days:number, blood_in_stool:boolean\n"
            "- fever:boolean, fever_duration_days:number\n"
            "- cough:boolean, cough_duration_days:number, chest_indrawing:boolean\n"
            "- convulsions:boolean, unconscious:boolean, unable_to_drink:boolean, vomiting_everything:boolean\n"
            "- edema_both_feet:boolean, malaria_area:boolean\n\n"
            "Normalization/invariants:\n- snake_case everything; coalesce synonyms into the canonical names above\n"
            "- encode age-specific thresholds as explicit numeric rules (e.g., fast breathing: <12m: resp_rate>=50; ≥12m: resp_rate>=40)\n"
            "- turn narrative ranges into concrete comparators; no “..” syntax\n"
            "- set priority: hospital≥90, clinic 50–89, home<50\n"
            "- always include guideline_ref using page tags like \"p12\"\n"
            "- deduplicate exact duplicates by condition signature; keep the higher priority\n\n"
            "Also return:\n{\n  \"canonical_map\": {canonical_name: {synonyms:[...], type, unit, allowed}},\n  \"modules_order\": [\"danger_signs\",\"diarrhea\",\"fever_malaria\",\"respiratory\",\"nutrition\"],\n  \"ask_plan\": [\n    {\"module\":\"diarrhea\",\"ask_if\":\"true\",\"vars\":[\"diarrhea\"],\"followups_if\":{\"diarrhea==true\":[\"blood_in_stool\",\"diarrhea_duration_days\"]}},\n    {\"module\":\"fever_malaria\",\"vars\":[\"fever\"],\"followups_if\":{\"fever==true\":[\"fever_duration_days\",\"temperature_c\",\"malaria_area\"]}},\n    {\"module\":\"respiratory\",\"vars\":[\"cough\"],\"followups_if\":{\"cough==true\":[\"cough_duration_days\",\"resp_rate\",\"chest_indrawing\"]}},\n    {\"module\":\"nutrition\",\"vars\":[\"muac_mm\",\"edema_both_feet\"],\"derive\":[\"muac_color=red if muac_mm<115\"]},\n    {\"module\":\"danger_signs\",\"vars\":[\"convulsions\",\"unconscious\",\"unable_to_drink\",\"vomiting_everything\",\"chest_indrawing\",\"edema_both_feet\"]}\n  ]\n}\n\n"
            "Be comprehensive. Do NOT drop rules due to size; emit all. If text implies only MUAC strap color, still add muac_mm as numeric with refs, and derive MUAC color from thresholds."
        )

        # Modular generation prompts (DMN first, then BPMN)
        self.modular_dmn_system_prompt = (
            "You convert RULES_JSON into BPMN 2.0 and modular DMN 1.4 for a triage chatbot. Be deterministic. No prose.\n\n"
            "Constraints:\n"
            "- Use ALL rules. If large, split DMN into modules: decide_danger_signs, decide_diarrhea, decide_fever_malaria, decide_respiratory, decide_nutrition, then aggregate_final.\n"
            "- FEEL only with literals and comparators (=/!=/</<=/>/>=, true/false). NO string functions like contains().\n"
            "- Each module’s decisionTable outputs three columns: triage (string), danger_sign (boolean), clinic_referral (boolean). Also include reason (string) and ref (string).\n"
            "- Aggregator precedence: if any module.danger_sign==true → final triage:hospital; else if any module.clinic_referral==true → triage:clinic; else triage:home. The aggregator must recompute booleans and also output reason/ref from the highest-priority firing module.\n"
            "- Every module has hitPolicy=\"FIRST\" with rules ordered by clinical severity.\n"
            "- Variables are the canonical names from RULES_JSON. Include <dmn:inputData> for each with correct typeRef.\n\n"
            "STRICT OUTPUT (exactly 4 fenced blocks, in this order; no prose):\n"
            "1) ```xml …single <dmn:definitions> with all module decisions and aggregate_final… ```\n"
            "2) ```json  // CANONICAL_MAP.json ```\n"
            "3) ```json  // QA_REPORT.json ```\n"
            "4) ```json  // ASK_PLAN.json ```\n"
        )

        self.modular_bpmn_system_prompt = (
            "You are a BPMN 2.0 workflow designer. Using the canonical names and decision ids from the DMN you were just shown (top-level decision id 'aggregate_final'), generate a single BPMN process that:\n"
            "- Hard-codes a diarrhea-first interview\n"
            "- Collects only inputs needed by modules (per ASK_PLAN), minimizing redundant questions\n"
            "- Invokes the DMN once after data collection\n"
            "- Routes by DMN flags (danger_sign / clinic_referral) with Home as default\n\n"
            "STRICT OUTPUT: exactly one fenced code block containing valid BPMN 2.0 XML (<bpmn:definitions>…</bpmn:definitions>). No prose.\n\n"
            "BPMN REQUIREMENTS\n"
            "Process id=\"chatchw_flow\" name=\"ChatCHW Clinical Interview\" isExecutable=\"false\".\n"
            "Nodes (in order): Start → Ask diarrhea → XOR 'Diarrhea present?' (true → Collect diarrhea details → Ask fever/malaria; false → Ask fever/malaria) → Ask cough/pneumonia → Ask malnutrition → businessRuleTask 'Evaluate decisions' decisionRef=\"aggregate_final\" → XOR 'Main triage' → Hospital|Clinic|Home.\n"
            "Flows from Main triage: Hospital if <conditionExpression xsi:type=\"tFormalExpression\">danger_sign == true</conditionExpression>; Clinic if <conditionExpression xsi:type=\"tFormalExpression\">clinic_referral == true</conditionExpression>; default to Home.\n"
            "Use variable name diarrhea exactly for the first gateway. Namespaces: xmlns:xsi, xmlns:bpmn=\"http://www.omg.org/spec/BPMN/20100524/MODEL\".\n"
        )

    # ------------------------- PDF text -------------------------

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text with explicit per-page markers for traceability."""
        out: List[str] = []
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                out.append(f"[PAGE {i}]\n{t}\n")
        return "\n".join(out).strip()

    # ------------------------- chunking -------------------------

    def _chunk_text(self, text: str) -> List[Tuple[int, str]]:
        """Chunk with overlap. Preserve page tags for refs."""
        chunks: List[Tuple[int, str]] = []
        i = 0
        n = len(text)
        idx = 0
        while i < n:
            j = min(n, i + self.chunk_chars)
            chunk = text[i:j]
            chunks.append((idx, chunk))
            idx += 1
            if j >= n:
                break
            i = j - self.overlap_chars
            if i < 0:
                i = 0
        return chunks

    # ------------------------- OpenAI call -------------------------

    def _call_openai_json(self, system: str, user: str) -> Dict[str, Any]:
        """Call OpenAI with a JSON-only expectation. Fallback parse if needed."""
        # Try response_format if available
        try:
            resp = self.client.chat.completions.create(
                model=self.model_extraction,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=6000,
                response_format={"type": "json_object"},  # may be ignored by some models
            )
        except Exception:
            resp = self.client.chat.completions.create(
                model=self.model_extraction,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=6000,
            )
        content = resp.choices[0].message.content or ""
        # Strip code fences if present
        content = content.strip()
        if content.startswith("```"):
            # find first newline after marker
            first = content.find("\n")
            if first != -1:
                content = content[first + 1 :]
            if content.endswith("```"):
                content = content[:-3]
        try:
            return json.loads(content)
        except Exception:
            # Attempt to locate first JSON object
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = content[start : end + 1]
                return json.loads(snippet)
            raise ValueError("Model did not return valid JSON")

    # ------------------------- high-level extraction -------------------------

    def extract_rules_from_pdf(self, pdf_path: str, module_name: str = "extracted") -> Dict[str, Any]:
        """Pipeline: PDF → chunks → per-chunk JSON → merged normalized rules object."""
        text = self.extract_text_from_pdf(pdf_path)
        return self.extract_rules_from_text(text, module_name=module_name)

    def extract_rules_from_text(self, text: str, module_name: str = "text_rules") -> Dict[str, Any]:
        chunks = self._chunk_text(text)
        merged: Dict[str, Any] = {"variables": [], "rules": []}

        for idx, chunk in chunks:
            user_prompt = (
                "Analyze WHO CHW guideline text. Extract variables and rules as per schema. "
                f"Module: {module_name}. Chunk {idx+1}/{len(chunks)}. "
                "Preserve any visible [PAGE N] markers in refs."
                "\n\nTEXT:\n" + chunk
            )
            out_obj = self._call_openai_json(self.system_prompt, user_prompt)
            merged = self._merge_extractions(merged, out_obj)

        normalized = self._normalize_merged(merged)
        cleaned = self.validate_and_clean_rules_obj(normalized)
        return cleaned

    # ------------------------- merge and normalize -------------------------

    def _merge_extractions(self, acc: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        # Merge variables by canonical name
        existing = {to_snake(v.get("name", "")): v for v in acc.get("variables", []) if isinstance(v, dict)}
        for v in new.get("variables", []) or []:
            if not isinstance(v, dict):
                continue
            name = to_snake(str(v.get("name", "")))
            if not name:
                continue
            cur = existing.get(name)
            if cur is None:
                # normalize fields
                existing[name] = {
                    "name": name,
                    "type": v.get("type", "string"),
                    "unit": v.get("unit"),
                    "allowed": v.get("allowed"),
                    "synonyms": sorted(list({to_snake(x) for x in (v.get("synonyms") or [])})),
                    "prompt": v.get("prompt") or name.replace("_", " "),
                    "refs": sorted(list({str(x) for x in (v.get("refs") or [])})),
                }
            else:
                cur["type"] = cur.get("type") or v.get("type")
                cur["unit"] = cur.get("unit") or v.get("unit")
                # merge lists
                cur["allowed"] = cur.get("allowed") or v.get("allowed")
                cur["synonyms"] = sorted(list({*cur.get("synonyms", []), *[to_snake(x) for x in (v.get("synonyms") or [])]}))
                cur["refs"] = sorted(list({*cur.get("refs", []), *[str(x) for x in (v.get("refs") or [])]}))
                if not cur.get("prompt"):
                    cur["prompt"] = v.get("prompt") or name.replace("_", " ")
        merged_vars = list(existing.values())

        # Merge rules by signature
        acc_rules = acc.get("rules", []) or []
        new_rules = new.get("rules", []) or []
        all_rules = acc_rules + new_rules
        dedup: Dict[Tuple, Dict[str, Any]] = {}
        for r in all_rules:
            if not isinstance(r, dict):
                continue
            sig = stable_rule_signature(r)
            # Keep higher priority or the one with more specific conditions
            keep = dedup.get(sig)
            if keep is None:
                dedup[sig] = r
            else:
                p_new = r.get("then", {}).get("priority", 0)
                p_old = keep.get("then", {}).get("priority", 0)
                if p_new > p_old or len(r.get("when", [])) > len(keep.get("when", [])):
                    dedup[sig] = r

        merged_rules = list(dedup.values())
        return {"variables": merged_vars, "rules": merged_rules}

    def _normalize_merged(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        # normalize variables
        for v in obj.get("variables", []):
            v["name"] = to_snake(str(v.get("name", "")))
            t = str(v.get("type", "string")).lower()
            v["type"] = "boolean" if t.startswith("bool") else "number" if t in {"number", "float", "double", "int", "integer"} else "string"
            if v.get("allowed") is not None and not isinstance(v["allowed"], list):
                v["allowed"] = [v["allowed"]]
        # normalize rules
        for i, r in enumerate(obj.get("rules", []), start=1):
            r.setdefault("rule_id", f"R-{i:04d}")
            # normalize when
            when = r.get("when", [])
            if isinstance(when, dict):
                when = [when]
            fixed_when: List[Dict[str, Any]] = []
            for c in when or []:
                fc = self._fix_condition(c)
                if fc:
                    fixed_when.append(fc)
            r["when"] = fixed_when
            # normalize then
            then = r.get("then", {}) or {}
            triage = (then.get("propose_triage") or "").lower()
            if triage not in {"hospital", "clinic", "home"}:
                triage = "home"
            then["propose_triage"] = triage
            # flags
            flags = then.get("set_flags") or []
            if not isinstance(flags, list):
                flags = [flags]
            flags = [to_snake(str(x)) for x in flags]
            # invariants
            if triage == "hospital" and "danger_sign" not in flags:
                flags.append("danger_sign")
            if triage == "clinic" and "clinic_referral" not in flags:
                flags.append("clinic_referral")
            then["set_flags"] = sorted(list(dict.fromkeys(flags)))
            # reasons
            reasons = then.get("reasons") or []
            if not isinstance(reasons, list):
                reasons = [reasons]
            then["reasons"] = [to_snake(str(x)) for x in reasons if str(x).strip()]
            # actions
            actions = then.get("actions") or []
            if not isinstance(actions, list):
                actions = []
            then["actions"] = actions
            # guideline_ref
            gr = then.get("guideline_ref") or ""
            if not gr:
                # Try to infer from any page tag seen in reasons or elsewhere
                gr = "WHO-CHW-2012"
            then["guideline_ref"] = gr
            # priority
            pr = then.get("priority")
            if not isinstance(pr, int):
                # derive priority: hospital > clinic > home
                pr = 90 if triage == "hospital" else 60 if triage == "clinic" else 10
            then["priority"] = int(pr)
            r["then"] = then
        return obj

    def _fix_condition(self, c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(c, dict):
            return None
        # observation comparator
        if {"obs", "op", "value"} <= set(c.keys()):
            obs = to_snake(str(c["obs"]))
            op = str(c["op"]).lower()
            if op not in {"lt", "le", "gt", "ge", "eq", "ne"}:
                return None
            val = c["value"]
            # convert numbers when possible
            num = coerce_number(val)
            if num is not None:
                val = num
            else:
                # maybe boolean
                b = coerce_bool(val)
                if b is not None:
                    val = b
                else:
                    val = str(val)
            return {"obs": obs, "op": op, "value": val}

        # symptom boolean
        if "sym" in c:
            sym = to_snake(str(c["sym"]))
            eq = c.get("eq")
            b = coerce_bool(eq)
            if b is None:
                b = True
            return {"sym": sym, "eq": b}

        # any_of / all_of
        if "any_of" in c:
            inner = [self._fix_condition(x) for x in (c.get("any_of") or [])]
            inner = [x for x in inner if x]
            return {"any_of": inner} if inner else None
        if "all_of" in c:
            inner = [self._fix_condition(x) for x in (c.get("all_of") or [])]
            inner = [x for x in inner if x]
            return {"all_of": inner} if inner else None

        return None

    # ------------------------- validation -------------------------

    def validate_and_clean_rules_obj(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Validate structure, dedup conflicts, enforce invariants and report issues inline."""
        variables = obj.get("variables") or []
        rules = obj.get("rules") or []

        # basic type checks
        assert isinstance(variables, list), "variables must be a list"
        assert isinstance(rules, list), "rules must be a list"

        # build set of known variable names
        var_names = {to_snake(v.get("name", "")) for v in variables if isinstance(v, dict)}

        # ensure every referenced var exists
        missing_vars: Dict[str, set] = {}
        for r in rules:
            for c in r.get("when", []):
                for key in ("obs", "sym"):
                    if key in c:
                        name = to_snake(str(c[key]))
                        if name and name not in var_names:
                            missing_vars.setdefault(name, set()).add(r.get("rule_id", ""))
                            # auto-add variable with unknown type
                            variables.append({"name": name, "type": "string", "unit": None, "allowed": None, "synonyms": [], "prompt": name.replace("_", " "), "refs": []})
                            var_names.add(name)

        # detect overlaps for UNIQUE semantics: naive check by sampling boolean combinations is infeasible.
        # As a heuristic: if two rules have identical condition signatures but different proposed triage, flag conflict and drop lower priority.
        by_when: Dict[Tuple, Dict[str, Any]] = {}
        drop_ids = set()
        for r in rules:
            sig = stable_rule_signature({"when": r.get("when", []), "then": {}})
            exist = by_when.get(sig)
            if exist is None:
                by_when[sig] = r
            else:
                t_old = exist.get("then", {}).get("propose_triage")
                t_new = r.get("then", {}).get("propose_triage")
                if t_old != t_new:
                    # keep higher priority
                    p_old = exist.get("then", {}).get("priority", 0)
                    p_new = r.get("then", {}).get("priority", 0)
                    if p_new > p_old:
                        drop_ids.add(exist.get("rule_id"))
                        by_when[sig] = r
                    else:
                        drop_ids.add(r.get("rule_id"))

        rules = [r for r in rules if r.get("rule_id") not in drop_ids]

        # final pack
        return {"variables": variables, "rules": rules}

    # ------------------------- BPMN/DMN generation -------------------------

    def _collect_ref_vars(self, rules: List[Dict]) -> set:
        """Collect all variables referenced in rule conditions."""
        ref = set()
        def rec(c):
            if not isinstance(c, dict): return
            if "obs" in c: ref.add(c["obs"])
            elif "sym" in c: ref.add(c["sym"])
            elif "any_of" in c: [rec(x) for x in c["any_of"]]
            elif "all_of" in c: [rec(x) for x in c["all_of"]]
        for r in rules:
            for c in r.get("when", []): rec(c)
        return {str(v).strip().lower().replace(" ","_") for v in ref}

    def _strip_circular_inputs(self, rules_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove conditions that reference derived outputs like danger_sign or clinic_referral."""
        banned = {"danger_sign", "clinic_referral", "triage", "danger_sign_present", "has_danger_sign"}
        cleaned: Dict[str, Any] = {"variables": rules_data.get("variables", []), "rules": []}
        for r in rules_data.get("rules", []):
            when = r.get("when", [])
            new_when = []
            for c in when:
                if isinstance(c, dict):
                    ok = True
                    if "obs" in c and str(c.get("obs")).strip().lower().replace(" ", "_") in banned:
                        ok = False
                    if "sym" in c and str(c.get("sym")).strip().lower().replace(" ", "_") in banned:
                        ok = False
                    if ok:
                        new_when.append(c)
            r2 = dict(r)
            r2["when"] = new_when
            cleaned["rules"].append(r2)
        return cleaned

    def _extract_fenced_blocks(self, text: str) -> List[Tuple[str, str]]:
        """Extract fenced code blocks in order as (lang, content)."""
        blocks: List[Tuple[str, str]] = []
        i = 0
        while True:
            start = text.find("```", i)
            if start == -1:
                break
            nl = text.find("\n", start)
            if nl == -1:
                break
            lang = text[start+3:nl].strip() or ""
            end = text.find("```", nl+1)
            if end == -1:
                break
            content = text[nl+1:end].strip()
            blocks.append((lang, content))
            i = end + 3
        return blocks

    def generate_modular_dmn_package(
        self,
        rules_data: Dict[str, Any],
        module_name: str = "WHO_CHW_Modular",
        who_pdf_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate modular DMN + CANONICAL_MAP + QA_REPORT + ASK_PLAN using the two-call prompt (DMN first)."""
        # Guardrail: strip circular inputs before prompting
        cleaned_rules = self._strip_circular_inputs(rules_data)
        user = (
            "RULES_JSON:\n" + json.dumps(cleaned_rules, separators=(",", ":")) +
            ("\n\nOPTIONAL_WHO_PDF_TEXT:\n" + who_pdf_text if who_pdf_text else "")
        )
        resp = self.client.chat.completions.create(
            model=self.model_extraction,
            messages=[{"role": "system", "content": self.modular_dmn_system_prompt}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=12000,
        )
        content = (resp.choices[0].message.content or "").strip()
        blocks = self._extract_fenced_blocks(content)
        # Helper: loose JSON parse with comment stripping
        def parse_json_loose(text: str) -> Any:
            cleaned_lines = []
            for ln in text.splitlines():
                ls = ln.strip()
                if ls.startswith("//"):
                    continue
                cleaned_lines.append(ln)
            cleaned = "\n".join(cleaned_lines).strip()
            return json.loads(cleaned)

        dmn_xml: Optional[str] = None
        canonical_map: Optional[Dict[str, Any]] = None
        qa_report: Optional[Dict[str, Any]] = None
        ask_plan: Optional[Any] = None

        # Iterate all blocks; pick first DMN and first plausible JSONs
        for lang, body in blocks:
            b = body.strip()
            if (dmn_xml is None) and ("<dmn:definitions" in b and "</dmn:definitions>" in b):
                dmn_xml = b
                continue
            # Try JSON
            try:
                obj = parse_json_loose(b)
            except Exception:
                continue
            # Classify
            if ask_plan is None and isinstance(obj, list):
                ask_plan = obj
                continue
            if isinstance(obj, dict):
                # qa_report heuristic: has some QA-like keys
                if qa_report is None and any(k in obj for k in ["triage_counts", "overlap_conflicts_fixed", "unmapped_variables"]):
                    qa_report = obj
                    continue
                if canonical_map is None:
                    canonical_map = obj

        if not (dmn_xml and canonical_map is not None and qa_report is not None and ask_plan is not None):
            # Save raw for debugging
            try:
                from pathlib import Path
                Path("modular_dmn_debug.txt").write_text(content, encoding="utf-8")
            except Exception:
                pass
            raise RuntimeError("Failed to parse modular DMN package blocks")
        return {
            "dmn_xml": dmn_xml,
            "canonical_map": canonical_map,
            "qa_report": qa_report,
            "ask_plan": ask_plan,
            "raw": content,
        }

    def generate_bpmn_from_modular(self, dmn_xml: str, canonical_map: Dict[str, Any], ask_plan: Any) -> str:
        """Generate BPMN from modular DMN, canonical map, and ask plan (second call)."""
        user = (
            "Here is the DMN and canonical/ask info to integrate.\n\n" 
            "CANONICAL_MAP.json:\n" + json.dumps(canonical_map) + "\n\n"
            "ASK_PLAN.json:\n" + json.dumps(ask_plan) + "\n\n"
            "DMN:\n```xml\n" + dmn_xml + "\n```\n"
        )
        resp = self.client.chat.completions.create(
            model=self.model_extraction,
            messages=[{"role": "system", "content": self.modular_bpmn_system_prompt}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=6000,
        )
        content = (resp.choices[0].message.content or "").strip()
        # Reuse BPMN extractor
        bpmn_xml = self._extract_bpmn_xml(content)
        if not bpmn_xml:
            raise RuntimeError("Modular BPMN generation failed")
        return bpmn_xml

    def generate_dmn_only(self, rules_data: Dict[str, Any], module_name: str = "WHO_CHW") -> str:
        """Generate DMN decision model only."""
        
        DMN_SYSTEM = """You output a minimal DMN 1.4 decision model as a single <dmn:definitions>. No prose.
Rules:
- Use only the MODEL namespace: https://www.omg.org/spec/DMN/20191111/MODEL/
- One decision id="decide_triage" name="Decide Triage" with one decisionTable hitPolicy="UNIQUE"
- Create <dmn:inputData> only for variables referenced in RULES_JSON conditions
- Columns: one input per referenced variable, FEEL basics only (= != < <= > >= true false "string")
- In <dmn:text> escape < and > as &lt; and &gt;
- Output column name="effect" typeRef="string" with tokens:
  triage:hospital|clinic|home
  flag:danger_sign when triage:hospital
  flag:clinic_referral when triage:clinic
  reason:short_snake_case
  ref:stable_ref
- Map rules:
  any_of → multiple rows
  all_of → conjunction in one row
  obs uses given op and literal
  sym uses true or false
- Enforce UNIQUE by tightening or splitting overlaps
- Do not include DMNDI or DI
- End with required <dmn:informationRequirement> links to each inputData
Return exactly one fenced xml code block with <dmn:definitions>."""

        ref_vars = self._collect_ref_vars(rules_data.get("rules", []))
        filtered = {
            "variables": [v for v in rules_data.get("variables", []) if v.get("name") in ref_vars],
            "rules": rules_data.get("rules", []),
        }
        
        user = "RULES_JSON:\n" + json.dumps(filtered, separators=(',',':')) + f"\nMODULE_NAME:{module_name}"
        
        resp = self.client.chat.completions.create(
            model=self.model_extraction,
            messages=[{"role":"system","content":DMN_SYSTEM},{"role":"user","content":user}],
            temperature=0.0,
            max_tokens=12000,
        )
        
        text = resp.choices[0].message.content.strip()
        dmn_xml = self._extract_dmn_xml(text)
        if not dmn_xml: 
            print(f"🐛 DMN generation failed. Response: {text[:500]}...")
            raise RuntimeError("DMN not found in OpenAI response")
        return dmn_xml

    def generate_bpmn_from_dmn(self, dmn_xml: str) -> str:
        """Generate BPMN process that references the DMN."""
        
        BPMN_SYSTEM = """You output a BPMN 2.0 process as a single <bpmn:definitions>. No prose.
Interview policy:
- Start by asking about diarrhea, branch on diarrhea_present
- Gather only relevant observations, then call the DMN, then route

Requirements:
- Namespaces: bpmn MODEL + xsi
- <bpmn:process id="chatchw_flow" name="ChatCHW Clinical Interview" isExecutable="false">
- Order:
  startEvent "Start"
  userTask "Ask diarrhea"
  exclusiveGateway "Diarrhea present?"
    if diarrhea_present == true → userTask "Collect diarrhea details" → userTask "Ask fever/malaria"
    else → userTask "Ask fever/malaria"
  userTask "Ask cough/pneumonia"
  userTask "Ask malnutrition"
  businessRuleTask "Evaluate decision (DMN)" that references decision id "decide_triage"
  exclusiveGateway "Main triage" with default to Home
  endEvents "Hospital" "Clinic" "Home"
- Gateway conditions:
  to Hospital: <conditionExpression xsi:type="tFormalExpression">danger_sign == true</conditionExpression>
  to Clinic:   <conditionExpression xsi:type="tFormalExpression">clinic_referral == true</conditionExpression>
  default to Home: no condition
- Use variable name diarrhea_present in expressions
Return exactly one fenced xml code block with <bpmn:definitions>."""

        user = "Here is the DMN to integrate:\n```xml\n" + dmn_xml + "\n```\nGenerate the BPMN per spec."
        
        resp = self.client.chat.completions.create(
            model=self.model_extraction,
            messages=[{"role":"system","content":BPMN_SYSTEM},{"role":"user","content":user}],
            temperature=0.0,
            max_tokens=4000,
        )
        
        text = resp.choices[0].message.content.strip()
        bpmn_xml = self._extract_bpmn_xml(text)
        if not bpmn_xml:
            print(f"🐛 BPMN generation failed. Response: {text[:500]}...")
            raise RuntimeError("BPMN not found in OpenAI response")
        return bpmn_xml

    def generate_bpmn_dmn_from_rules(self, rules_data: Dict[str, Any], module_name: str = "WHO_CHW") -> Dict[str, str]:
        """Generate both BPMN and DMN using the split approach: DMN first, then BPMN."""
        print(f"🤖 Generating BPMN and DMN using split approach...")
        print(f"📊 Processing {len(rules_data.get('variables', []))} variables and {len(rules_data.get('rules', []))} rules")
        
        try:
            # Step 1: Generate DMN first
            print("🔄 Step 1: Generating DMN decision model...")
            dmn_xml = self.generate_dmn_only(rules_data, module_name)
            print("✅ DMN generated successfully")
            
            # Step 2: Generate BPMN that references the DMN
            print("🔄 Step 2: Generating BPMN workflow that references DMN...")
            bpmn_xml = self.generate_bpmn_from_dmn(dmn_xml)
            print("✅ BPMN generated successfully")
            
            return {
                "bpmn_xml": bpmn_xml,
                "dmn_xml": dmn_xml,
                "raw_response": f"Split generation completed:\n1. DMN: {len(dmn_xml)} chars\n2. BPMN: {len(bpmn_xml)} chars"
            }
            
        except Exception as e:
            raise RuntimeError(f"Split BPMN/DMN generation failed: {e}")

    def _extract_bpmn_xml(self, response_text: str) -> Optional[str]:
        """Extract BPMN XML from OpenAI response."""
        # Look for BPMN XML blocks
        blocks = response_text.split("```xml")
        for block in blocks[1:]:  # Skip first split part
            if "</bpmn:definitions>" in block:
                # Extract until closing tag
                bpmn_start = block.find("<bpmn:definitions")
                if bpmn_start != -1:
                    bpmn_end = block.find("</bpmn:definitions>", bpmn_start)
                    if bpmn_end != -1:
                        bpmn_end += len("</bpmn:definitions>")
                        return block[bpmn_start:bpmn_end].strip()
        return None

    def _extract_dmn_xml(self, response_text: str) -> Optional[str]:
        """Extract DMN XML from OpenAI response."""
        # Look for DMN XML blocks  
        blocks = response_text.split("```xml")
        for block in blocks[1:]:  # Skip first split part
            if "</dmn:definitions>" in block:
                # Extract until closing tag
                dmn_start = block.find("<dmn:definitions")
                if dmn_start != -1:
                    dmn_end = block.find("</dmn:definitions>", dmn_start)
                    if dmn_end != -1:
                        dmn_end += len("</dmn:definitions>")
                        return block[dmn_start:dmn_end].strip()
        return None

# End of module
