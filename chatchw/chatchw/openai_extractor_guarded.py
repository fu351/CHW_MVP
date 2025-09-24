"""
OpenAI-only, guarded extractor implementing a 5-step pipeline:
1) Per-section extraction → JSON IR
2) Global merge + canonicalization → merged IR
3) Modular DMN (FIRST) + ASK_PLAN
4) BPMN (diarrhea-first) from DMN + ASK_PLAN
5) Coverage audit mapping RULES_JSON to DMN rows

All steps use strict prompts and robust parsing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pypdf

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


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


class OpenAIGuardedExtractor:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o") -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required")
        if OpenAI is None:
            raise RuntimeError("openai package not available")
        self.client = OpenAI(api_key=key)
        self.model = model

        # Step 1 prompt (per-section)
        self.section_system = (
            "You extract WHO CHW clinical rules. Return ONLY JSON.\n\n"
            "SCHEMA\n{\n  \"variables\":[{\"name\":snake,\"type\":\"number|boolean|string\",\"unit\":null|unit,\"allowed\":null|[...],\n                \"synonyms\":[snake...],\"prompt\":short,\"refs\":[page_or_section]}],\n  \"rules\":[{\"rule_id\":str,\n            \"when\":[ {\"obs\":var,\"op\":\"lt|le|gt|ge|eq|ne\",\"value\":num|bool|string} |\n                     {\"sym\":var,\"eq\":true|false} |\n                     {\"all_of\":[COND...]} | {\"any_of\":[COND...]} ],\n            \"then\":{\"triage\":\"hospital|clinic|home\",\n                    \"flags\":[snake...],\n                    \"reasons\":[snake...],\n                    \"actions\":[{\"id\":snake,\"if_available\":bool}],\n                    \"guideline_ref\":str,\"priority\":int}}],\n  \"canonical_map\":{},\n  \"qa\":{\"notes\":[]}\n}\n\n"
            "RULES\n- Use canonical names if present: diarrhea, blood_in_stool, diarrhea_duration_days, fever, temperature_c, fever_duration_days, cough, cough_duration_days, resp_rate, age_months, chest_indrawing, muac_mm, edema_both_feet, convulsions, unconscious, unable_to_drink, vomiting_everything, malaria_area\n- Add all seen aliases into synonyms and rewrite conditions to the canonical\n- No derived outputs in conditions (ban danger_sign, clinic_referral, triage)\n- Encode literal thresholds only. No invented cutoffs. No “..”\n- Priority tiers: hospital≥90, clinic 50–89, home<50\n- Every rule gets guideline_ref like \"p41\" or section id\n\n"
            "OUTPUT\nOnly the JSON object for THIS section"
        )

        # Step 2 prompt (merge)
        self.merge_system = (
            "Merge these section JSON objects into one comprehensive IR. Return ONLY JSON with the same schema as step 1 plus:\n"
            "- canonical_map filled for all variables\n- qa.unmapped_vars\n- qa.dedup_dropped\n- qa.overlap_fixed\nRULES\n- Rewrite all rules to canonical names\n- If two rules conflict on the same condition set tighten bounds or split any_of so both can exist without overlap\n- Keep every literal threshold\nINPUT\n<PASTE ALL SECTION JSONs>"
        )

        # Step 3 prompt (DMN + ASK_PLAN) - enforced aggregator inputs and per-sign rows
        self.dmn_system = (
            "Convert RULES_JSON into modular DMN 1.4 using the DMN 1.4 MODEL namespace (2019-11-11). Return exactly two fenced blocks:\n"
            "1) ```xml <dmn:definitions>…```  2) ```json ASK_PLAN```\n\n"
            "DMN REQUIREMENTS\n"
            "- Root tag MUST be <dmn:definitions xmlns:dmn=\"https://www.omg.org/spec/DMN/20191111/MODEL/\">. No other DMN namespaces.\n"
            "- Decisions: decide_danger_signs, decide_diarrhea, decide_fever_malaria, decide_respiratory, decide_nutrition, aggregate_final\n"
            "- Each module decision uses <dmn:decisionTable hitPolicy=\"FIRST\">\n  Inputs: only relevant canonical variables\n  Outputs: triage:string, danger_sign:boolean, clinic_referral:boolean, reason:string, ref:string\n  Rows: use FEEL literals/comparators only; escape &lt; and &gt; in <dmn:text>.\n  Invariants: hospital → danger_sign=true; clinic → clinic_referral=true. No effect strings.\n"
            "- decide_danger_signs MUST have one row per sign (no row requiring multiple signs):\n  convulsions==true → triage:\"hospital\", danger_sign:true, reason:\"convulsions\", ref:\"pXX\"\n  unconscious==true → triage:\"hospital\", danger_sign:true, reason:\"unconscious\"\n  unable_to_drink==true → triage:\"hospital\", danger_sign:true, reason:\"unable_to_drink\"\n  vomiting_everything==true → triage:\"hospital\", danger_sign:true, reason:\"vomiting_everything\"\n  chest_indrawing==true → triage:\"hospital\", danger_sign:true, reason:\"severe_respiratory_distress\"\n  muac_mm < 115 → triage:\"clinic\", clinic_referral:true, reason:\"severe_acute_malnutrition\"\n  edema_both_feet==true → triage:\"clinic\", clinic_referral:true, reason:\"bilateral_pitting_edema\"\n"
            "- decide_diarrhea:\n  blood_in_stool==true → clinic_referral:true, triage:\"clinic\", reason:\"dysentery\"\n  diarrhea_duration_days >= 14 → clinic_referral:true, triage:\"clinic\", reason:\"persistent_diarrhea\"\n"
            "- decide_fever_malaria:\n  malaria_area==true AND fever==true AND fever_duration_days in [3..6] → clinic_referral:true\n  fever==true AND temperature_c >= 39 → clinic_referral:true\n  fever==true AND fever_duration_days >= 7 → clinic_referral:true\n"
            "- decide_respiratory (age thresholds, do not duplicate chest_indrawing clinic):\n  age_months < 12 AND resp_rate >= 50 → clinic_referral:true, reason:\"fast_breathing_infant\"\n  age_months >= 12 AND resp_rate >= 40 → clinic_referral:true, reason:\"fast_breathing_child\"\n"
            "- decide_nutrition: MUAC 115–124 → clinic_referral:true; >=125 → home\n"
            "- aggregate_final MUST HAVE INPUT COLUMNS referencing module outputs:\n  Inputs: decide_danger_signs.danger_sign (boolean), decide_diarrhea.clinic_referral, decide_fever_malaria.clinic_referral, decide_respiratory.clinic_referral, decide_nutrition.clinic_referral\n  Rules (FIRST):\n   Row1: danger_sign == true → triage:\"hospital\", reason:\"danger_sign\", ref:\"aggregator\"\n   Row2: any clinic_referral input == true → triage:\"clinic\", reason:\"needs_clinic_referral\", ref:\"aggregator\"\n   Row3: otherwise → triage:\"home\", reason:\"home_care\", ref:\"aggregator\"\n  The aggregator does NOT compute per-module reasons; engine will surface top module reason/ref.\n"
            "- Add <dmn:informationRequirement> from aggregate_final to each module decision.\n"
            "- Ensure <dmn:inputData> exists for every canonical variable (boolean|number|string).\n"
            "- No contains() or non-DMN elements.\n\n"
            "ASK_PLAN (use canonical names; parent-first with gated follow-ups):\n"
            "[ {\"module\":\"diarrhea\",\"ask\":[\"diarrhea\"],\"followups_if\":{\"diarrhea==true\":[\"blood_in_stool\",\"diarrhea_duration_days\"]}},\n  {\"module\":\"fever_malaria\",\"ask\":[\"fever\"],\"followups_if\":{\"fever==true\":[\"temperature_c\",\"fever_duration_days\",\"malaria_area\"]}},\n  {\"module\":\"respiratory\",\"ask\":[\"cough\"],\"followups_if\":{\"cough==true\":[\"resp_rate\",\"cough_duration_days\",\"chest_indrawing\"]}},\n  {\"module\":\"nutrition\",\"ask\":[\"muac_mm\",\"edema_both_feet\"]},\n  {\"module\":\"danger_signs\",\"ask\":[\"convulsions\",\"unconscious\",\"unable_to_drink\",\"vomiting_everything\",\"chest_indrawing\",\"edema_both_feet\"]}]\n"
        )

        # Step 4 prompt (BPMN)
        self.bpmn_system = (
            "Produce one BPMN 2.0 <bpmn:definitions> only. Use xmlns:bpmn=\"http://www.omg.org/spec/BPMN/20100524/MODEL\" and xmlns:xsi. No vendor extensions.\n\n"
            "Process id=\"chatchw_flow\" isExecutable=\"false\"\n"
            "Start → userTask \"Ask diarrhea\" → XOR \"Diarrhea present?\"\n true → userTask \"Collect diarrhea details\" → userTask \"Ask fever/malaria\"\n false → userTask \"Ask fever/malaria\"\n→ userTask \"Ask cough/pneumonia\" → userTask \"Ask malnutrition\"\n→ businessRuleTask \"Evaluate decisions\" decisionRef=\"aggregate_final\"\n→ XOR \"Main triage\"\n  flow to \"Hospital\" with <conditionExpression xsi:type=\"tFormalExpression\">danger_sign == true</conditionExpression>\n  flow to \"Clinic\" with <conditionExpression xsi:type=\"tFormalExpression\">clinic_referral == true</conditionExpression>\n  default flow to \"Home\"\nUse canonical variable names exactly. Valid namespaces. No prose.\nINPUT\nProvide the DMN you just produced and the ASK_PLAN to align variable names"
        )

        # Step 5 prompt (coverage)
        self.coverage_system = (
            "Given RULES_JSON and the DMN XML, return ONLY JSON:\n{\"unmapped_rule_ids\":[...], \"module_counts\":{...}, \"notes\":[...]}\nRules are mapped if their literal conditions appear as inputEntry cells in some module row with same triage tier.\nINPUT\nRULES_JSON: <merged IR>\nDMN: <xml>"
        )

    # ---------------- PDF sectioning -----------------
    def extract_sections_from_pdf(self, pdf_path: str) -> List[Tuple[str, str]]:
        """Return a list of (section_id, text) with page markers retained.
        Heuristic: use page chunks; caller can be improved later.
        """
        out: List[Tuple[str, str]] = []
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                sec_id = f"p{i}"
                out.append((sec_id, f"[PAGE {i}]\n{t}"))
        return out

    # ---------------- OpenAI helpers -----------------
    def _chat_json(self, system: str, user: str, max_tokens: int = 6000) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        raw = _strip_code_fences(content)
        # First try strict JSON
        try:
            return json.loads(raw)
        except Exception:
            pass
        # Loose cleanup: strip // and /* */ comments, and trailing commas
        cleaned_lines: List[str] = []
        in_block_comment = False
        for ln in raw.splitlines():
            s = ln
            if in_block_comment:
                if "*/" in s:
                    s = s.split("*/", 1)[1]
                    in_block_comment = False
                else:
                    continue
            if "/*" in s:
                before, _after = s.split("/*", 1)
                s = before
                in_block_comment = True
            # strip // comments
            idx = s.find('//')
            if idx != -1:
                s = s[:idx]
            cleaned_lines.append(s)
        cleaned = "\n".join(cleaned_lines)
        # remove trailing commas before ] or }
        import re
        cleaned = re.sub(r",\s*(\]|\})", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            # Attempt bracket-balanced extraction of the first JSON object
            s = cleaned
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = s[start:end+1]
                # try again after trailing comma cleanup
                candidate = re.sub(r",\s*(\]|\})", r"\1", candidate)
                try:
                    return json.loads(candidate)
                except Exception:
                    pass
            # Save debug output
            try:
                Path("guarded_debug_last.txt").write_text(content, encoding="utf-8")
            except Exception:
                pass
            raise

    def _chat_text(self, system: str, user: str, max_tokens: int = 12000) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    # ---------------- Schema guards -----------------
    def _ensure_section_schema(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure presence and types for section JSON
        out: Dict[str, Any] = {}
        out["variables"] = obj.get("variables") if isinstance(obj.get("variables"), list) else []
        out["rules"] = obj.get("rules") if isinstance(obj.get("rules"), list) else []
        out["canonical_map"] = obj.get("canonical_map") if isinstance(obj.get("canonical_map"), dict) else {}
        qa = obj.get("qa") if isinstance(obj.get("qa"), dict) else {}
        if not isinstance(qa.get("notes"), list):
            qa["notes"] = []
        out["qa"] = qa
        return out

    def _ensure_merged_schema(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure presence for merged IR
        out = self._ensure_section_schema(obj)
        qa = out.get("qa", {})
        if not isinstance(qa.get("unmapped_vars"), list):
            qa["unmapped_vars"] = []
        if not isinstance(qa.get("dedup_dropped"), int):
            qa["dedup_dropped"] = 0
        if not isinstance(qa.get("overlap_fixed"), list):
            qa["overlap_fixed"] = []
        out["qa"] = qa
        return out

    # ---------------- Step 1: per-section -----------------
    def extract_rules_per_section(self, pdf_path: str) -> List[Dict[str, Any]]:
        sections = self.extract_sections_from_pdf(pdf_path)
        results: List[Dict[str, Any]] = []
        for sec_id, text in sections:
            user = f"SECTION_ID: {sec_id}\n\nTEXT:\n{text}"
            try:
                obj = self._chat_json(self.section_system, user)
                results.append(self._ensure_section_schema(obj))
            except Exception:
                # Skip bad sections but keep going
                continue
        return results

    # ---------------- Step 2: merge -----------------
    def merge_sections(self, section_objs: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = json.dumps(section_objs, ensure_ascii=False)
        user = f"INPUT\n{payload}"
        try:
            merged = self._chat_json(self.merge_system, user)
            return self._ensure_merged_schema(merged)
        except Exception:
            # Fallback: deterministic merge in Python to guarantee JSON
            variables: Dict[str, Dict[str, Any]] = {}
            rules: List[Dict[str, Any]] = []
            for sec in section_objs:
                for v in (sec.get("variables") or []):
                    if not isinstance(v, dict):
                        continue
                    name = str(v.get("name", "")).strip().lower()
                    if not name:
                        continue
                    cur = variables.get(name)
                    if cur is None:
                        # shallow copy and normalize lists
                        vv = dict(v)
                        if not isinstance(vv.get("synonyms"), list):
                            vv["synonyms"] = []
                        if not isinstance(vv.get("refs"), list):
                            vv["refs"] = []
                        variables[name] = vv
                    else:
                        # merge synonyms/refs; keep first type/unit
                        syn = set(cur.get("synonyms") or []) | set(v.get("synonyms") or [])
                        refs = set(cur.get("refs") or []) | set(v.get("refs") or [])
                        cur["synonyms"] = sorted(list(syn))
                        cur["refs"] = sorted(list(refs))
                for r in (sec.get("rules") or []):
                    if isinstance(r, dict):
                        rules.append(r)
            merged_py = {
                "variables": list(variables.values()),
                "rules": rules,
                "canonical_map": {},
                "qa": {"notes": ["fallback_merge_used"], "unmapped_vars": [], "dedup_dropped": 0, "overlap_fixed": []},
            }
            return self._ensure_merged_schema(merged_py)

    # ---------------- Step 3: DMN + ASK_PLAN -----------------
    def generate_dmn_and_ask_plan(self, merged_ir: Dict[str, Any]) -> Tuple[str, Any]:
        user = "RULES_JSON:\n" + json.dumps(merged_ir, ensure_ascii=False)
        text = self._chat_text(self.dmn_system, user, max_tokens=12000)
        # Save raw for troubleshooting
        try:
            Path("dmn_ask_debug_last.txt").write_text(text, encoding="utf-8")
        except Exception:
            pass
        blocks = _extract_fenced_blocks(text)
        dmn_xml: Optional[str] = None
        ask_plan: Optional[Any] = None
        
        def _parse_json_loose(s: str) -> Optional[Any]:
            import re
            raw = _strip_code_fences(s).strip()
            # quick strict parse
            try:
                return json.loads(raw)
            except Exception:
                pass
            # strip // comments and /* */
            lines: List[str] = []
            in_block = False
            for ln in raw.splitlines():
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
                idx = t.find('//')
                if idx != -1:
                    t = t[:idx]
                lines.append(t)
            cleaned = "\n".join(lines)
            cleaned = re.sub(r",\s*(\]|\})", r"\1", cleaned)
            try:
                return json.loads(cleaned)
            except Exception:
                # attempt to find array or object
                lb, rb = cleaned.find('['), cleaned.rfind(']')
                if lb != -1 and rb != -1 and rb > lb:
                    cand = cleaned[lb:rb+1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        pass
                lb, rb = cleaned.find('{'), cleaned.rfind('}')
                if lb != -1 and rb != -1 and rb > lb:
                    cand = cleaned[lb:rb+1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        pass
            return None
        # First pass: use fenced blocks
        for lang, body in blocks:
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
            # Try to locate first plausible JSON array/object in whole text
            parsed = _parse_json_loose(text)
            if isinstance(parsed, (list, dict)):
                ask_plan = parsed
        # Normalize ASK_PLAN if dict-wrapped
        if isinstance(ask_plan, dict):
            # Common shapes: {"ASK_PLAN": [...]} or {"questions": [...]}
            if isinstance(ask_plan.get("ASK_PLAN"), list):
                ask_plan = ask_plan["ASK_PLAN"]
            elif isinstance(ask_plan.get("ask_plan"), list):
                ask_plan = ask_plan["ask_plan"]
            elif isinstance(ask_plan.get("questions"), list):
                ask_plan = {"questions": ask_plan["questions"]}
        # Minimal sanitizer to fix malformed <dmn:text> closures inside entries
        def _sanitize_dmn(xml: str) -> str:
            import re
            s = xml
            # Ensure every <dmn:outputEntry><dmn:text>… has a closing </dmn:text>
            s = re.sub(r"(<dmn:outputEntry>\s*<dmn:text>)([^<]*?)(</dmn:outputEntry>)",
                       r"\1\2</dmn:text>\3", s, flags=re.DOTALL)
            # Ensure empty output entries have explicit empty string
            s = s.replace("<dmn:outputEntry><dmn:text></dmn:text></dmn:outputEntry>",
                          "<dmn:outputEntry><dmn:text>\"\"</dmn:text></dmn:outputEntry>")
            s = s.replace("<dmn:outputEntry><dmn:text></dmn:outputEntry>",
                          "<dmn:outputEntry><dmn:text>\"\"</dmn:text></dmn:outputEntry>")
            # Do the same for inputEntry
            s = re.sub(r"(<dmn:inputEntry>\s*<dmn:text>)([^<]*?)(</dmn:inputEntry>)",
                       r"\1\2</dmn:text>\3", s, flags=re.DOTALL)
            s = s.replace("<dmn:inputEntry><dmn:text></dmn:text></dmn:inputEntry>",
                          "<dmn:inputEntry><dmn:text>-</dmn:text></dmn:inputEntry>")
            s = s.replace("<dmn:inputEntry><dmn:text></dmn:inputEntry>",
                          "<dmn:inputEntry><dmn:text>-</dmn:text></dmn:inputEntry>")
            return s

        if dmn_xml:
            dmn_xml = _sanitize_dmn(dmn_xml)

        if not dmn_xml or ask_plan is None:
            # Save debug
            try:
                Path("dmn_ask_debug_last.txt").write_text(text, encoding="utf-8")
            except Exception:
                pass
            raise RuntimeError("Failed to get DMN and ASK_PLAN from model")
        return dmn_xml, ask_plan

    # ---------------- Step 4: BPMN -----------------
    def generate_bpmn(self, dmn_xml: str, ask_plan: Any) -> str:
        user = (
            "DMN:\n```xml\n" + dmn_xml + "\n```\n\n" +
            "ASK_PLAN:\n" + json.dumps(ask_plan, ensure_ascii=False)
        )
        text = self._chat_text(self.bpmn_system, user, max_tokens=6000)
        blocks = _extract_fenced_blocks(text)
        for lang, body in blocks:
            if "<bpmn:definitions" in body:
                return body.strip()
        raise RuntimeError("Failed to get BPMN from model")

    # ---------------- Step 5: coverage -----------------
    def audit_coverage(self, merged_ir: Dict[str, Any], dmn_xml: str) -> Dict[str, Any]:
        user = "RULES_JSON:\n" + json.dumps(merged_ir, ensure_ascii=False) + "\nDMN:\n```xml\n" + dmn_xml + "\n```"
        return self._chat_json(self.coverage_system, user)


