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

        # Step 3 prompt (DMN + ASK_PLAN)
        self.dmn_system = (
            "Convert RULES_JSON into modular DMN 1.4 using the DMN 1.4 MODEL namespace (2019-11-11). Return exactly two fenced blocks:\n"
            "1) ```xml <dmn:definitions>…```  2) ```json ASK_PLAN```\n\n"
            "DMN\n- Root tag MUST be <dmn:definitions xmlns:dmn=\"https://www.omg.org/spec/DMN/20191111/MODEL/\"> (no other DMN namespaces).\n"
            "- Decisions: decide_danger_signs, decide_diarrhea, decide_fever_malaria, decide_respiratory, decide_nutrition, aggregate_final\n"
            "- Each module: <dmn:decisionTable hitPolicy=\"FIRST\">\n  Inputs: only relevant canonical variables\n  Outputs: columns triage:string, danger_sign:boolean, clinic_referral:boolean, reason:string, ref:string\n  Rows: highest severity first. Use FEEL literals and comparators only. Escape &lt; &gt;\n  Invariants: hospital → danger_sign=true, clinic → clinic_referral=true\n"
            "- aggregate_final precedence:\n  if any module.danger_sign=true → triage:hospital\n  else if any module.clinic_referral=true → triage:clinic\n  else → triage:home\n  Carry reason/ref from the highest priority firing module\n"
            "- Add <dmn:inputData> for every used canonical variable with correct typeRef.\n- Add <dmn:informationRequirement> from aggregate_final to every module decision.\n- No string contains(). No effect blobs. No non-DMN elements.\n"
            "ASK_PLAN\n[ {\"module\":\"diarrhea\",\"ask\":[\"diarrhea\"],\"followups_if\":{\"diarrhea==true\":[\"blood_in_stool\",\"diarrhea_duration_days\"]}},\n  {\"module\":\"fever_malaria\",\"ask\":[\"fever\"],\"followups_if\":{\"fever==true\":[\"temperature_c\",\"fever_duration_days\",\"malaria_area\"]}},\n  {\"module\":\"respiratory\",\"ask\":[\"cough\"],\"followups_if\":{\"cough==true\":[\"resp_rate\",\"cough_duration_days\",\"chest_indrawing\"]}},\n  {\"module\":\"nutrition\",\"ask\":[\"muac_mm\",\"edema_both_feet\"]},\n  {\"module\":\"danger_signs\",\"ask\":[\"convulsions\",\"unconscious\",\"unable_to_drink\",\"vomiting_everything\",\"chest_indrawing\",\"edema_both_feet\"]}]\nINPUT\nRULES_JSON: <PASTE MERGED IR FROM STEP 2>"
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
        blocks = _extract_fenced_blocks(text)
        dmn_xml: Optional[str] = None
        ask_plan: Optional[Any] = None
        # First pass: use fenced blocks
        for lang, body in blocks:
            b = body.strip()
            if dmn_xml is None and "<dmn:definitions" in b and "</dmn:definitions>" in b:
                dmn_xml = b
                continue
            if ask_plan is None:
                try:
                    parsed = json.loads(_strip_code_fences(b))
                    if isinstance(parsed, list) or isinstance(parsed, dict):
                        ask_plan = parsed
                except Exception:
                    pass
        # Fallbacks
        if dmn_xml is None:
            s = text
            start = s.find("<dmn:definitions")
            end = s.find("</dmn:definitions>")
            if start != -1 and end != -1 and end > start:
                end += len("</dmn:definitions>")
                dmn_xml = s[start:end].strip()
        if ask_plan is None:
            # Try to locate first plausible JSON array in whole text
            s = text
            lb = s.find('[')
            rb = s.rfind(']')
            if lb != -1 and rb != -1 and rb > lb:
                candidate = s[lb:rb+1]
                try:
                    ask_plan = json.loads(_strip_code_fences(candidate))
                except Exception:
                    pass
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


