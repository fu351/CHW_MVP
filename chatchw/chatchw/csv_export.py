from __future__ import annotations

import csv
from typing import Any, Dict, Iterable


CSV_HEADERS = [
    "age_months",
    "sex",
    "temp",
    "resp_rate",
    "muac_mm",
    "feels_very_hot",
    "blood_in_stool",
    "diarrhea_days",
    "convulsion",
    "edema_both_feet",
    "malaria_present",
    "cholera_present",
    "triage",
    "reasons",
    "trace_rules",
]


def _obs_map(enc: Dict[str, Any]) -> Dict[str, float]:
    res: Dict[str, float] = {}
    for o in enc.get("observations", []):
        try:
            res[str(o.get("id"))] = float(o.get("value"))
        except Exception:
            pass
    return res


def _row(rec: Dict[str, Any]) -> Dict[str, Any]:
    enc: Dict[str, Any] = rec["input"]
    dec: Dict[str, Any] = rec["decision"]
    om = _obs_map(enc)
    trace_rules = []
    for t in dec.get("trace", []):
        try:
            trace_rules.append(str(t.get("rule_id")))
        except Exception:
            pass
    return {
        "age_months": enc.get("age_months"),
        "sex": enc.get("sex"),
        "temp": om.get("temp"),
        "resp_rate": om.get("resp_rate"),
        "muac_mm": om.get("muac_mm"),
        "feels_very_hot": enc.get("symptoms", {}).get("feels_very_hot"),
        "blood_in_stool": enc.get("symptoms", {}).get("blood_in_stool"),
        "diarrhea_days": enc.get("symptoms", {}).get("diarrhea_days"),
        "convulsion": enc.get("symptoms", {}).get("convulsion"),
        "edema_both_feet": enc.get("symptoms", {}).get("edema_both_feet"),
        "malaria_present": enc.get("context", {}).get("malaria_present"),
        "cholera_present": enc.get("context", {}).get("cholera_present"),
        "triage": dec.get("triage"),
        "reasons": "|".join(dec.get("reasons", [])),
        "trace_rules": "|".join(trace_rules),
    }


def to_csv(records: Iterable[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        w.writeheader()
        for rec in records:
            w.writerow(_row(rec))
