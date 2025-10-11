from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .schema import Rule


def load_rules_dir(rules_dir: str | Path) -> Dict[str, List[Rule]]:
    rules_path = Path(rules_dir)
    if not rules_path.exists():
        raise FileNotFoundError(f"rules dir not found: {rules_dir}")
    result: Dict[str, List[Rule]] = {}
    for p in sorted(rules_path.glob("*.json")):
        if p.name.endswith(".meta.json") or p.name == "v1.meta.json":
            continue
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("rules", [])
        module_name = p.stem
        result[module_name] = [Rule(**r) for r in data]
    return result
