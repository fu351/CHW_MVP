from pathlib import Path
import json
from guarded_extractor import OpenAIGuardedExtractor

OUT_DIR = Path("out")
DMN = (OUT_DIR/"decisions.dmn").read_text(encoding="utf-8")
ASK_RAW = json.loads((OUT_DIR/"ask_plan.json").read_text(encoding="utf-8"))
ASK = ASK_RAW.get("ASK_PLAN") if isinstance(ASK_RAW, dict) and "ASK_PLAN" in ASK_RAW else ASK_RAW
MERGED = json.loads((OUT_DIR/"merged_ir.json").read_text(encoding="utf-8"))

media_dir = Path("forms/app/orchestrator-media"); media_dir.mkdir(parents=True, exist_ok=True)
xlsx_path = Path("forms/app/orchestrator.xlsx"); xlsx_path.parent.mkdir(parents=True, exist_ok=True)

# dummy key ok: we won’t call the API for these steps
gx = OpenAIGuardedExtractor(api_key="DUMMY")

# CSVs per module
csv_map = gx.export_csvs_from_dmn(DMN, str(media_dir))
print("Media CSVs:", csv_map)

# XLSForm (questions)
gx.export_xlsx_from_dmn(MERGED, ASK, str(xlsx_path), template_xlsx_path=None)

# Wire pulldata() lookups
gx.wire_decisions_into_xlsx(str(xlsx_path), DMN, MERGED)

print("Done. XLSForm:", xlsx_path)
