from pathlib import Path
import json
from guarded_extractor import OpenAIGuardedExtractor

# Point this at your PDF
pdf_path = r"C:\Users\liang\Downloads\CHW_MVP\WHO CHW guide 2012.pdf"

gx = OpenAIGuardedExtractor()  # uses OPENAI_API_KEY env var

# Pipeline
sections = gx.extract_rules_per_section(pdf_path)
merged_ir = gx.merge_sections(sections)
dmn_xml, ask_plan = gx.generate_dmn_and_ask_plan(merged_ir)
bpmn_xml = gx.generate_bpmn(dmn_xml, ask_plan)
coverage = gx.audit_coverage(merged_ir, dmn_xml)

# Save outputs
out = Path("outputs"); out.mkdir(exist_ok=True)
(out / "decisions.dmn").write_text(dmn_xml, encoding="utf-8")
(out / "workflow.bpmn").write_text(bpmn_xml, encoding="utf-8")
(out / "ask_plan.json").write_text(json.dumps(ask_plan, indent=2), encoding="utf-8")
(out / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
print("✅ Done. Check the `outputs/` folder.")
