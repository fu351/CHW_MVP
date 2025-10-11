#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from guarded_extractor import OpenAIGuardedExtractor

def main():
    ap = argparse.ArgumentParser(description="Run guarded extractor pipeline end-to-end.")
    ap.add_argument("--pdf", required=True, help="Path to guideline PDF")
    ap.add_argument("--out", default="out", help="Output directory (default: out)")
    ap.add_argument("--model-section", default="gpt-4o-mini")
    ap.add_argument("--model-merge",   default="gpt-4o-mini")
    ap.add_argument("--model-dmn",     default="gpt-4o")
    ap.add_argument("--model-bpmn",    default="gpt-4o")
    ap.add_argument("--model-audit",   default="gpt-4o-mini")
    args = ap.parse_args()

    # --- Env check ---
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("❌ OPENAI_API_KEY not set. Export it in this shell and re-run.", file=sys.stderr)
        sys.exit(1)

    # --- Path check (warn if Windows path while running in Linux/WSL) ---
    pdf_path = args.pdf
    if os.name == "posix" and (":\\" in pdf_path or ":\\" in pdf_path):
        print("⚠️  Detected Windows-style path on Linux. In WSL, use /mnt/c/... form.", file=sys.stderr)

    pdf = Path(pdf_path)
    if not pdf.exists():
        print(f"❌ PDF not found: {pdf}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Instantiate extractor with your chosen models ---
    print("▶️  Initializing extractor …")
    gx = OpenAIGuardedExtractor(
        model_section=args.model_section,
        model_merge=args.model_merge,
        model_dmn=args.model_dmn,
        model_bpmn=args.model_bpmn,
        model_audit=args.model_audit,
    )

    # STEP 1
    print("STEP 1/5: Extracting rules per section …")
    sections = gx.extract_rules_per_section(str(pdf))
    (out_dir / "sections.json").write_text(json.dumps(sections, indent=2), encoding="utf-8")
    print(f"✅ Step 1 done. Sections extracted: {len(sections)}  → {out_dir/'sections.json'}")

    # STEP 2
    print("STEP 2/5: Merging + canonicalizing …")
    merged_ir = gx.merge_sections(sections)
    (out_dir / "merged_ir.json").write_text(json.dumps(merged_ir, indent=2), encoding="utf-8")
    print(f"✅ Step 2 done. Variables: {len(merged_ir.get('variables', []))}, Rules: {len(merged_ir.get('rules', []))}  → {out_dir/'merged_ir.json'}")

    # STEP 3
    print("STEP 3/5: Generating DMN + ASK_PLAN …")
    dmn_xml, ask_plan = gx.generate_dmn_and_ask_plan(merged_ir)
    (out_dir / "decisions.dmn").write_text(dmn_xml, encoding="utf-8")
    (out_dir / "ask_plan.json").write_text(json.dumps(ask_plan, indent=2), encoding="utf-8")
    print(f"✅ Step 3 done. DMN + ASK_PLAN written → {out_dir/'decisions.dmn'}, {out_dir/'ask_plan.json'}")

    # STEP 4
    print("STEP 4/5: Generating BPMN from DMN + ASK_PLAN …")
    bpmn_xml = gx.generate_bpmn(dmn_xml, ask_plan)
    (out_dir / "workflow.bpmn").write_text(bpmn_xml, encoding="utf-8")
    print(f"✅ Step 4 done. BPMN written → {out_dir/'workflow.bpmn'}")

    # STEP 5
    print("STEP 5/5: Auditing coverage …")
    coverage = gx.audit_coverage(merged_ir, dmn_xml)
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(f"✅ Step 5 done. Coverage written → {out_dir/'coverage.json'}")

    print("🎉 All done. Outputs:")
    for p in ["sections.json", "merged_ir.json", "decisions.dmn", "ask_plan.json", "workflow.bpmn", "coverage.json"]:
        print("   •", out_dir / p)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        # Also point to extractor’s debug folder for deeper traces
        print(f"💥 Error: {e}", file=sys.stderr)
        print("   Check ./.chatchw_debug/* for raw model outputs and traces.", file=sys.stderr)
        sys.exit(1)
