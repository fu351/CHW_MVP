#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from argparse import BooleanOptionalAction  # Python 3.9+

import guarded_extractor as ge
from guarded_extractor import OpenAIGuardedExtractor

# ---------- Utilities ----------
def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON: {p} ({e})")

def _write_json(p: Path, obj):
    try:
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to write JSON: {p} ({e})")

def _require_env():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("❌ OPENAI_API_KEY not set. Export it and re-run.", file=sys.stderr)
        sys.exit(1)

def _check_pdf_path(pdf_str: str) -> Path:
    if os.name == "posix" and ":\\\\" in pdf_str:
        print("⚠️  Detected Windows-style path on Linux/WSL. Use /mnt/c/... form.", file=sys.stderr)
    pdf = Path(pdf_str)
    if not pdf.exists():
        print(f"❌ PDF not found: {pdf}", file=sys.stderr)
        sys.exit(1)
    return pdf

def _new_extractor(args) -> OpenAIGuardedExtractor:
    # Resolve models: umbrella → per-step
    m_all = args.model or os.getenv("CHATCHW_MODEL", "gpt-5")
    m_section = args.model_section or m_all
    m_merge   = args.model_merge   or m_all
    m_dmn     = args.model_dmn     or m_all
    m_bpmn    = args.model_bpmn    or m_all
    m_audit   = args.model_audit   or m_all

    # Strict-merge (default True)
    ge.STRICT_MERGE = args.strict_merge

    print("Models:",
          f"section={m_section}, merge={m_merge}, dmn={m_dmn}, bpmn={m_bpmn}, audit={m_audit}")
    print("Strict merge:", ge.STRICT_MERGE)

    return OpenAIGuardedExtractor(
        model_section=m_section,
        model_merge=m_merge,
        model_dmn=m_dmn,
        model_bpmn=m_bpmn,
        model_audit=m_audit,
        canonical_config_path=args.canonical_config,
    )

def _ensure_out(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

# ---------- Step runners ----------
def run_extract(args) -> None:
    _require_env()
    pdf = _check_pdf_path(args.pdf)
    out_dir = Path(args.out); _ensure_out(out_dir)

    gx = _new_extractor(args)
    print("STEP 1/5: Extracting rules per section …")
    sections = gx.extract_rules_per_section(str(pdf))
    _write_json(out_dir / "sections.json", sections)
    print(f"✅ Step 1 done. Sections extracted: {len(sections)}  → {out_dir/'sections.json'}")

def run_merge(args) -> None:
    _require_env()
    out_dir = Path(args.out); _ensure_out(out_dir)

    # Load sections (allow override)
    sections_path = Path(args.sections or (out_dir / "sections.json"))
    if not sections_path.exists():
        raise FileNotFoundError(f"Missing sections JSON: {sections_path}")
    sections = _read_json(sections_path)

    gx = _new_extractor(args)
    print("STEP 2/5: Merging + canonicalizing …")
    merged_ir = gx.merge_sections(sections)
    _write_json(out_dir / "merged_ir.json", merged_ir)
    print(f"✅ Step 2 done. Variables: {len(merged_ir.get('variables', []))}, Rules: {len(merged_ir.get('rules', []))}  → {out_dir/'merged_ir.json'}")

def run_dmn(args) -> None:
    _require_env()
    out_dir = Path(args.out); _ensure_out(out_dir)

    merged_path = Path(args.merged or (out_dir / "merged_ir.json"))
    if not merged_path.exists():
        raise FileNotFoundError(f"Missing merged IR: {merged_path}")
    merged_ir = _read_json(merged_path)

    gx = _new_extractor(args)
    print("STEP 3/5: Generating DMN + ASK_PLAN …")
    dmn_xml, ask_plan = gx.generate_dmn_and_ask_plan(merged_ir)
    (out_dir / "decisions.dmn").write_text(dmn_xml, encoding="utf-8")
    _write_json(out_dir / "ask_plan.json", ask_plan)
    print(f"✅ Step 3 done. DMN + ASK_PLAN written → {out_dir/'decisions.dmn'}, {out_dir/'ask_plan.json'}")

def run_bpmn(args) -> None:
    _require_env()
    out_dir = Path(args.out); _ensure_out(out_dir)

    dmn_path = Path(args.dmn or (out_dir / "decisions.dmn"))
    ask_path = Path(args.ask or (out_dir / "ask_plan.json"))
    if not dmn_path.exists():
        raise FileNotFoundError(f"Missing DMN XML: {dmn_path}")
    if not ask_path.exists():
        raise FileNotFoundError(f"Missing ASK_PLAN JSON: {ask_path}")

    dmn_xml = dmn_path.read_text(encoding="utf-8")
    ask_plan = _read_json(ask_path)

    gx = _new_extractor(args)
    print("STEP 4/5: Generating BPMN from DMN + ASK_PLAN …")
    bpmn_xml = gx.generate_bpmn(dmn_xml, ask_plan)
    (out_dir / "workflow.bpmn").write_text(bpmn_xml, encoding="utf-8")
    print(f"✅ Step 4 done. BPMN written → {out_dir/'workflow.bpmn'}")

def run_audit(args) -> None:
    _require_env()
    out_dir = Path(args.out); _ensure_out(out_dir)

    merged_path = Path(args.merged or (out_dir / "merged_ir.json"))
    dmn_path = Path(args.dmn or (out_dir / "decisions.dmn"))
    if not merged_path.exists():
        raise FileNotFoundError(f"Missing merged IR: {merged_path}")
    if not dmn_path.exists():
        raise FileNotFoundError(f"Missing DMN XML: {dmn_path}")

    merged_ir = _read_json(merged_path)
    dmn_xml = dmn_path.read_text(encoding="utf-8")

    gx = _new_extractor(args)
    print("STEP 5/5: Auditing coverage …")
    coverage = gx.audit_coverage(merged_ir, dmn_xml)
    _write_json(out_dir / "coverage.json", coverage)
    print(f"✅ Step 5 done. Coverage written → {out_dir/'coverage.json'}")

def run_all(args) -> None:
    # Just chain the steps with the same options
    if not args.pdf:
        print("❌ --pdf is required for 'all'", file=sys.stderr)
        sys.exit(2)
    run_extract(args)
    run_merge(args)
    run_dmn(args)
    run_bpmn(args)
    run_audit(args)

# ---------- CLI ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run guarded extractor pipeline (step-by-step or all).")
    sub = p.add_subparsers(dest="cmd", required=False)

    # Shared options
    def add_shared(a: argparse.ArgumentParser):
        a.add_argument("--out", default="out", help="Output directory (default: out)")
        # Umbrella + per-step model controls
        a.add_argument("--model", default=None, help="Use one model for ALL steps (defaults to CHATCHW_MODEL or gpt-5)")
        a.add_argument("--model-section", default=None)
        a.add_argument("--model-merge",   default=None)
        a.add_argument("--model-dmn",     default=None)
        a.add_argument("--model-bpmn",    default=None)
        a.add_argument("--model-audit",   default=None)
        a.add_argument("--canonical-config", default="chatchw/config/canonical_config.json")
        a.add_argument("--strict-merge", default=True, action=BooleanOptionalAction,
                       help="Force model-generated merged_ir and disable local fallback (default ON; use --no-strict-merge to disable)")

    # all
    pa = sub.add_parser("all", help="Run the entire pipeline")
    add_shared(pa)
    pa.add_argument("--pdf", required=True, help="Path to guideline PDF")
    pa.set_defaults(func=run_all)

    # extract
    pe = sub.add_parser("extract", help="Step 1: Extract section rules from PDF")
    add_shared(pe)
    pe.add_argument("--pdf", required=True, help="Path to guideline PDF")
    pe.set_defaults(func=run_extract)

    # merge
    pm = sub.add_parser("merge", help="Step 2: Merge + canonicalize")
    add_shared(pm)
    pm.add_argument("--sections", default=None, help="Path to sections.json (defaults to OUT/sections.json)")
    pm.set_defaults(func=run_merge)

    # dmn
    pd = sub.add_parser("dmn", help="Step 3: Generate DMN + ASK_PLAN")
    add_shared(pd)
    pd.add_argument("--merged", default=None, help="Path to merged_ir.json (defaults to OUT/merged_ir.json)")
    pd.set_defaults(func=run_dmn)

    # bpmn
    pb = sub.add_parser("bpmn", help="Step 4: Generate BPMN from DMN + ASK_PLAN")
    add_shared(pb)
    pb.add_argument("--dmn", default=None, help="Path to decisions.dmn (defaults to OUT/decisions.dmn)")
    pb.add_argument("--ask", default=None, help="Path to ask_plan.json (defaults to OUT/ask_plan.json)")
    pb.set_defaults(func=run_bpmn)

    # audit
    pu = sub.add_parser("audit", help="Step 5: Audit coverage")
    add_shared(pu)
    pu.add_argument("--merged", default=None, help="Path to merged_ir.json (defaults to OUT/merged_ir.json)")
    pu.add_argument("--dmn", default=None, help="Path to decisions.dmn (defaults to OUT/decisions.dmn)")
    pu.set_defaults(func=run_audit)

    # default: if no subcommand, behave like "all"
    p.add_argument("--pdf", help="Path to guideline PDF (used when no subcommand is given)")
    add_shared(p)
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    # If no subcommand, treat as 'all'
    if args.cmd is None:
        if not args.pdf:
            parser.error("When no subcommand is provided, --pdf is required (behaves like 'all').")
        args.cmd = "all"
        args.func = run_all

    try:
        args.func(args)  # type: ignore[attr-defined]
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"💥 Error: {e}", file=sys.stderr)
        print("   Check ./.chatchw_debug/* for raw model outputs and traces.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
