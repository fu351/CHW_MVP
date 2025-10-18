#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import guarded_extractor as ge
from guarded_extractor import OpenAIGuardedExtractor

ARTIFACTS = [
    "sections.json",
    "fact_sheet.json",
    "merged_ir.json",
    "decisions.dmn",
    "ask_plan.json",
    "workflow.bpmn",
    "coverage.json",
]

# ---------- tiny logging helpers ----------
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def log(msg: str):
        line = f"[{_ts()}] {msg}"
        print(line)
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
    return log

def _safe_read_json(p: Path, log=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        if log:
            log(f"❌ Failed to read JSON {p}: {e}")
        raise

def write_json(p: Path, obj, log=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    if log:
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        log(f"created: {p}  ({size} bytes)")

# ---------- optional fact sheet (built from section outputs) ----------
def _flatten_rule_conditions(rule: Dict[str, Any]) -> List[str]:
    out = []
    def one(c):
        if isinstance(c, dict):
            if "sym" in c:
                return f"{c['sym']}=={str(c.get('eq')).lower()}"
            if "obs" in c:
                return f"{c['obs']} {c['op']} {json.dumps(c['value'])}"
        return None
    for c in rule.get("when", []) or []:
        if isinstance(c, dict) and ("all_of" in c or "any_of" in c):
            k = "all_of" if "all_of" in c else "any_of"
            parts = [one(s) for s in (c.get(k) or [])]
            parts = [p for p in parts if p]
            if parts:
                out.append(f"{k}:" + " && ".join(parts))
        else:
            s = one(c)
            if s:
                out.append(s)
    return out

def build_fact_sheet_from_sections(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    facts: List[Dict[str, Any]] = []
    for idx, sec in enumerate(sections, start=1):
        sid = f"section_{idx:03d}"
        # variable facts
        for v in (sec.get("variables") or []):
            if not isinstance(v, dict): continue
            facts.append({
                "topic": "variable",
                "name": v.get("name"),
                "type": v.get("type"),
                "unit": v.get("unit"),
                "allowed": v.get("allowed"),
                "synonyms": v.get("synonyms"),
                "refs": v.get("refs"),
                "section": sid,
            })
        # rule facts (criteria + outputs + ref)
        for r in (sec.get("rules") or []):
            if not isinstance(r, dict): continue
            conds = _flatten_rule_conditions(r)
            th = r.get("then") or {}
            facts.append({
                "topic": "criterion",
                "rule_id": r.get("rule_id"),
                "conditions": conds,
                "triage": th.get("triage"),
                "flags": th.get("flags"),
                "reasons": th.get("reasons"),
                "actions": th.get("actions"),
                "guideline_ref": th.get("guideline_ref"),
                "section": sid,
            })
    return {"facts": facts, "qa": {"notes": []}}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Run guarded extractor pipeline end-to-end (or a single step).")
    ap.add_argument("--pdf", required=True, help="Path to guideline PDF")
    ap.add_argument("--out", default="out", help="Output directory (default: out)")
    ap.add_argument("--log", default=None, help="Optional log file path (default: <out>/pipeline.log)")
    # Simple one-flag model override
    ap.add_argument("--model", default="gpt-5", help="Model id to use for all stages (default: gpt-5)")
    # Per-stage overrides still allowed
    ap.add_argument("--model-section", default=None)
    ap.add_argument("--model-merge",   default=None)
    ap.add_argument("--model-dmn",     default=None)
    ap.add_argument("--model-bpmn",    default=None)
    ap.add_argument("--model-audit",   default=None)
    ap.add_argument("--canonical-config", default="chatchw/config/canonical_config.json")
    ap.add_argument("--no-strict-merge", action="store_true",
                    help="Disable strict model-merge (use deterministic local fallback). Default is strict ON.")
    ap.add_argument("--no-merge-fallback", action="store_true",
                    help="Do not auto-fallback to local merge if strict merge fails.")
    ap.add_argument("--run", choices=["all","1","2","3","4","5"], default="all",
                    help="Which step to run: all (default) or a single step number.")
    ap.add_argument("--write-per-section", action="store_true",
                    help="Also write each section JSON to out/sections/section_###.json with logs.")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else (out_dir / "pipeline.log")
    log = make_logger(log_path)

    # --- Env check ---
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        log("❌ OPENAI_API_KEY not set. Export it and re-run.")
        sys.exit(1)

    # --- Path check ---
    if os.name == "posix" and (":\\" in args.pdf or ":\\\\\\" in args.pdf):
        log("⚠️  Detected Windows-style path on Linux/WSL. Use /mnt/c/... form.")

    pdf = Path(args.pdf)
    if not pdf.exists():
        log(f"❌ PDF not found: {pdf}")
        sys.exit(1)

    # --- Strict merge default ON unless disabled ---
    ge.STRICT_MERGE = not args.no_strict_merge

    # --- Model wiring (global default + per-stage override) ---
    m_all = args.model or "gpt-5"
    m_section = args.model_section or m_all
    m_merge   = args.model_merge   or m_all
    m_dmn     = args.model_dmn     or m_all
    m_bpmn    = args.model_bpmn    or m_all
    m_audit   = args.model_audit   or m_all

    log(f"Models: section={m_section}, merge={m_merge}, dmn={m_dmn}, bpmn={m_bpmn}, audit={m_audit}")
    log(f"Strict merge: {ge.STRICT_MERGE}")
    log(f"PDF: {pdf}")
    log(f"Out: {out_dir}")
    log(f"Log: {log_path}")

    # --- Initialize extractor ---
    log("▶️  Initializing extractor …")
    t0 = time.time()
    gx = OpenAIGuardedExtractor(
        model_section=m_section,
        model_merge=m_merge,
        model_dmn=m_dmn,
        model_bpmn=m_bpmn,
        model_audit=m_audit,
        canonical_config_path=args.canonical_config,
    )
    log(f"init.ok in {time.time()-t0:.2f}s")

    # Helpers to resolve inputs for later steps
    sec_path = out_dir / "sections.json"
    facts_path = out_dir / "fact_sheet.json"
    mir_path = out_dir / "merged_ir.json"
    dmn_path = out_dir / "decisions.dmn"
    ask_path = out_dir / "ask_plan.json"
    bpmn_path= out_dir / "workflow.bpmn"
    cov_path = out_dir / "coverage.json"

    ran_any = False

    # ---------------- Step 1 ----------------
    if args.run in ("all", "1"):
        log("STEP 1/5: Extracting rules per section …")
        t1 = time.time()
        sections = gx.extract_rules_per_section(str(pdf))
        log(f"step1.elapsed: {time.time()-t1:.2f}s")

        if not sections:
            log("⚠️  No extractable text chunks found. The PDF might be scanned/image-only.")
            log("    Tips: run OCR (e.g., ocrmypdf) or ensure pdfminer/pytesseract are available.")

        # (optional) write each section individually
        if args.write_per_section:
            sec_dir = out_dir / "sections"
            sec_dir.mkdir(parents=True, exist_ok=True)
            for i, sec in enumerate(sections, start=1):
                p = sec_dir / f"section_{i:03d}.json"
                write_json(p, sec, log=log)

        write_json(sec_path, sections, log=log)
        log(f"step1.summary: sections={len(sections)} → {sec_path}")

        # Build a lightweight human-audit fact sheet straight from sections
        fact_sheet = build_fact_sheet_from_sections(sections)
        write_json(facts_path, fact_sheet, log=log)
        log(f"step1.fact_sheet: facts={len(fact_sheet.get('facts', []))} → {facts_path}")

        ran_any = True
        if args.run == "1":
            log("🎉 Done (step 1 only).")
            return
    else:
        if not sec_path.exists():
            log(f"❌ Missing {sec_path}. Run step 1 first (--run 1 or --run all).")
            sys.exit(1)
        sections = _safe_read_json(sec_path, log=log)
        if facts_path.exists():
            fact_sheet = _safe_read_json(facts_path, log=log)
            log(f"loaded: {facts_path} (facts={len(fact_sheet.get('facts', []))})")
        else:
            log("ⓘ No fact_sheet.json found; continuing without it.")

    # Early bail if strict merge and nothing to merge
    if ge.STRICT_MERGE and not sections:
        log("❌ Strict merge is ON and there are zero sections to merge. Aborting before step 2.")
        sys.exit(1)

    # ---------------- Step 2 ----------------
    if args.run in ("all", "2"):
        log("STEP 2/5: Merging + canonicalizing …")
        t2 = time.time()
        try:
            merged_ir = gx.merge_sections(sections)
        except Exception as e:
            # Auto-fallback if strict merge failed and fallback allowed
            if ge.STRICT_MERGE and not args.no_merge_fallback:
                log(f"step2.warn: strict merge failed ({e.__class__.__name__}: {e}). Retrying with local fallback …")
                ge.STRICT_MERGE = False
                merged_ir = gx.merge_sections(sections)
                log("step2.info: local merge succeeded after fallback.")
            else:
                raise
        log(f"step2.elapsed: {time.time()-t2:.2f}s")
        write_json(mir_path, merged_ir, log=log)
        vcnt = len(merged_ir.get("variables", []))
        rcnt = len(merged_ir.get("rules", []))
        log(f"step2.summary: variables={vcnt}, rules={rcnt} → {mir_path}")
        ran_any = True
        if args.run == "2":
            log("🎉 Done (step 2 only).")
            return
    else:
        if not mir_path.exists():
            log(f"❌ Missing {mir_path}. Run step 2 first (--run 2 or --run all).")
            sys.exit(1)
        merged_ir = _safe_read_json(mir_path, log=log)
        log(f"loaded: {mir_path} (variables={len(merged_ir.get('variables', []))}, rules={len(merged_ir.get('rules', []))})")

    # ---------------- Step 3 ----------------
    if args.run in ("all", "3"):
        log("STEP 3/5: Generating DMN + ASK_PLAN …")
        t3 = time.time()
        dmn_xml, ask_plan = gx.generate_dmn_and_ask_plan(merged_ir)
        log(f"step3.elapsed: {time.time()-t3:.2f}s")
        dmn_path.write_text(dmn_xml, encoding="utf-8")
        try:
            dmn_size = dmn_path.stat().st_size
        except Exception:
            dmn_size = 0
        log(f"created: {dmn_path}  ({dmn_size} bytes)")

        write_json(ask_path, ask_plan, log=log)
        ap_len = len(ask_plan if isinstance(ask_plan, list) else (ask_plan.get("ASK_PLAN", []) if isinstance(ask_plan, dict) else []))
        log(f"step3.summary: ask_plan_blocks={ap_len} → {ask_path}")
        ran_any = True
        if args.run == "3":
            log("🎉 Done (step 3 only).")
            return
    else:
        if not dmn_path.exists() or not ask_path.exists():
            log(f"❌ Missing {dmn_path} and/or {ask_path}. Run step 3 first (--run 3 or --run all).")
            sys.exit(1)
        dmn_xml = dmn_path.read_text(encoding="utf-8")
        ask_raw = _safe_read_json(ask_path, log=log)
        # Normalize ASK_PLAN if wrapped
        ask_plan = ask_raw.get("ASK_PLAN") if isinstance(ask_raw, dict) and "ASK_PLAN" in ask_raw else ask_raw
        ap_len = len(ask_plan if isinstance(ask_plan, list) else [])
        log(f"loaded: {dmn_path}, {ask_path} (ask_plan_blocks={ap_len})")

    # ---------------- Step 4 ----------------
    if args.run in ("all", "4"):
        log("STEP 4/5: Generating BPMN from DMN + ASK_PLAN …")
        t4 = time.time()
        bpmn_xml = gx.generate_bpmn(dmn_xml, ask_plan)
        log(f"step4.elapsed: {time.time()-t4:.2f}s")
        bpmn_path.write_text(bpmn_xml, encoding="utf-8")
        try:
            bpmn_size = bpmn_path.stat().st_size
        except Exception:
            bpmn_size = 0
        log(f"created: {bpmn_path}  ({bpmn_size} bytes)")
        ran_any = True
        if args.run == "4":
            log("🎉 Done (step 4 only).")
            return
    else:
        if not bpmn_path.exists():
            log(f"❌ Missing {bpmn_path}. Run step 4 first (--run 4 or --run all).")
            sys.exit(1)
        bpmn_xml = bpmn_path.read_text(encoding="utf-8")
        log(f"loaded: {bpmn_path}")

    # ---------------- Step 5 ----------------
    if args.run in ("all", "5"):
        log("STEP 5/5: Auditing coverage …")
        t5 = time.time()
        coverage = gx.audit_coverage(merged_ir, dmn_xml)
        log(f"step5.elapsed: {time.time()-t5:.2f}s")
        write_json(cov_path, coverage, log=log)
        unmapped = len(coverage.get("unmapped_rule_ids") or [])
        mc = coverage.get("module_counts") or {}
        log(f"step5.summary: unmapped_rules={unmapped}, module_counts={mc} → {cov_path}")
        ran_any = True
        if args.run == "5":
            log("🎉 Done (step 5 only).")
            return

    if ran_any:
        log("🎉 All done. Outputs:")
        for p in ARTIFACTS:
            dst = out_dir / p
            if dst.exists():
                try:
                    size = dst.stat().st_size
                except Exception:
                    size = 0
                log(f"   • {dst}  ({size} bytes)")
            else:
                log(f"   • {dst}  (missing)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"💥 Error: {e}", file=sys.stderr)
        print("   Check ./.chatchw_debug/* for raw model outputs and traces.", file=sys.stderr)
        sys.exit(1)
