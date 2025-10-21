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
    # orchestrator.xlsx is written where you ask (not always in out/)
]

# ---------- tiny logging helpers ----------
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def log(msg: str):
        line = f"[{_ts()}] {msg}"
        print(line, flush=True)
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

# ---------- optional fact sheet ----------
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
    ap = argparse.ArgumentParser(description="Extract → DMN → CSVs → XLSForm → wire pulldata()")
    ap.add_argument("--pdf", required=False, help="Path to guideline PDF (required for steps 1–5; optional for --run 6)")
    ap.add_argument("--out", default="out", help="Output directory for intermediate artifacts (default: out)")
    ap.add_argument("--log", default=None, help="Optional log file path (default: <out>/pipeline.log)")
    # Simple one-flag model override
    ap.add_argument("--model", default="gpt-5", help="Model id to use for all stages (default: gpt-5)")
    # Per-stage overrides
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
    ap.add_argument("--run", choices=["all","1","2","3","4","5","6"], default="all",
                    help="Which step to run: all (default) or a single step number.")
    ap.add_argument("--write-per-section", action="store_true",
                    help="Also write each section JSON to out/sections/section_###.json with logs.")

    # deployment-oriented outputs
    ap.add_argument("--media-dir", default="forms/app/orchestrator-media",
                    help="Where to write modular CSVs (default: forms/app/orchestrator-media)")
    ap.add_argument("--xlsx-out", default="forms/app/orchestrator.xlsx",
                    help="Where to write the XLSForm (default: forms/app/orchestrator.xlsx)")
    ap.add_argument("--xlsx-template", default=None,
                    help="Optional path to an XLSX template to fill")

    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else (out_dir / "pipeline.log")
    log = make_logger(log_path)

    # --- Env check (skip for run 6) ---
    if args.run != "6":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            log("❌ OPENAI_API_KEY not set. Export it and re-run.")
            sys.exit(1)

    # --- Path / PDF check ---
    pdf = Path(args.pdf) if args.pdf else None
    if args.run != "6" and not pdf:
        log("❌ --pdf is required for steps 1–5.")
        sys.exit(1)
    if pdf:
        if os.name == "posix" and (":\\" in args.pdf or ":\\\\\\" in args.pdf):
            log("⚠️  Detected Windows-style path on Linux/WSL. Use /mnt/c/... form.")
        if not pdf.exists():
            if args.run == "6":
                log(f"ⓘ PDF not found: {pdf} — continuing because --run 6 only needs prior artifacts.")
            else:
                log(f"❌ PDF not found: {pdf}")
                sys.exit(1)

    ge.STRICT_MERGE = not args.no_strict_merge

    # Models
    m_all = args.model or "gpt-5"
    m_section = args.model_section or m_all
    m_merge   = args.model_merge   or m_all
    m_dmn     = args.model_dmn     or m_all
    m_bpmn    = args.model_bpmn    or m_all
    m_audit   = args.model_audit   or m_all

    log(f"Models: section={m_section}, merge={m_merge}, dmn={m_dmn}, bpmn={m_bpmn}, audit={m_audit}")
    log(f"Strict merge: {ge.STRICT_MERGE}")
    log(f"PDF: {pdf}" if pdf else "PDF: (not required for this run)")
    log(f"Out: {out_dir}")
    log(f"Log: {log_path}")

    # --- Initialize extractor (no client needed for run 6) ---
    log("▶️  Initializing extractor …")
    t0 = time.time()
    gx = OpenAIGuardedExtractor(
        api_key=(os.getenv("OPENAI_API_KEY") if args.run != "6" else None),
        model_section=m_section,
        model_merge=m_merge,
        model_dmn=m_dmn,
        model_bpmn=m_bpmn,
        model_audit=m_audit,
        canonical_config_path=args.canonical_config,
    )
    log(f"init.ok in {time.time()-t0:.2f}s")

    # Paths for intermediates
    sec_path  = out_dir / "sections.json"
    facts_path= out_dir / "fact_sheet.json"
    mir_path  = out_dir / "merged_ir.json"
    dmn_path  = out_dir / "decisions.dmn"
    ask_path  = out_dir / "ask_plan.json"
    bpmn_path = out_dir / "workflow.bpmn"
    cov_path  = out_dir / "coverage.json"

    # XLS/CSV destinations (not necessarily in out/)
    media_dir = Path(args.media_dir)
    xlsx_path = Path(args.xlsx_out)

    # normalize template early
    template_abs = None
    if args.xlsx_template:
        template_abs = str(Path(args.xlsx_template).resolve())
        log(f"xlsx.template: {template_abs}")
    xlsx_abs = str(xlsx_path.resolve())
    media_abs = str(media_dir.resolve())
    log(f"xlsx.out: {xlsx_abs}")
    log(f"media.dir: {media_abs}")

    ran_any = False

    # ---------------- Step 1 ----------------
    if args.run in ("all", "1"):
        log("STEP 1/6: Extracting rules per section …")
        t1 = time.time()
        sections = gx.extract_rules_per_section(str(pdf))
        log(f"step1.elapsed: {time.time()-t1:.2f}s")

        if not sections:
            log("⚠️  No extractable text chunks found. The PDF might be scanned/image-only.")
            log("    Tips: run OCR (e.g., ocrmypdf) or ensure pdfminer/pytesseract are available.")

        if args.write_per_section:
            sec_dir = out_dir / "sections"
            sec_dir.mkdir(parents=True, exist_ok=True)
            for i, sec in enumerate(sections, start=1):
                p = sec_dir / f"section_{i:03d}.json"
                write_json(p, sec, log=log)

        write_json(sec_path, sections, log=log)
        log(f"step1.summary: sections={len(sections)} → {sec_path}")

        fact_sheet = build_fact_sheet_from_sections(sections)
        write_json(facts_path, fact_sheet, log=log)
        log(f"step1.fact_sheet: facts={len(fact_sheet.get('facts', []))} → {facts_path}")

        try:
            u = getattr(gx, "get_usage_summary", lambda: None)()
            if u:
                log(f"usage.cumulative: prompt={u['prompt']} completion={u['completion']} total={u['total']}")
        except Exception:
            pass

        ran_any = True
        if args.run == "1":
            log("🎉 Done (step 1 only).")
            return
    else:
        if args.run in ("2","3","4","5"):
            if not sec_path.exists():
                log(f"❌ Missing {sec_path}. Run step 1 first (--run 1 or --run all).")
                sys.exit(1)
            sections = _safe_read_json(sec_path, log=log)
            if facts_path.exists():
                fact_sheet = _safe_read_json(facts_path, log=log)
                log(f"loaded: {facts_path} (facts={len(fact_sheet.get('facts', []))})")
            else:
                log("ⓘ No fact_sheet.json found; continuing without it.")

    if ge.STRICT_MERGE and 'sections' in locals() and not sections and args.run in ("all","2","3"):
        log("❌ Strict merge is ON and there are zero sections to merge. Aborting before step 2.")
        sys.exit(1)

    # ---------------- Step 2 ----------------
    if args.run in ("all", "2"):
        log("STEP 2/6: Merging + canonicalizing …")
        t2 = time.time()
        try:
            merged_ir = gx.merge_sections(sections)
        except Exception as e:
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
        try:
            u = getattr(gx, "get_usage_summary", lambda: None)()
            if u:
                log(f"usage.cumulative: prompt={u['prompt']} completion={u['completion']} total={u['total']}")
        except Exception:
            pass
        ran_any = True
        if args.run == "2":
            log("🎉 Done (step 2 only).")
            return
    else:
        if args.run in ("3","4","5","6"):
            if not mir_path.exists():
                log(f"❌ Missing {mir_path}. Run step 2 first (--run 2 or --run all).")
                sys.exit(1)
            merged_ir = _safe_read_json(mir_path, log=log)
            log(f"loaded: {mir_path} (variables={len(merged_ir.get('variables', []))}, rules={len(merged_ir.get('rules', []))})")

    # ---------------- Step 3 ----------------
    if args.run in ("all", "3"):
        log("STEP 3/6: Generating DMN + ASK_PLAN …")
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
        try:
            u = getattr(gx, "get_usage_summary", lambda: None)()
            if u:
                log(f"usage.cumulative: prompt={u['prompt']} completion={u['completion']} total={u['total']}")
        except Exception:
            pass
        ran_any = True
        if args.run == "3":
            log("🎉 Done (step 3 only).")
            return
    else:
        if args.run in ("4","5","6"):
            if not dmn_path.exists() or not ask_path.exists():
                log(f"❌ Missing {dmn_path} and/or {ask_path}. Run step 3 first (--run 3 or --run all).")
                sys.exit(1)
            dmn_xml = dmn_path.read_text(encoding="utf-8")
            ask_raw = _safe_read_json(ask_path, log=log)
            ask_plan = ask_raw.get("ASK_PLAN") if isinstance(ask_raw, dict) and "ASK_PLAN" in ask_raw else ask_raw
            ap_len = len(ask_plan if isinstance(ask_plan, list) else [])
            log(f"loaded: {dmn_path}, {ask_path} (ask_plan_blocks={ap_len})")

    # ---------------- Step 4 ----------------
    if args.run in ("all", "4"):
        log("STEP 4/6: Generating BPMN from DMN + ASK_PLAN …")
        t4 = time.time()
        bpmn_xml = gx.generate_bpmn(dmn_xml, ask_plan)
        log(f"step4.elapsed: {time.time()-t4:.2f}s")
        bpmn_path.write_text(bpmn_xml, encoding="utf-8")
        try:
            bpmn_size = bpmn_path.stat().st_size
        except Exception:
            bpmn_size = 0
        log(f"created: {bpmn_path}  ({bpmn_size} bytes)")
        try:
            u = getattr(gx, "get_usage_summary", lambda: None)()
            if u:
                log(f"usage.cumulative: prompt={u['prompt']} completion={u['completion']} total={u['total']}")
        except Exception:
            pass
        ran_any = True
        if args.run == "4":
            log("🎉 Done (step 4 only).")
            return
    else:
        if args.run == "5":
            if not bpmn_path.exists():
                log("ⓘ No BPMN found. Continuing; coverage step does not require BPMN.")

    # ---------------- Step 5 ----------------
    if args.run in ("all", "5"):
        log("STEP 5/6: Auditing coverage …")
        t5 = time.time()
        coverage = gx.audit_coverage(merged_ir, dmn_xml)
        log(f"step5.elapsed: {time.time()-t5:.2f}s")
        write_json(cov_path, coverage, log=log)
        unmapped = len(coverage.get("unmapped_rule_ids") or [])
        mc = coverage.get("module_counts") or {}
        log(f"step5.summary: unmapped_rules={unmapped}, module_counts={mc} → {cov_path}")
        try:
            u = getattr(gx, "get_usage_summary", lambda: None)()
            if u:
                log(f"usage.cumulative: prompt={u['prompt']} completion={u['completion']} total={u['total']}")
        except Exception:
            pass
        ran_any = True
        if args.run == "5":
            log("🎉 Done (step 5 only).")
            return

    # ---------------- Step 6 ----------------
    if args.run in ("all", "6"):
        log("STEP 6/6: Exporting XLSX + media CSVs + wiring pulldata() …")

        # collect inputs if missing from scope
        missing = []
        if 'merged_ir' not in locals():
            if mir_path.exists():
                merged_ir = _safe_read_json(mir_path, log=log)
                log(f"loaded: {mir_path} (variables={len(merged_ir.get('variables', []))}, rules={len(merged_ir.get('rules', []))})")
            else:
                missing.append(str(mir_path))
        if 'dmn_xml' not in locals():
            if dmn_path.exists():
                dmn_xml = dmn_path.read_text(encoding="utf-8")
                log(f"loaded: {dmn_path}")
            else:
                missing.append(str(dmn_path))
        if 'ask_plan' not in locals():
            if ask_path.exists():
                ask_raw = _safe_read_json(ask_path, log=log)
                ask_plan = ask_raw.get("ASK_PLAN") if isinstance(ask_raw, dict) and "ASK_PLAN" in ask_raw else ask_raw
                log(f"loaded: {ask_path}")
            else:
                missing.append(str(ask_path))
        if missing:
            log("❌ Missing required inputs for XLSX export: " + ", ".join(missing))
            sys.exit(1)

        # deps
        try:
            import openpyxl  # noqa: F401
        except Exception:
            log("❌ openpyxl not installed. Try:  python -m pip install openpyxl")
            sys.exit(3)

        # 6a) DMN → per-module CSVs (for jr://file-csv/)
        media_dir.mkdir(parents=True, exist_ok=True)
        csv_map = gx.export_csvs_from_dmn(dmn_xml, str(media_dir))
        for mod, path in csv_map.items():
            try:
                size = Path(path).stat().st_size
            except Exception:
                size = 0
            log(f"media.csv[{mod}]: {path}  ({size} bytes)")

        # 6b) Build XLSForm (questions) at the requested path
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        outp = gx.export_xlsx_from_dmn(
            merged_ir=merged_ir,
            ask_plan=ask_plan if isinstance(ask_plan, list) else (ask_plan.get("ASK_PLAN", []) if isinstance(ask_plan, dict) else []),
            out_xlsx_path=str(xlsx_path),
            template_xlsx_path=template_abs
        )
        try:
            size = Path(outp).stat().st_size
        except Exception:
            size = 0
        log(f"xlsx.created: {outp}  ({size} bytes)")

        # 6c) Wire calculates that reproduce the DMN (module_key + pulldata())
        gx.wire_decisions_into_xlsx(str(xlsx_path), dmn_xml, merged_ir)
        log("xlsx.wired: added <module>_key and pulldata('dmn_<module>.csv', ...) calculations")

        ran_any = True
        if args.run == "6":
            log("🎉 Done (step 6 only).")
            return

    if ran_any:
        log("🎉 All done. Intermediate outputs in:")
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
        log(f"➡ XLSForm: {xlsx_path}")
        log(f"➡ Media CSVs: {media_dir}")

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
