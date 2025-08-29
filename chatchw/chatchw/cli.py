from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Enforce offline mode early
from .disable_net import init as _disable_net_init

_disable_net_init()

import typer  # noqa: E402
from pm4py.objects.bpmn.importer import importer as bpmn_importer  # noqa: E402
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer  # noqa: E402

from .bpmn_builder import build_bpmn  # noqa: E402
from .dmn_builder import generate_dmn  # noqa: E402
from .dmn_viz import render_drd  # noqa: E402
from .engine import decide  # noqa: E402
from .sandbox_fs import (  # noqa: E402
    append_jsonl,
    ensure_dirs,
    read_json,
    resolve_in_sandbox,
)
from .schema import (  # noqa: E402
    Action,
    Condition,
    ContextFlags,
    Decision,
    EncounterInput,
    Observation,
    Rule,
    RulepackMeta,
    Symptoms,
)


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_rules_from_dir(root: Path, rel_rules_dir: str) -> Dict[str, List[Rule]]:
    rules_dir = resolve_in_sandbox(root, rel_rules_dir)
    rulepacks: Dict[str, List[Rule]] = {}
    for path in sorted(Path(rules_dir).glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        if path.name == "v1.meta.json":
            continue
        with path.open("r", encoding="utf-8") as f:
            arr = json.load(f)
            if isinstance(arr, dict):
                arr = arr.get("rules", [])
            module_name = path.stem
            rs = [Rule(**r) for r in arr]
            rulepacks[module_name] = rs
    return rulepacks


def _copy_bundled_rules_into(root: Path) -> None:
    here = Path(__file__).resolve().parent
    src_dir = here / "rules"
    dst_dir = resolve_in_sandbox(root, "models/rules")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.glob("*.json"):
        dst = dst_dir / p.name
        with p.open("r", encoding="utf-8") as sf, dst.open("w", encoding="utf-8") as df:
            df.write(sf.read())


@app.command("init-sandbox")
def init_sandbox(
    root: Path = typer.Option(Path("./sandbox"), "--root", help="Sandbox root directory")
):
    d = ensure_dirs(root)
    here = Path(__file__).resolve().parent
    examples_dir = here / "examples"
    for p in examples_dir.glob("*"):
        target = resolve_in_sandbox(root, f"inputs/{p.name}")
        target.write_bytes(p.read_bytes())
    _copy_bundled_rules_into(root)
    typer.echo(f"Initialized sandbox at {Path(root).resolve()}")


@app.command("decide")
def cli_decide(
    input: str = typer.Option(..., "--input", help="Relative path to input JSON under sandbox root"),
    rules: str = typer.Option(..., "--rules", help="Relative path to rules dir under sandbox root"),
    out: Optional[str] = typer.Option(None, "--out", help="Relative path for decision JSON output (optional)"),
    root: Path = typer.Option(Path("./sandbox"), "--root", help="Sandbox root directory"),
):
    ensure_dirs(root)
    enc_obj = read_json(root, input)
    enc = EncounterInput(**enc_obj)
    rulepacks = _load_rules_from_dir(root, rules)
    decision = decide(enc, rulepacks)
    rec = {"input": enc.model_dump(), "decision": decision.model_dump()}
    if out:
        out_path = resolve_in_sandbox(root, out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(decision.model_dump(), indent=2), encoding="utf-8")
    else:
        out_file = f"logs/decision_{os.getpid()}.json"
        out_path = resolve_in_sandbox(root, out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(decision.model_dump(), indent=2), encoding="utf-8")
    append_jsonl(root, "logs/encounters.jsonl", rec)
    typer.echo(str(out_path))


@app.command("simulate")
def cli_simulate(
    n: int = typer.Option(..., "--n", min=1, help="Number of synthetic encounters"),
    module: Optional[str] = typer.Option(None, "--module", help="Limit to one module name (e.g., fever)"),
    context: str = typer.Option(..., "--context", help="Relative path to context JSON under sandbox root"),
    rules: str = typer.Option(..., "--rules", help="Relative path to rules dir under sandbox root"),
    root: Path = typer.Option(Path("./sandbox"), "--root", help="Sandbox root directory"),
):
    ensure_dirs(root)
    ctx_obj = read_json(root, context)
    base_ctx = ContextFlags(**ctx_obj)
    rulepacks = _load_rules_from_dir(root, rules)

    if module:
        rulepacks = {module: rulepacks.get(module, [])}

    rng = random.Random(42)
    for i in range(n):
        enc = EncounterInput(
            age_months=rng.randint(0, 180),
            sex=rng.choice(["m", "f"]),
            observations=[
                Observation(id="temp", value=round(rng.uniform(36.0, 40.5), 1)),
                Observation(id="resp_rate", value=float(rng.randint(15, 50))),
                Observation(id="muac_mm", value=float(rng.randint(100, 150))),
            ],
            symptoms=Symptoms(
                feels_very_hot=rng.choice([True, False]),
                blood_in_stool=rng.choice([True, False]),
                diarrhea_days=rng.randint(0, 20),
                convulsion=rng.choice([True, False, False]),
                edema_both_feet=rng.choice([True, False]),
            ),
            context=base_ctx,
        )
        decision = decide(enc, rulepacks)
        rec = {"input": enc.model_dump(), "decision": decision.model_dump()}
        append_jsonl(root, "logs/sim.jsonl", rec)
    typer.echo("ok")


def _iter_conditions(conds: Iterable[Condition]) -> Iterable[Condition]:
    for c in conds:
        yield c
        from .schema import AnyOfCondition, AllOfCondition
        if isinstance(c, AnyOfCondition):
            for x in _iter_conditions(c.any_of):
                yield x
        elif isinstance(c, AllOfCondition):
            for x in _iter_conditions(c.all_of):
                yield x


@app.command("validate")
def cli_validate(
    rules: str = typer.Option(..., "--rules", help="Relative path to rules dir under sandbox root"),
    root: Path = typer.Option(Path("./sandbox"), "--root", help="Sandbox root directory"),
):
    rs_map = _load_rules_from_dir(root, rules)
    has_fever_boundary = False
    has_muac_boundary = False

    from .schema import ObservationCondition

    for _mod, ruleset in rs_map.items():
        for r in ruleset:
            for c in _iter_conditions(r.when):
                if isinstance(c, ObservationCondition):
                    if c.obs == "temp" and c.op in ("ge", "gt") and float(c.value) >= 38.5:
                        has_fever_boundary = True
                    if c.obs == "muac_mm" and c.op in ("lt", "le") and float(c.value) <= 115.0:
                        has_muac_boundary = True

    if not has_fever_boundary or not has_muac_boundary:
        typer.echo("Missing required boundary rules", err=True)
        raise typer.Exit(code=1)
    typer.echo("ok")


@app.command("export-csv")
def cli_export_csv(
    out: str = typer.Option(..., "--out", help="Relative path for CSV output under sandbox root"),
    root: Path = typer.Option(Path("./sandbox"), "--root", help="Sandbox root directory"),
):
    from .csv_export import to_csv

    enc_path = resolve_in_sandbox(root, "logs/encounters.jsonl")
    if not enc_path.exists():
        typer.echo("no encounters", err=True)
        raise typer.Exit(code=1)
    records = []
    with enc_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    out_path = resolve_in_sandbox(root, out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_csv(records, str(out_path))
    typer.echo(str(out_path))


@app.command("generate-bpmn")
def cli_generate_bpmn(
    rules: str = typer.Option(..., "--rules", help="Relative path to rules dir under sandbox root"),
    out: str = typer.Option(..., "--out", help="Relative path for BPMN XML output under sandbox root"),
    root: Path = typer.Option(Path("./sandbox"), "--root", help="Sandbox root directory"),
):
    ensure_dirs(root)
    rulepacks = _load_rules_from_dir(root, rules)
    xml = build_bpmn(rulepacks)
    out_path = resolve_in_sandbox(root, out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml, encoding="utf-8")
    typer.echo(str(out_path))


@app.command("render-bpmn")
def cli_render_bpmn(
    bpmn: str = typer.Option(..., "--bpmn", help="Relative path to BPMN XML under sandbox root"),
    out: str = typer.Option(..., "--out", help="Relative path for rendered image (.svg|.png) under sandbox root"),
    root: Path = typer.Option(Path("./sandbox"), "--root", help="Sandbox root directory"),
):
    in_path = resolve_in_sandbox(root, bpmn)
    out_path = resolve_in_sandbox(root, out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        bpmn_graph = bpmn_importer.apply(str(in_path))
        gviz = bpmn_visualizer.apply(bpmn_graph)
        bpmn_visualizer.save(gviz, str(out_path))
    except Exception:
        # Fallback: write a minimal SVG placeholder if Graphviz is unavailable
        ext = out_path.suffix.lower()
        if ext == ".svg":
            svg = (
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"600\" height=\"120\">"
                "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>"
                f"<text x=\"12\" y=\"24\" font-family=\"Arial\" font-size=\"14\">BPMN placeholder for {in_path.name}</text>"
                "</svg>"
            )
            out_path.write_text(svg, encoding="utf-8")
        else:
            # For non-SVG, write a simple text marker
            out_path.write_bytes(b"BPMN placeholder; Graphviz not available")
    typer.echo(str(out_path))


@app.command("generate-dmn")
def cli_generate_dmn(
    rules: str = typer.Option(..., "--rules", help="Relative path to rules dir under sandbox root"),
    out: str = typer.Option(..., "--out", help="Relative path for DMN XML output under sandbox root"),
    root: Path = typer.Option(Path("./sandbox"), "--root", help="Sandbox root directory"),
):
    ensure_dirs(root)
    rulepacks = _load_rules_from_dir(root, rules)
    xml = generate_dmn(rulepacks)
    out_path = resolve_in_sandbox(root, out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml, encoding="utf-8")
    typer.echo(str(out_path))


@app.command("render-dmn")
def cli_render_dmn(
    dmn: str = typer.Option(..., "--dmn", help="Relative path to DMN XML under sandbox root"),
    out: str = typer.Option(..., "--out", help="Relative path for rendered DRD image (.svg|.png) under sandbox root"),
    root: Path = typer.Option(Path("./sandbox"), "--root", help="Sandbox root directory"),
):
    dmn_path = resolve_in_sandbox(root, dmn)
    out_path = resolve_in_sandbox(root, out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        render_drd(str(dmn_path), str(out_path))
    except SystemExit as e:
        if e.code == 2:
            raise typer.Exit(code=2)
        raise
    typer.echo(str(out_path))


@app.command("version")
def cli_version():
    try:
        from importlib.metadata import version as _pkg_version

        typer.echo(_pkg_version("chatchw"))
    except Exception:
        typer.echo("0.1.0")


if __name__ == "__main__":
    app()

