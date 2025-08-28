from pathlib import Path

from chatchw.cli import app
from typer.testing import CliRunner


def test_render_bpmn_and_dmn(tmp_path: Path):
    runner = CliRunner()
    root = tmp_path / "sbx"
    res = runner.invoke(app, ["init-sandbox", "--root", str(root)])
    assert res.exit_code == 0

    gen = runner.invoke(
        app,
        [
            "generate-bpmn",
            "--root",
            str(root),
            "--rules",
            "models/rules",
            "--out",
            "exports/chatchw.bpmn",
        ],
    )
    assert gen.exit_code == 0
    bpmn_path = Path(gen.stdout.strip())
    assert bpmn_path.exists() and bpmn_path.stat().st_size > 0

    rend = runner.invoke(
        app,
        [
            "render-bpmn",
            "--root",
            str(root),
            "--bpmn",
            "exports/chatchw.bpmn",
            "--out",
            "exports/chatchw_bpmn.svg",
        ],
    )
    assert rend.exit_code == 0
    bpmn_svg = Path(rend.stdout.strip())
    assert bpmn_svg.exists() and bpmn_svg.stat().st_size > 0

    gen_dmn = runner.invoke(
        app,
        [
            "generate-dmn",
            "--root",
            str(root),
            "--rules",
            "models/rules",
            "--out",
            "exports/chatchw.dmn",
        ],
    )
    assert gen_dmn.exit_code == 0
    dmn_path = Path(gen_dmn.stdout.strip())
    assert dmn_path.exists() and dmn_path.stat().st_size > 0

    rend_dmn = runner.invoke(
        app,
        [
            "render-dmn",
            "--root",
            str(root),
            "--dmn",
            "exports/chatchw.dmn",
            "--out",
            "exports/chatchw_drd.svg",
        ],
    )
    assert rend_dmn.exit_code in (0, 2)
    if rend_dmn.exit_code == 0:
        dmn_svg = Path(rend_dmn.stdout.strip())
        assert dmn_svg.exists() and dmn_svg.stat().st_size > 0
    else:
        dot_path = root / "exports/chatchw_drd.dot"
        assert dot_path.exists() and dot_path.stat().st_size > 0

