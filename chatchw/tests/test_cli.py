import json
from pathlib import Path

from chatchw.cli import app
from typer.testing import CliRunner


def test_cli_end_to_end(tmp_path: Path):
    runner = CliRunner()
    root = tmp_path / "sbx"

    res = runner.invoke(app, ["init-sandbox", "--root", str(root)])
    assert res.exit_code == 0

    res2 = runner.invoke(
        app,
        [
            "decide",
            "--root",
            str(root),
            "--input",
            "inputs/input.encounter.example.json",
            "--rules",
            "models/rules",
        ],
    )
    assert res2.exit_code == 0
    out_path = Path(res2.stdout.strip())
    assert out_path.exists()

    res3 = runner.invoke(
        app,
        [
            "export-csv",
            "--root",
            str(root),
            "--out",
            "exports/encounters.csv",
        ],
    )
    assert res3.exit_code == 0
    csv_path = Path(res3.stdout.strip())
    assert csv_path.exists()
    header_line = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header_line == "age_months,sex,temp,resp_rate,muac_mm,feels_very_hot,blood_in_stool,diarrhea_days,convulsion,edema_both_feet,malaria_present,cholera_present,triage,reasons,trace_rules"

