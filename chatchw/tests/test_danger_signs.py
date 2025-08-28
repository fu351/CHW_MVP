from pathlib import Path

from chatchw.cli import app
from chatchw.engine import decide
from chatchw.schema import ContextFlags, EncounterInput, Observation, Symptoms
from typer.testing import CliRunner


def test_convulsion_forces_hospital(tmp_path: Path):
    runner = CliRunner()
    root = tmp_path / "sbx"
    res = runner.invoke(app, ["init-sandbox", "--root", str(root)])
    assert res.exit_code == 0

    from chatchw.cli import _load_rules_from_dir
    rp = _load_rules_from_dir(root, "models/rules")

    enc = EncounterInput(
        age_months=6,
        sex="f",
        observations=[
            {"id": "temp", "value": 37.5},
            {"id": "resp_rate", "value": 22.0},
            {"id": "muac_mm", "value": 130.0},
        ],
        symptoms=Symptoms(feels_very_hot=False, blood_in_stool=False, diarrhea_days=1, convulsion=True, edema_both_feet=False),
        context=ContextFlags(malaria_present=True, cholera_present=False, stockout={"antimalarial": True}),
    )
    dec = decide(enc, rp)
    assert dec.triage == "hospital"
    assert any(t.rule_id == "FEV-99" for t in dec.trace)

