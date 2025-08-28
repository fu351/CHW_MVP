import json
from pathlib import Path

import pytest

from chatchw.engine import decide
from chatchw.schema import EncounterInput, Observation, Symptoms, ContextFlags, Rule
from chatchw.cli import app
from typer.testing import CliRunner


def _make_enc(temp: float, muac: float) -> EncounterInput:
    return EncounterInput(
        age_months=24,
        sex="m",
        observations=[
            {"id": "temp", "value": temp},
            {"id": "resp_rate", "value": 24.0},
            {"id": "muac_mm", "value": muac},
        ],
        symptoms=Symptoms(feels_very_hot=True, blood_in_stool=False, diarrhea_days=2, convulsion=False, edema_both_feet=False),
        context=ContextFlags(malaria_present=False, cholera_present=False),
    )


def test_fever_boundary_38_5(tmp_path: Path):
    runner = CliRunner()
    root = tmp_path / "sbx"
    res = runner.invoke(app, ["init-sandbox", "--root", str(root)])
    assert res.exit_code == 0

    rules_dir = "models/rules"
    enc = _make_enc(38.5, 130.0)
    from chatchw.cli import _load_rules_from_dir
    rp = _load_rules_from_dir(root, rules_dir)

    dec = decide(enc, rp)
    assert "fever.high" in dec.reasons

    enc2 = _make_enc(37.0, 130.0)
    dec2 = decide(enc2, rp)
    assert ("fever.high" in dec2.reasons) is False


def test_muac_boundary_115(tmp_path: Path):
    runner = CliRunner()
    root = tmp_path / "sbx"
    res = runner.invoke(app, ["init-sandbox", "--root", str(root)])
    assert res.exit_code == 0

    from chatchw.cli import _load_rules_from_dir
    rp = _load_rules_from_dir(root, "models/rules")

    enc = _make_enc(36.8, 114.0)
    dec = decide(enc, rp)
    assert dec.triage in ("clinic", "hospital")

    enc2 = _make_enc(36.8, 115.0)
    dec2 = decide(enc2, rp)
    assert dec2.triage in ("home", "clinic", "hospital")
    assert ("muac.low" in dec.reasons) is True
    assert ("muac.low" in dec2.reasons) is False

