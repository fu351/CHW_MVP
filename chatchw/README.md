# ChatCHW v0.1 (Offline, Sandboxed CLI MVP)

ChatCHW is an offline CLI that applies deterministic rules for common community health workflows, generates intermediary BPMN and DMN artifacts, and renders them locally to SVG/PNG. All file I/O is restricted to a sandbox root. Network access is disabled at runtime.

## Requirements

- Python 3.11
- Graphviz system binary installed and on PATH (for DMN DRD rendering)
- The CLI uses only local file system operations under a sandbox directory

## Install

```bash
python -m venv .venv
# On Linux/macOS
source .venv/bin/activate
# On Windows PowerShell
.venv\Scripts\Activate.ps1

pip install --upgrade pip
cd chatchw
pip install -e .
```

Verify Graphviz is installed:
```bash
dot -V
```

## CLI Overview

All commands accept `--root` which defaults to `./sandbox`. All reads/writes must be under this sandbox root.

- Initialize a sandbox with examples and bundled rules:
```bash
chatchw init-sandbox --root ./sandbox
```

- Run a decision on an example input using sandbox-copied rules:
```bash
chatchw decide --root ./sandbox --input inputs/input.encounter.example.json --rules models/rules
```

- Export decisions to CSV:
```bash
chatchw export-csv --root ./sandbox --out exports/encounters.csv
```

- Validate rule files and boundaries:
```bash
chatchw validate --root ./sandbox --rules models/rules
```

- Generate BPMN and render:
```bash
chatchw generate-bpmn --root ./sandbox --rules models/rules --out exports/chatchw.bpmn
chatchw render-bpmn --root ./sandbox --bpmn exports/chatchw.bpmn --out exports/chatchw_bpmn.svg
```

- Generate DMN and render the DRD:
```bash
chatchw generate-dmn --root ./sandbox --rules models/rules --out exports/chatchw.dmn
chatchw render-dmn --root ./sandbox --dmn exports/chatchw.dmn --out exports/chatchw_drd.svg
```

- Version:
```bash
chatchw version
```

## Packaging Notes

- Installable via pip or pipx:
  - `pipx install .`
- PyInstaller is not included as a dependency, but you can build a standalone binary locally if needed:
  - `pyinstaller --onefile -n chatchw chatchw/cli.py`
  - Ensure you carry the `rules/` and other runtime assets into your sandbox via `init-sandbox`.

## Deps (pinned)

- Typer (CLI)
- Pydantic v2 (schemas)
- pytest, hypothesis (tests)
- python-dateutil, rfc3339-validator (timestamps/validation)
- pm4py (BPMN import/export/visualization)
- graphviz (Python wrapper; Graphviz binary required on PATH)

## Offline and Sandboxed

- Network is disabled at runtime. Any attempt to use sockets or HTTP raises “network disabled”.
- All file access is constrained to a sandbox root. Paths that escape the sandbox raise “path escapes sandbox”.

