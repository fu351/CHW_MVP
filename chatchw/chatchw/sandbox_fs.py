from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SandboxPathError(ValueError):
    pass


def resolve_in_sandbox(root: str | Path, rel: str | Path) -> Path:
    root_path = Path(root).resolve()
    rel_path = (root_path / rel).resolve()
    try:
        rel_path.relative_to(root_path)
    except Exception:
        raise SandboxPathError("path escapes sandbox")
    return rel_path


def ensure_dirs(root: str | Path) -> dict[str, Path]:
    root_path = Path(root).resolve()
    dirs = {
        "root": root_path,
        "logs": resolve_in_sandbox(root_path, "logs"),
        "exports": resolve_in_sandbox(root_path, "exports"),
        "inputs": resolve_in_sandbox(root_path, "inputs"),
        "models": resolve_in_sandbox(root_path, "models"),
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def read_json(root: str | Path, rel: str | Path) -> Any:
    p = resolve_in_sandbox(root, rel)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(root: str | Path, rel: str | Path, obj: Any) -> Path:
    p = resolve_in_sandbox(root, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return p


def append_jsonl(root: str | Path, rel: str | Path, obj: Any) -> Path:
    p = resolve_in_sandbox(root, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")
    return p


def write_bytes(root: str | Path, rel: str | Path, data: bytes) -> Path:
    p = resolve_in_sandbox(root, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        f.write(data)
    return p

