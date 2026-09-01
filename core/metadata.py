"""Stage metadata sidecars.

Every stage writes a small JSON file next to its output describing what it
produced and under which configuration.  The next stage *reads* that file
instead of re-deriving the information, which is what removes the hidden
coupling in the old pipeline: stage 3 used to have to know that stage 2 ran
seven cycles, that lifetime bins came from a hardcoded list, and that the final
weight column was called ``final_weight``.  Now it is all written down.

A sidecar always has:

    schema          int, bumped when the layout changes incompatibly
    stage           str, e.g. "reweight"
    created         ISO timestamp
    runid           the RunID dict
    inputs          list of paths this stage consumed
    config          the configuration actually used (not the defaults)
    ... stage-specific payload
"""

from __future__ import annotations

import getpass
import json
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .runid import RunID

SCHEMA = 1


def _git_revision() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def provenance() -> Dict[str, Any]:
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "git": _git_revision(),
    }


def write(
    path: os.PathLike | str,
    *,
    stage: str,
    runid: Optional[RunID] = None,
    inputs: Iterable[os.PathLike | str] = (),
    config: Optional[Dict[str, Any]] = None,
    **payload: Any,
) -> Path:
    """Write a sidecar and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: Dict[str, Any] = {
        "schema": SCHEMA,
        "stage": stage,
        **provenance(),
        "runid": runid.to_dict() if runid is not None else None,
        "inputs": [str(p) for p in inputs],
        "config": config or {},
    }
    doc.update(payload)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, default=_fallback)
    return path


def _fallback(obj: Any) -> Any:
    """Make numpy scalars/arrays and dataclass-ish objects JSON-serialisable."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def read(path: os.PathLike | str, *, expect_stage: Optional[str] = None) -> Dict[str, Any]:
    """Read a sidecar, failing loudly if it is missing or from the wrong stage."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing metadata sidecar {path}.\n"
            "Every stage writes one next to its output; if it is absent the "
            "input was produced by an older version of the pipeline, or the "
            "--tag / --out-root you passed does not point at that output."
        )
    with open(path) as f:
        doc = json.load(f)
    if doc.get("schema") != SCHEMA:
        raise ValueError(
            f"{path}: schema {doc.get('schema')} but this code expects {SCHEMA}. "
            "Re-run the producing stage."
        )
    if expect_stage is not None and doc.get("stage") != expect_stage:
        raise ValueError(
            f"{path}: expected a '{expect_stage}' sidecar but found "
            f"'{doc.get('stage')}'."
        )
    return doc


def runid_of(doc: Dict[str, Any]) -> RunID:
    if not doc.get("runid"):
        raise ValueError("Sidecar has no runid")
    return RunID.from_dict(doc["runid"])