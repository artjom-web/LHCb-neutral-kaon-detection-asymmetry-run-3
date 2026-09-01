#!/usr/bin/env bash
#
# Worker for a stage-1 skim (scripts/s1_skim.py) HTCondor job.  All CLI
# arguments come from condor `arguments =` and are passed straight through.
#
#   * The repo root comes from env REPO (baked into every .sub by
#     generate_dag.py as `environment = "REPO=..."`); the BASH_SOURCE
#     fallback only matters if REPO is unset.
#   * Python defaults to `lb-conda default python3` (LHCb conda env, has
#     ROOT).  Override by exporting PYTHON="path/to/python" (also honors
#     multi-word commands).


set -euo pipefail

export PATH="/cvmfs/lhcb.cern.ch/bin:$PATH"

if [[ -n "${PYTHON:-}" ]]; then
    read -r -a PYTHON_CMD <<< "$PYTHON"
else
    PYTHON_CMD=(lb-conda default python3)
fi

if [[ -z "${REPO:-}" ]]; then
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

export KSPI_OUT_ROOT="${KSPI_OUT_ROOT:-/eos/user/a/ahulsber/scripts/analysis}"

if [[ ! -d "$REPO" ]]; then
    echo "ERROR: repo not found at \$REPO='$REPO' (baked into the .sub)." >&2
    exit 1
fi

cd "$REPO" || { echo "ERROR: cannot cd to $REPO" >&2; exit 1; }
exec "${PYTHON_CMD[@]}" scripts/s1_skim.py "$@"


