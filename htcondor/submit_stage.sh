#!/usr/bin/env bash
#
# Submit a single stage's DAG.  Regenerates that stage's files from
# htcondor.yaml, clears any stale DAGMan bookkeeping, then submits and
# (optionally) waits for it to finish.  No manual cleanup needed between runs.
#
#   ./htcondor/submit_stage.sh s3            # generate + submit + wait
#   ./htcondor/submit_stage.sh s3 --no-wait  # generate + submit only
#   ./htcondor/submit_stage.sh s3 -f         # force-overwrite existing DAG files
#
# Always runs from the repo root (DAG paths are repo-root-relative).  Run from
# anywhere; requires KSPI_OUT_ROOT (and PYTHON, if overriding) in the env.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
FOLDER="$(basename "$HERE")"
PY="${PYTHON:-python3}"

stage="${1:?usage: submit_stage.sh sN [--no-wait] [-f]}"
shift

case "$stage" in
    s1|s2|s3|s4|s5|s6) ;;
    *) echo "!! unknown stage: $stage (expected s1..s6)"; exit 2 ;;
esac

no_wait=0
force=""
for arg in "$@"; do
    case "$arg" in
        --no-wait) no_wait=1 ;;
        -f|--force) force="-f" ;;
        *) echo "!! unknown option: $arg"; exit 2 ;;
    esac
done

echo "==> generating $stage"
$PY "$FOLDER/generate_dag.py" --stage "$stage"

run_dir="$(ls -1dt "$FOLDER/$stage"/run_* 2>/dev/null | head -1)"
dag="${run_dir}/${stage}.dag"
if [[ ! -f "$dag" ]]; then
    echo "!! no DAG at $dag"; exit 1
fi

# Clear stale DAGMan bookkeeping from any previous run of this stage so
# condor_submit_dag never complains about existing files.
rm -f "${dag}.dagman."* "${dag}.condor.sub" "${dag}.lib."*

echo "==> submitting $dag ($(grep -c '^JOB ' "$dag") node(s))"
condor_submit_dag $force "$dag"

if [[ "$no_wait" -eq 0 ]]; then
    echo "    waiting for ${dag}.dagman.log"
    condor_wait "${dag}.dagman.log"
    echo "==> $stage finished"
else
    echo "==> $stage submitted (not waiting)"
fi