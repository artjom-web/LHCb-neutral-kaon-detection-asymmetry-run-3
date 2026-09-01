#!/usr/bin/env bash
#
# Chain driver: run s1 -> s2 -> s3 -> s4 -> s5 through HTCondor, waiting for
# each stage's DAG to fully drain before submitting the next.  Stage 6 is a
# single quick job, run manually (its command is printed at the end).
#
#   ./htcondor/run_chain.sh            # regenerate DAGs, then run s1..s5
#   ./htcondor/run_chain.sh 3 4        # only a subset of stages, in order
#
# Always runs from the repo root.  Requires KSPI_OUT_ROOT (and PYTHON if
# overriding) in the submit-time environment (getenv=true passes it to jobs).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
FOLDER="$(basename "$HERE")"

PY="${PYTHON:-python3}"
ORDER=(s1 s2 s3 s4 s5)
if [[ $# -gt 0 ]]; then
    ORDER=()
    for n in "$@"; do ORDER+=( "s${n}" ); done
fi

echo "==> Generating HTCondor files for all stages"
$PY "$FOLDER/generate_dag.py" --all
echo

for stage in "${ORDER[@]}"; do
    run_dir="$(ls -1dt "$FOLDER/$stage"/run_* 2>/dev/null | head -1)"
    dag="${run_dir}/${stage}.dag"
    if [[ ! -f "$dag" ]]; then
        $PY "$FOLDER/generate_dag.py" --stage "$stage"
    fi

    echo "=================================================================="
    echo "==> Submitting $stage ($(grep -c '^JOB ' "$dag") node(s))"
    echo "=================================================================="
    # Clear stale DAGMan bookkeeping so condor_submit_dag never complains.
    rm -f "${dag}.dagman."* "${dag}.condor.sub" "${dag}.lib."*
    condor_submit_dag "$dag"
    echo "    ... waiting for ${dag}.dagman.log"
    condor_wait "${dag}.dagman.log"
    echo "==> $stage finished"
    echo
done

echo "=================================================================="
echo "  s1..s5 chain complete."
echo "=================================================================="
echo
echo "Run s6 manually once happy with s3, e.g. via HTCondor:"
echo "  python3 ${FOLDER}/generate_dag.py --stage s6 && condor_submit_dag ${FOLDER}/s6/s6.dag"