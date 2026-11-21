#!/usr/bin/env bash
# Train and evaluate WBL on the fixed 30 cm wall using the vendored tracker.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
vendor="$root/third_party/supermimic_quadruped"
runs="$root/amp_supermimic/outputs/wall_jump_tracking"
run=${2:-"$runs/$(date +%Y%m%d-%H%M%S)"}
export PYTHONPATH="$vendor/src:$root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

if [[ -n "${WBL_SM_PY:-}" ]]; then
    py="$WBL_SM_PY"
else
    mkdir -p "$runs"
    py="$runs/.venv/bin/python"
    if [[ ! -x "$py" ]] || ! "$py" -B -c "import mjlab, torch"; then
        uv_bin="${WBL_UV:-$(command -v uv || true)}"
        if [[ -z "$uv_bin" ]]; then
            bootstrap="${WBL_BOOTSTRAP_PY:-/home/qyw/miniconda3/envs/trackerlab_amp/bin/python}"
            "$bootstrap" -m pip install --target "$runs/.uv" uv
            uv_bin="$runs/.uv/bin/uv"
        fi
        UV_PROJECT_ENVIRONMENT="$runs/.venv" "$uv_bin" sync --project "$vendor" \
            --python "${WBL_PYTHON312:-3.12}" --locked --no-dev --no-install-project
    fi
fi

call() {
    "$py" -B -u -m amp_supermimic.wall_jump_tracking.cli --work "$run/input" "$@"
}
fit() {
    local stage=$1
    shift
    call train --stage "$stage" --output "$run/$stage" \
        --iterations "${WBL_ITERS:-6000}" --envs "${WBL_ENVS:-256}" \
        --batch "${WBL_BATCH:-256}" --horizon "${WBL_HORIZON:-16}" "$@" 2>&1 | tee "$run/$stage-train.log"
}
check() {
    local stage=$1
    shift
    call evaluate --stage "$stage" --output "$run/$stage-eval" \
        --envs "${WBL_EVAL_ENVS:-16}" --checkpoint "$run/$stage/ckpt_latest.pt" --perturb "$@"
}
has_success() {
    if ! "$py" -B - "$run/$1-eval/evaluation.json" <<'PY'
import json
import sys
from pathlib import Path

import hashlib

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text())
expected_stage = report_path.parent.name.removesuffix("-eval")
checkpoint = report_path.parent.parent / expected_stage / "ckpt_latest.pt"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
passed = (report["policy"] == "learned" and report["stage"] == expected_stage
          and report["successes"] > 0 and report["checkpoint_sha256"] == digest)
raise SystemExit(0 if passed else 1)
PY
    then
        echo "No verified $1 success. Review $run/$1-eval/evaluation.json before continuing." >&2
        return 1
    fi
}
first() {
    mkdir -p "$run"
    test ! -e "$run/first"
    if [[ -n "${WBL_FIRST_TEACHER:-}" ]]; then
        call prepare --first-teacher "$WBL_FIRST_TEACHER"
        fit first --teacher
    else
        call prepare
        fit first
    fi
    check first
}
complete() {
    has_success first
    call prepare-followup --evaluation "$run/first-eval" --checkpoint "$run/first/ckpt_latest.pt"
    fit second --warm-start "$run/first/ckpt_latest.pt"
    check second
    has_success second
    fit full --warm-start "$run/second/ckpt_latest.pt"
    check full --export-amp
    has_success full
}

case "${1:-first}" in
    first) first ;;
    continue) complete ;;
    all) first; complete ;;
    eval)
        stage=${3:-first}
        call evaluate --stage "$stage" --output "$run/$stage-recheck-$(date +%Y%m%d-%H%M%S)" \
            --envs "${WBL_EVAL_ENVS:-16}" --checkpoint "${4:-$run/$stage/ckpt_latest.pt}" --perturb --export-amp ;;
    *) echo "Usage: $0 [first|continue|all|eval] [run directory] [eval stage] [checkpoint]" >&2; exit 2 ;;
esac
