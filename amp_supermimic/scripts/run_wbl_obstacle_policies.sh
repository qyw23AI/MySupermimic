#!/usr/bin/env bash
# Train independent WBL obstacle actors, then evaluate each at competition geometry.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/../.."

mode=${1:-all}
output=${2:-amp_supermimic/outputs/physx/obstacles-$(date +%Y%m%d-%H%M%S)}
case "$mode" in
  all) tasks=(bridge_trot high_wall) ;;
  bridge_trot|high_wall) tasks=("$mode") ;;
  *) echo "Usage: $0 [all|bridge_trot|high_wall] [run directory]" >&2; exit 2 ;;
esac

python=${WBL_PYTHON:-/home/qyw/miniconda3/envs/himloco/bin/python}
warm_start=${WBL_WARM_START:-amp_supermimic/outputs/trackerlab-amp/wblwbl_open_flat_10k/last.pt}
test -s "$warm_start"
mkdir -p "$output"

for kind in "${tasks[@]}"; do
  case "$kind" in
    bridge_trot) level=9; speed=.32 ;;
    high_wall) level=6; speed=.45 ;;
  esac
  task="wblwbl_dog_v1_physx_$kind"
  run="$output/$kind"
  evaluation="$output/$kind-target-eval"
  for path in "$run" "$evaluation" "$run.log" "$evaluation.log"; do
    if [[ -e "$path" ]]; then
      echo "Output already exists: $path. Choose a new run directory." >&2
      exit 1
    fi
  done
  "$python" -B -m amp_supermimic.isaaclab_migration preflight --task "$task" \
    >"$output/$kind-preflight.json"
  "$python" -B -u -m amp_supermimic.isaaclab_migration train \
    --task "$task" --run-dir "$run" --warm-start "$warm_start" \
    --num-envs "${WBL_NUM_ENVS:-512}" --iterations "${WBL_ITERATIONS:-6000}" 2>&1 | tee "$run.log"
  "$python" -B -u -m amp_supermimic.isaaclab_migration evaluate \
    --checkpoint "$run/last.pt" --output "$evaluation" --level "$level" \
    --seed 801 --num-envs 16 --steps 2000 --command "$speed" 0 0 2>&1 | tee "$evaluation.log"
  cat "$evaluation/evaluation.json"
done
