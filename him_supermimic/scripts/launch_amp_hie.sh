#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${HIM_PYTHON:-/home/qyw/miniconda3/envs/himloco/bin/python}"
MODE="${1:-train}"
if (( $# )); then shift; fi
case "$MODE" in
  train)
    OUTPUT="${AMP_OUTPUT:-$ROOT/him_supermimic/outputs/amp-hie-$(date +%Y%m%d-%H%M%S)}"
    exec "$PYTHON" -u -B "$ROOT/him_supermimic/scripts/train_amp_hie.py" --headless --output "$OUTPUT" "$@" ;;
  play)
    RUN="${1:?请提供训练输出目录}"; shift
    exec "$PYTHON" -u -B "$ROOT/him_supermimic/scripts/compare_online.py" --run "$RUN" --output "$RUN/play-$(date +%Y%m%d-%H%M%S)" "$@" ;;
  tensorboard)
    exec "$PYTHON" -m tensorboard.main --logdir "$ROOT/him_supermimic/outputs" --port 6006 "$@" ;;
  *) echo 'Usage: bash launch_amp_hie.sh train [args] | play RUN [args] | tensorboard'; exit 2 ;;
esac
