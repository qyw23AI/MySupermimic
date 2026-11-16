#!/usr/bin/env python3
"""Convert an existing A5 report to TensorBoard events."""

import argparse
import json
from pathlib import Path
import sys

from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from him_supermimic.tensorboard_metrics import EVAL_TAGS, UPDATE_TAGS, log_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    output = args.output or args.report.parent / "tensorboard-backfill"
    output.mkdir(parents=True, exist_ok=False)
    with SummaryWriter(str(output)) as writer:
        log_metrics(writer, report["baseline"], EVAL_TAGS)
        for row in report["evaluations"]:
            log_metrics(writer, row, EVAL_TAGS)
        for row in report["updates"]:
            log_metrics(writer, row, UPDATE_TAGS)
    print(output.resolve())


if __name__ == "__main__":
    main()
