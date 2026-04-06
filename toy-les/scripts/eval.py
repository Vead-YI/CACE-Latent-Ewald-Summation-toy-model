#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toy_les.eval import (
    plot_learning_curve_from_summary,
    predict_split,
    save_comparison_plots,
    save_evaluation_plots,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate toy LES checkpoints and save plots.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="*", default=[])
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "figures")
    parser.add_argument("--learning-curve-summary", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"predictions": [], "plots": {}}

    predictions = []
    for checkpoint_path in args.checkpoint:
        prediction = predict_split(
            checkpoint_path=checkpoint_path,
            dataset_path=args.dataset,
            split=args.split,
        )
        predictions.append(prediction)
        saved = save_evaluation_plots(prediction, args.output_dir)
        report["predictions"].append(
            {
                "checkpoint": str(checkpoint_path),
                "model_name": prediction["model_name"],
                "metrics": prediction["metrics"],
                "plots": saved,
            }
        )

    if len(predictions) >= 2:
        report["plots"]["comparison"] = save_comparison_plots(predictions, args.output_dir)

    if args.learning_curve_summary is not None:
        report["plots"]["learning_curve"] = plot_learning_curve_from_summary(
            args.learning_curve_summary,
            args.output_dir,
        )

    report_path = args.output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved evaluation report to {report_path}")
    print(f"Saved figures under {args.output_dir}")


if __name__ == "__main__":
    main()
