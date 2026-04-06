#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from toy_les.data_gen import build_default_bundle
from toy_les.model import ModelConfig
from toy_les.train import TrainConfig, run_learning_curve


def load_config(path: Path | None) -> dict:
    cfg = build_default_bundle()
    cfg["model"] = ModelConfig().to_dict()
    cfg["train"] = TrainConfig().to_dict()
    if path is None:
        return cfg

    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config at {path} must parse to a mapping.")
    for key, value in loaded.items():
        if isinstance(value, dict) and key in cfg and isinstance(cfg[key], dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toy LES learning-curve experiments.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--subset-sizes", type=int, nargs="+", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg_dict = dict(cfg["train"])
    if args.epochs is not None:
        train_cfg_dict["epochs"] = args.epochs
    if args.device is not None:
        train_cfg_dict["device"] = args.device
    train_cfg_dict["save_dir"] = str((ROOT / train_cfg_dict["save_dir"]).resolve())

    train_cfg = TrainConfig.from_dict(train_cfg_dict)
    model_cfg = ModelConfig(**cfg["model"])
    dataset_path = args.dataset if args.dataset is not None else ROOT / cfg["output"]["path"]
    if not dataset_path.exists():
        raise SystemExit(
            f"Dataset not found at {dataset_path}. Run `python scripts/generate_data.py` first."
        )

    summary = run_learning_curve(
        dataset_path=dataset_path,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        seed=cfg["seed"],
        subset_sizes=args.subset_sizes,
        modes=["sr", "sr_lr"],
    )

    print("Learning curve summary:")
    for model_name, results in summary["results"].items():
        for subset_size, entry in results.items():
            print(f"  {model_name} n={subset_size}: {entry['test_metrics']}")
    summary_path = Path(train_cfg.save_dir) / f"learning_curve_seed{cfg['seed']}.json"
    print(f"Saved summary to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
