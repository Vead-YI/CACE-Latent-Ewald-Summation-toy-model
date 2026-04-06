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

from toy_les.data_gen import DatasetConfig, build_default_bundle
from toy_les.model import ModelConfig
from toy_les.physics import ToyPhysicsConfig
from toy_les.train import TrainConfig, train_model


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
    parser = argparse.ArgumentParser(description="Train a minimal toy LES model.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    parser.add_argument("--dataset", type=Path, default=None, help="Override dataset path from the config.")
    parser.add_argument("--model", choices=["sr", "sr_lr"], default="sr_lr")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    _ = DatasetConfig.from_dict(cfg["dataset"])
    _ = ToyPhysicsConfig.from_dict(cfg["physics"])
    model_cfg = ModelConfig(**cfg["model"])
    train_cfg_dict = dict(cfg["train"])
    if args.epochs is not None:
        train_cfg_dict["epochs"] = args.epochs
    if args.device is not None:
        train_cfg_dict["device"] = args.device
    if args.run_name is not None:
        train_cfg_dict["run_name"] = args.run_name
    train_cfg_dict["save_dir"] = str((ROOT / train_cfg_dict["save_dir"]).resolve())
    train_cfg = TrainConfig.from_dict(train_cfg_dict)

    dataset_path = args.dataset if args.dataset is not None else ROOT / cfg["output"]["path"]
    if not dataset_path.exists():
        raise SystemExit(
            f"Dataset not found at {dataset_path}. Run `python toy-les/scripts/generate_data.py` first."
        )

    summary = train_model(
        dataset_path=dataset_path,
        model_name=args.model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        seed=cfg["seed"],
    )

    run_name = train_cfg.run_name or f"{args.model}_seed{cfg['seed']}"
    print(f"Saved run outputs to {(ROOT / train_cfg.save_dir / run_name).resolve()}")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Test metrics: {summary['test_metrics']}")


if __name__ == "__main__":
    main()
