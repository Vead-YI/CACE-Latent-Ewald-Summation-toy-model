#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toy_les.data_gen import DatasetConfig, build_default_bundle, generate_dataset, save_dataset
from toy_les.physics import ToyPhysicsConfig


def _load_config(path: Path | None) -> dict:
    cfg = build_default_bundle()
    if path is None:
        return cfg
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to load a custom config file. Install `pyyaml` or run without --config.") from exc

    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config at {path} must parse to a mapping.")
    for key, value in loaded.items():
        if isinstance(value, dict) and key in cfg and isinstance(cfg[key], dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def _apply_preset(cfg: dict, preset: str) -> dict:
    if preset == "default":
        return cfg
    if preset == "smoke":
        cfg["dataset"].update(
            {
                "n_train": 32,
                "n_val": 8,
                "n_test": 8,
                "n_particles": 8,
                "box_size": 4.5,
                "min_dist": 0.8,
            }
        )
        cfg["output"]["path"] = "data/processed/toy_les_smoke.npz"
        return cfg
    raise SystemExit(f"Unknown preset: {preset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an analytic toy LES dataset.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config.")
    parser.add_argument(
        "--preset",
        choices=["default", "smoke"],
        default="default",
        help="Convenience preset for quick testing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output path from the config.",
    )
    args = parser.parse_args()

    cfg = _apply_preset(_load_config(args.config), args.preset)
    dataset_cfg = DatasetConfig.from_dict(cfg["dataset"])
    physics_cfg = ToyPhysicsConfig.from_dict(cfg["physics"])
    output_path = args.output if args.output is not None else ROOT / cfg["output"]["path"]

    dataset = generate_dataset(dataset_cfg, physics_cfg)
    saved_path = save_dataset(dataset, output_path)

    print(f"Saved dataset to {saved_path}")
    print(f"Samples: train={dataset_cfg.n_train}, val={dataset_cfg.n_val}, test={dataset_cfg.n_test}")
    print(f"Particles per sample: {dataset_cfg.n_particles}")
    print(f"Energy array shape: {dataset['energy'].shape}")
    print(f"Forces array shape: {dataset['forces'].shape}")


if __name__ == "__main__":
    main()
