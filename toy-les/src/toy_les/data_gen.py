from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .physics import ToyPhysicsConfig, compute_energy_and_forces


@dataclass(frozen=True)
class DatasetConfig:
    seed: int = 42
    n_train: int = 2000
    n_val: int = 250
    n_test: int = 250
    n_particles: int = 16
    box_size: float = 6.0
    min_dist: float = 0.8
    max_position_attempts: int = 5000

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_val + self.n_test

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "DatasetConfig":
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_default_bundle() -> Dict[str, Any]:
    return {
        "seed": 42,
        "dataset": DatasetConfig().to_dict(),
        "physics": ToyPhysicsConfig().to_dict(),
        "output": {"path": "data/processed/toy_les_dataset.npz"},
    }


def sample_types(n_particles: int, rng: np.random.Generator) -> np.ndarray:
    if n_particles % 2 != 0:
        raise ValueError("n_particles must be even so the toy system is exactly charge neutral.")
    types = np.zeros(n_particles, dtype=np.int64)
    types[: n_particles // 2] = 1
    rng.shuffle(types)
    return types


def sample_positions(
    n_particles: int,
    box_size: float,
    min_dist: float,
    max_attempts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    positions = np.zeros((n_particles, 3), dtype=np.float64)
    for idx in range(n_particles):
        accepted = False
        for _ in range(max_attempts):
            candidate = rng.uniform(0.0, box_size, size=3)
            if idx == 0:
                positions[idx] = candidate
                accepted = True
                break
            distances = np.linalg.norm(positions[:idx] - candidate[None, :], axis=-1)
            if np.all(distances >= min_dist):
                positions[idx] = candidate
                accepted = True
                break
        if not accepted:
            raise RuntimeError(
                f"Failed to place particle {idx} after {max_attempts} attempts. "
                "Increase box_size, reduce n_particles, or lower min_dist."
            )
    return positions


def generate_sample(
    dataset_cfg: DatasetConfig,
    physics_cfg: ToyPhysicsConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    types = sample_types(dataset_cfg.n_particles, rng)
    positions = sample_positions(
        n_particles=dataset_cfg.n_particles,
        box_size=dataset_cfg.box_size,
        min_dist=dataset_cfg.min_dist,
        max_attempts=dataset_cfg.max_position_attempts,
        rng=rng,
    )

    pos_tensor = torch.tensor(positions, dtype=torch.float64)
    type_tensor = torch.tensor(types, dtype=torch.long)
    result = compute_energy_and_forces(pos_tensor, type_tensor, physics_cfg)

    return {
        "positions": positions.astype(np.float32),
        "types": types.astype(np.int64),
        "energy": np.asarray(result["energy_total"].detach().cpu().numpy(), dtype=np.float32),
        "energy_short": np.asarray(result["energy_short"].detach().cpu().numpy(), dtype=np.float32),
        "energy_long": np.asarray(result["energy_long"].detach().cpu().numpy(), dtype=np.float32),
        "forces": result["forces"].detach().cpu().numpy().astype(np.float32),
        "true_charges": result["true_charges"].detach().cpu().numpy().astype(np.float32),
    }


def generate_dataset(
    dataset_cfg: DatasetConfig,
    physics_cfg: ToyPhysicsConfig,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(dataset_cfg.seed)
    samples = [generate_sample(dataset_cfg, physics_cfg, rng) for _ in range(dataset_cfg.n_total)]

    positions = np.stack([sample["positions"] for sample in samples], axis=0)
    types = np.stack([sample["types"] for sample in samples], axis=0)
    energy = np.stack([sample["energy"] for sample in samples], axis=0)
    energy_short = np.stack([sample["energy_short"] for sample in samples], axis=0)
    energy_long = np.stack([sample["energy_long"] for sample in samples], axis=0)
    forces = np.stack([sample["forces"] for sample in samples], axis=0)
    true_charges = np.stack([sample["true_charges"] for sample in samples], axis=0)

    indices = np.arange(dataset_cfg.n_total)
    rng.shuffle(indices)

    train_end = dataset_cfg.n_train
    val_end = train_end + dataset_cfg.n_val

    return {
        "positions": positions,
        "types": types,
        "energy": energy,
        "energy_short": energy_short,
        "energy_long": energy_long,
        "forces": forces,
        "true_charges": true_charges,
        "box_size": np.asarray([dataset_cfg.box_size], dtype=np.float32),
        "train_idx": indices[:train_end].astype(np.int64),
        "val_idx": indices[train_end:val_end].astype(np.int64),
        "test_idx": indices[val_end:].astype(np.int64),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "seed": dataset_cfg.seed,
                    "dataset": dataset_cfg.to_dict(),
                    "physics": physics_cfg.to_dict(),
                    "type_encoding": {"0": "negative", "1": "positive"},
                },
                indent=2,
                sort_keys=True,
            )
        ),
    }


def save_dataset(dataset: Dict[str, np.ndarray], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    return output_path
