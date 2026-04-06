from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class ToyLESDataset(Dataset):
    """Simple `.npz` dataset reader for the fixed-size toy system."""

    def __init__(self, path: str | Path, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got {split}")

        archive = np.load(Path(path), allow_pickle=True)
        indices = archive[f"{split}_idx"]

        self.positions = torch.from_numpy(archive["positions"][indices]).float()
        self.types = torch.from_numpy(archive["types"][indices]).long()
        self.energy = torch.from_numpy(archive["energy"][indices]).float()
        self.energy_short = torch.from_numpy(archive["energy_short"][indices]).float()
        self.energy_long = torch.from_numpy(archive["energy_long"][indices]).float()
        self.forces = torch.from_numpy(archive["forces"][indices]).float()
        self.true_charges = torch.from_numpy(archive["true_charges"][indices]).float()

    def __len__(self) -> int:
        return self.positions.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "positions": self.positions[index],
            "types": self.types[index],
            "energy": self.energy[index],
            "energy_short": self.energy_short[index],
            "energy_long": self.energy_long[index],
            "forces": self.forces[index],
            "true_charges": self.true_charges[index],
        }
