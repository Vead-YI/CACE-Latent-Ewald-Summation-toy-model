"""Minimal toy LES package."""

from .data_gen import DatasetConfig, build_default_bundle, generate_dataset, save_dataset
from .dataset import ToyLESDataset
from .model import ModelConfig, ShortRangeOnlyModel, ToyLESModel
from .physics import ToyPhysicsConfig, compute_energy_components, compute_true_charges
from .train import TrainConfig, run_ablation, train_model

__all__ = [
    "DatasetConfig",
    "ModelConfig",
    "ShortRangeOnlyModel",
    "TrainConfig",
    "ToyLESDataset",
    "ToyPhysicsConfig",
    "ToyLESModel",
    "build_default_bundle",
    "compute_energy_components",
    "compute_true_charges",
    "generate_dataset",
    "run_ablation",
    "save_dataset",
    "train_model",
]
