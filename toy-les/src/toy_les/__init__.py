"""Minimal toy LES package."""

from .data_gen import DatasetConfig, build_default_bundle, generate_dataset, save_dataset
from .dataset import ToyLESDataset
from .eval import (
    load_checkpoint,
    load_model_from_checkpoint,
    plot_learning_curve_from_summary,
    predict_split,
    save_comparison_plots,
    save_evaluation_plots,
)
from .model import ModelConfig, ShortRangeOnlyModel, ToyLESModel
from .physics import ToyPhysicsConfig, compute_energy_components, compute_true_charges
from .train import TrainConfig, run_ablation, run_learning_curve, train_model

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
    "load_checkpoint",
    "load_model_from_checkpoint",
    "plot_learning_curve_from_summary",
    "predict_split",
    "run_ablation",
    "run_learning_curve",
    "save_dataset",
    "save_comparison_plots",
    "save_evaluation_plots",
    "train_model",
]
