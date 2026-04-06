from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable

_MPL_DIR = Path(__file__).resolve().parents[2] / ".mplcache"
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))
_MPL_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_energy_parity(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path, title: str = "Energy parity") -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=18, alpha=0.75)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black")
    ax.set_xlabel("True total energy")
    ax.set_ylabel("Predicted total energy")
    ax.set_title(title)
    _save_figure(fig, output_path)


def plot_force_parity(f_true: np.ndarray, f_pred: np.ndarray, output_path: str | Path, title: str = "Force parity") -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(f_true.reshape(-1), f_pred.reshape(-1), s=6, alpha=0.35)
    lo = float(min(f_true.min(), f_pred.min()))
    hi = float(max(f_true.max(), f_pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black")
    ax.set_xlabel("True force component")
    ax.set_ylabel("Predicted force component")
    ax.set_title(title)
    _save_figure(fig, output_path)


def plot_charge_scatter(
    q_true: np.ndarray,
    q_pred: np.ndarray,
    output_path: str | Path,
    title: str = "Latent charge vs true charge",
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(q_true.reshape(-1), q_pred.reshape(-1), s=12, alpha=0.5)
    lo = float(min(q_true.min(), q_pred.min()))
    hi = float(max(q_true.max(), q_pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black")
    ax.set_xlabel("True charge")
    ax.set_ylabel("Predicted latent charge")
    ax.set_title(title)
    _save_figure(fig, output_path)


def plot_ablation_bar(
    metric_by_model: Dict[str, float],
    output_path: str | Path,
    title: str = "Ablation comparison",
    ylabel: str = "Test RMSE",
) -> None:
    labels = list(metric_by_model.keys())
    values = [metric_by_model[label] for label in labels]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(labels, values, color=["#6c8ebf", "#d98324", "#4aa564", "#c44e52"][: len(labels)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save_figure(fig, output_path)


def plot_learning_curve(
    curve_by_model: Dict[str, Dict[int, float]],
    output_path: str | Path,
    title: str = "Learning curve",
    ylabel: str = "Test RMSE",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name, values in curve_by_model.items():
        xs = sorted(values.keys())
        ys = [values[x] for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=model_name)
    ax.set_xlabel("Training subset size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _save_figure(fig, output_path)


def plot_configuration_snapshot(
    positions: np.ndarray,
    true_charges: np.ndarray,
    pred_charges: np.ndarray | None,
    output_path: str | Path,
    title: str = "Particle configuration",
) -> None:
    fig, axes = plt.subplots(1, 2 if pred_charges is not None else 1, figsize=(10 if pred_charges is not None else 5, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    scatter0 = axes[0].scatter(positions[:, 0], positions[:, 1], c=true_charges, cmap="coolwarm", s=70, edgecolor="black", linewidth=0.4)
    axes[0].set_title("True charges")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(scatter0, ax=axes[0], fraction=0.046, pad=0.04)

    if pred_charges is not None:
        scatter1 = axes[1].scatter(positions[:, 0], positions[:, 1], c=pred_charges, cmap="coolwarm", s=70, edgecolor="black", linewidth=0.4)
        axes[1].set_title("Predicted latent charges")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(scatter1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    _save_figure(fig, output_path)
