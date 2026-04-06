from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import ToyLESDataset
from .model import ModelConfig, ShortRangeOnlyModel, ToyLESModel


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    epochs: int = 150
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    energy_weight: float = 1.0
    force_weight: float = 100.0
    grad_clip_norm: float = 5.0
    device: str = "cpu"
    log_interval: int = 10
    run_name: str | None = None
    save_dir: str = "outputs/runs"

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TrainConfig":
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(model_name: str, model_cfg: ModelConfig) -> torch.nn.Module:
    if model_name == "sr":
        return ShortRangeOnlyModel(model_cfg)
    if model_name == "sr_lr":
        return ToyLESModel(model_cfg)
    raise ValueError(f"Unknown model name: {model_name}")


def build_dataloaders(
    dataset_path: str | Path,
    train_cfg: TrainConfig,
    seed: int,
    train_subset_size: int | None = None,
) -> Dict[str, DataLoader]:
    dataset_path = Path(dataset_path)
    train_ds = ToyLESDataset(dataset_path, split="train")
    val_ds = ToyLESDataset(dataset_path, split="val")
    test_ds = ToyLESDataset(dataset_path, split="test")

    if train_subset_size is not None:
        if train_subset_size <= 0:
            raise ValueError("train_subset_size must be positive.")
        train_subset_size = min(train_subset_size, len(train_ds))
        rng = np.random.default_rng(seed)
        subset_idx = rng.choice(len(train_ds), size=train_subset_size, replace=False)
        train_ds = Subset(train_ds, np.sort(subset_idx).tolist())

    return {
        "train": DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False),
    }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def charge_metrics(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    predicted = predicted.detach().reshape(-1)
    target = target.detach().reshape(-1)

    direct_mse = torch.mean((predicted - target) ** 2)
    flipped_mse = torch.mean((-predicted - target) ** 2)
    aligned = predicted if direct_mse <= flipped_mse else -predicted

    aligned_np = aligned.cpu().numpy()
    target_np = target.cpu().numpy()
    if np.std(aligned_np) < 1e-12 or np.std(target_np) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(aligned_np, target_np)[0, 1])

    return {
        "charge_mae": float(torch.mean(torch.abs(aligned - target)).item()),
        "charge_rmse": float(torch.sqrt(torch.mean((aligned - target) ** 2)).item()),
        "charge_corr": corr,
    }


def compute_batch_metrics(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    energy_error = outputs["energy_total"] - batch["energy"]
    force_error = outputs["forces"] - batch["forces"]
    metrics = {
        "energy_mae": float(torch.mean(torch.abs(energy_error)).item()),
        "energy_rmse": float(torch.sqrt(torch.mean(energy_error**2)).item()),
        "force_mae": float(torch.mean(torch.abs(force_error)).item()),
        "force_rmse": float(torch.sqrt(torch.mean(force_error**2)).item()),
    }

    if "latent_charges" in outputs:
        metrics.update(charge_metrics(outputs["latent_charges"], batch["true_charges"]))
    return metrics


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    train_cfg: TrainConfig,
) -> Dict[str, torch.Tensor]:
    n_particles = batch["positions"].shape[1]
    energy_loss = torch.mean(((outputs["energy_total"] - batch["energy"]) / n_particles) ** 2)
    force_loss = torch.mean((outputs["forces"] - batch["forces"]) ** 2)
    total_loss = train_cfg.energy_weight * energy_loss + train_cfg.force_weight * force_loss
    return {
        "loss": total_loss,
        "energy_loss": energy_loss,
        "force_loss": force_loss,
    }


def _aggregate_epoch_stats(step_stats: list[Dict[str, float]]) -> Dict[str, float]:
    if not step_stats:
        return {}
    keys = step_stats[0].keys()
    return {
        key: float(sum(entry[key] for entry in step_stats) / len(step_stats))
        for key in keys
    }


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    step_stats: list[Dict[str, float]] = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["positions"], batch["types"], compute_forces=True)
        losses = compute_losses(outputs, batch, train_cfg)
        metrics = compute_batch_metrics(outputs, batch)

        if training:
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if train_cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip_norm)
            optimizer.step()

        step_stats.append(
            {
                "loss": float(losses["loss"].detach().item()),
                "energy_loss": float(losses["energy_loss"].detach().item()),
                "force_loss": float(losses["force_loss"].detach().item()),
                **metrics,
            }
        )

    return _aggregate_epoch_stats(step_stats)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    with torch.enable_grad():
        return run_epoch(model, dataloader, train_cfg, device, optimizer=None)


def _format_metrics(metrics: Dict[str, float]) -> str:
    keys = ["loss", "energy_rmse", "force_rmse", "charge_corr"]
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return ", ".join(parts)


def train_model(
    dataset_path: str | Path,
    model_name: str,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
    train_subset_size: int | None = None,
) -> Dict[str, Any]:
    set_seed(seed)

    device = torch.device(train_cfg.device)
    dataloaders = build_dataloaders(dataset_path, train_cfg, seed=seed, train_subset_size=train_subset_size)
    model = build_model(model_name, model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    if train_subset_size is None:
        run_name = train_cfg.run_name or f"{model_name}_seed{seed}"
    else:
        run_name = train_cfg.run_name or f"{model_name}_n{train_subset_size}_seed{seed}"
    run_dir = Path(train_cfg.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history: list[Dict[str, Any]] = []
    best_state: Dict[str, Any] | None = None
    best_val_loss = math.inf

    for epoch in range(1, train_cfg.epochs + 1):
        train_metrics = run_epoch(model, dataloaders["train"], train_cfg, device, optimizer=optimizer)
        val_metrics = evaluate_model(model, dataloaders["val"], train_cfg, device)
        epoch_record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_record)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }

        if epoch == 1 or epoch % train_cfg.log_interval == 0 or epoch == train_cfg.epochs:
            print(
                f"[{model_name}] epoch {epoch:04d} | "
                f"train: {_format_metrics(train_metrics)} | "
                f"val: {_format_metrics(val_metrics)}"
            )

    assert best_state is not None
    model.load_state_dict(best_state["model_state_dict"])
    test_metrics = evaluate_model(model, dataloaders["test"], train_cfg, device)

    checkpoint = {
        "model_name": model_name,
        "seed": seed,
        "train_subset_size": train_subset_size,
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "best_epoch": best_state["epoch"],
        "best_val_metrics": best_state["val_metrics"],
        "test_metrics": test_metrics,
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, run_dir / "best.pt")

    summary = {
        "model_name": model_name,
        "seed": seed,
        "dataset_path": str(Path(dataset_path)),
        "train_subset_size": train_subset_size,
        "best_epoch": best_state["epoch"],
        "best_val_metrics": best_state["val_metrics"],
        "test_metrics": test_metrics,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_ablation(
    dataset_path: str | Path,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
    modes: list[str] | None = None,
) -> Dict[str, Any]:
    modes = modes or ["sr", "sr_lr"]
    results = {}
    for model_name in modes:
        run_cfg = TrainConfig(**{**train_cfg.to_dict(), "run_name": f"{model_name}_seed{seed}"})
        results[model_name] = train_model(dataset_path, model_name, model_cfg, run_cfg, seed)

    summary = {
        "seed": seed,
        "dataset_path": str(Path(dataset_path)),
        "results": {
            name: {
                "best_epoch": result["best_epoch"],
                "best_val_metrics": result["best_val_metrics"],
                "test_metrics": result["test_metrics"],
            }
            for name, result in results.items()
        },
    }

    save_dir = Path(train_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / f"ablation_seed{seed}.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_learning_curve(
    dataset_path: str | Path,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
    subset_sizes: list[int],
    modes: list[str] | None = None,
) -> Dict[str, Any]:
    modes = modes or ["sr", "sr_lr"]
    results: Dict[str, Dict[int, Dict[str, Any]]] = {mode: {} for mode in modes}

    for subset_size in subset_sizes:
        for model_name in modes:
            run_cfg = TrainConfig(
                **{
                    **train_cfg.to_dict(),
                    "run_name": f"{model_name}_n{subset_size}_seed{seed}",
                }
            )
            results[model_name][subset_size] = train_model(
                dataset_path=dataset_path,
                model_name=model_name,
                model_cfg=model_cfg,
                train_cfg=run_cfg,
                seed=seed,
                train_subset_size=subset_size,
            )

    summary = {
        "seed": seed,
        "dataset_path": str(Path(dataset_path)),
        "subset_sizes": subset_sizes,
        "results": {
            model_name: {
                str(subset_size): {
                    "best_epoch": result["best_epoch"],
                    "best_val_metrics": result["best_val_metrics"],
                    "test_metrics": result["test_metrics"],
                }
                for subset_size, result in entries.items()
            }
            for model_name, entries in results.items()
        },
    }

    save_dir = Path(train_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / f"learning_curve_seed{seed}.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
