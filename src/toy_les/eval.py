from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ToyLESDataset
from .model import ModelConfig
from .plot_utils import (
    plot_ablation_bar,
    plot_charge_scatter,
    plot_configuration_snapshot,
    plot_energy_parity,
    plot_force_parity,
    plot_learning_curve,
)
from .train import TrainConfig, build_model, charge_metrics, compute_batch_metrics, move_batch_to_device


def _align_charge_sign(q_pred: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    direct = np.mean((q_pred - q_true) ** 2)
    flipped = np.mean((-q_pred - q_true) ** 2)
    return q_pred if direct <= flipped else -q_pred


def load_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(checkpoint_path), map_location=device)


def load_model_from_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model_cfg = ModelConfig(**checkpoint["model_config"])
    model = build_model(checkpoint["model_name"], model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_split(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    split: str = "test",
    batch_size: int = 64,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    dataset = ToyLESDataset(dataset_path, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device_t = torch.device(device)

    energy_true = []
    energy_pred = []
    forces_true = []
    forces_pred = []
    q_true = []
    q_pred = []
    positions = []
    types = []

    batch_metrics = []
    with torch.enable_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device_t)
            outputs = model(batch["positions"], batch["types"], compute_forces=True)
            batch_metrics.append(compute_batch_metrics(outputs, batch))

            energy_true.append(batch["energy"].detach().cpu().numpy())
            energy_pred.append(outputs["energy_total"].detach().cpu().numpy())
            forces_true.append(batch["forces"].detach().cpu().numpy())
            forces_pred.append(outputs["forces"].detach().cpu().numpy())
            q_true.append(batch["true_charges"].detach().cpu().numpy())
            positions.append(batch["positions"].detach().cpu().numpy())
            types.append(batch["types"].detach().cpu().numpy())
            if "latent_charges" in outputs:
                q_pred.append(outputs["latent_charges"].detach().cpu().numpy())

    result = {
        "checkpoint": str(Path(checkpoint_path)),
        "model_name": checkpoint["model_name"],
        "split": split,
        "energy_true": np.concatenate(energy_true, axis=0),
        "energy_pred": np.concatenate(energy_pred, axis=0),
        "forces_true": np.concatenate(forces_true, axis=0),
        "forces_pred": np.concatenate(forces_pred, axis=0),
        "true_charges": np.concatenate(q_true, axis=0),
        "positions": np.concatenate(positions, axis=0),
        "types": np.concatenate(types, axis=0),
        "metrics": {
            key: float(np.mean([entry[key] for entry in batch_metrics]))
            for key in batch_metrics[0]
        },
    }
    if q_pred:
        pred_charges = np.concatenate(q_pred, axis=0)
        result["pred_charges_raw"] = pred_charges
        result["pred_charges"] = _align_charge_sign(pred_charges, result["true_charges"])
        result["metrics"].update(charge_metrics(torch.from_numpy(result["pred_charges"]), torch.from_numpy(result["true_charges"])))
    return result


def save_evaluation_plots(prediction: Dict[str, np.ndarray], output_dir: str | Path) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = prediction["model_name"]
    saved = {}
    energy_path = output_dir / f"{model_name}_energy_parity.png"
    force_path = output_dir / f"{model_name}_force_parity.png"
    plot_energy_parity(prediction["energy_true"], prediction["energy_pred"], energy_path, title=f"{model_name}: energy parity")
    plot_force_parity(prediction["forces_true"], prediction["forces_pred"], force_path, title=f"{model_name}: force parity")
    saved["energy_parity"] = str(energy_path)
    saved["force_parity"] = str(force_path)

    if "pred_charges" in prediction:
        charge_path = output_dir / f"{model_name}_charge_scatter.png"
        plot_charge_scatter(prediction["true_charges"], prediction["pred_charges"], charge_path, title=f"{model_name}: latent charge vs true charge")
        saved["charge_scatter"] = str(charge_path)

        sample_index = int(np.argmax(np.abs(prediction["energy_pred"] - prediction["energy_true"])))
        snapshot_path = output_dir / f"{model_name}_configuration_snapshot.png"
        plot_configuration_snapshot(
            prediction["positions"][sample_index],
            prediction["true_charges"][sample_index],
            prediction["pred_charges"][sample_index],
            snapshot_path,
            title=f"{model_name}: sample {sample_index}",
        )
        saved["configuration_snapshot"] = str(snapshot_path)
    return saved


def save_comparison_plots(predictions: list[Dict[str, np.ndarray]], output_dir: str | Path) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    energy_rmse = {prediction["model_name"]: prediction["metrics"]["energy_rmse"] for prediction in predictions}
    force_rmse = {prediction["model_name"]: prediction["metrics"]["force_rmse"] for prediction in predictions}

    energy_bar = output_dir / "ablation_energy_rmse.png"
    force_bar = output_dir / "ablation_force_rmse.png"
    plot_ablation_bar(energy_rmse, energy_bar, title="Ablation: energy RMSE", ylabel="Energy RMSE")
    plot_ablation_bar(force_rmse, force_bar, title="Ablation: force RMSE", ylabel="Force RMSE")
    saved["ablation_energy_rmse"] = str(energy_bar)
    saved["ablation_force_rmse"] = str(force_bar)
    return saved


def plot_learning_curve_from_summary(summary_path: str | Path, output_dir: str | Path) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(summary_path).read_text())
    curve = {}
    for model_name, results in summary["results"].items():
        curve[model_name] = {
            int(train_size): float(entry["test_metrics"]["force_rmse"])
            for train_size, entry in results.items()
        }

    output_path = output_dir / "learning_curve_force_rmse.png"
    plot_learning_curve(curve, output_path, title="Learning curve: force RMSE", ylabel="Force RMSE")
    return str(output_path)
