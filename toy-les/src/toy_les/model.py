from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    n_species: int = 2
    n_rbf: int = 16
    hidden_dim: int = 64
    encoder_hidden_dims: tuple[int, ...] = (64, 64)
    head_hidden_dims: tuple[int, ...] = (64, 32)
    short_range_cutoff: float = 2.5
    long_range_strength: float = 1.0
    long_range_softening: float = 0.35
    neutralize_charges: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _build_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.SiLU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def _pairwise_distances(positions: torch.Tensor) -> torch.Tensor:
    displacement = positions[:, :, None, :] - positions[:, None, :, :]
    return torch.linalg.norm(displacement, dim=-1)


def _upper_triangle_mask(n_particles: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(n_particles, n_particles, dtype=torch.bool, device=device), diagonal=1)


def _smooth_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    x = distances / cutoff
    values = 1.0 - 6.0 * x**5 + 15.0 * x**4 - 10.0 * x**3
    return torch.where(distances < cutoff, values, torch.zeros_like(distances))


class PairwiseRBF(nn.Module):
    """Gaussian radial basis expansion for pair distances."""

    def __init__(self, n_rbf: int, cutoff: float) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_rbf)
        widths = torch.full((n_rbf,), cutoff / max(n_rbf - 1, 1))
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-(diff**2) / (self.widths**2 + 1e-12))


class LocalFeatureEncoder(nn.Module):
    """
    Minimal local encoder:
    - expand pair distances with RBFs
    - aggregate neighbor features by neighbor species
    - concatenate center species one-hot
    - project to a learned hidden representation per particle
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.rbf = PairwiseRBF(n_rbf=config.n_rbf, cutoff=config.short_range_cutoff)
        input_dim = config.n_species + config.n_species * config.n_rbf
        self.encoder = _build_mlp(input_dim, config.encoder_hidden_dims, config.hidden_dim)

    def forward(self, positions: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 3:
            raise ValueError(f"positions must have shape [B, N, 3], got {positions.shape}")
        if types.ndim != 2:
            raise ValueError(f"types must have shape [B, N], got {types.shape}")

        distances = _pairwise_distances(positions)
        rbf = self.rbf(distances)

        eye = torch.eye(distances.shape[-1], dtype=torch.bool, device=distances.device).unsqueeze(0).unsqueeze(-1)
        cutoff = _smooth_cutoff(distances, self.config.short_range_cutoff).unsqueeze(-1)
        rbf = rbf * cutoff
        rbf = rbf.masked_fill(eye, 0.0)

        neighbor_species = F.one_hot(types, num_classes=self.config.n_species).to(positions.dtype)
        center_species = neighbor_species

        env = torch.einsum("bijr,bjs->birs", rbf, neighbor_species)
        env = env.reshape(positions.shape[0], positions.shape[1], -1)
        features = torch.cat([center_species, env], dim=-1)
        return self.encoder(features)


class ShortRangeHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.atom_mlp = _build_mlp(config.hidden_dim, config.head_hidden_dims, 1)

    def forward(self, atom_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        per_atom_energy = self.atom_mlp(atom_features).squeeze(-1)
        total_energy = per_atom_energy.sum(dim=-1)
        return {
            "short_per_atom_energy": per_atom_energy,
            "energy_short": total_energy,
        }


class LatentChargeHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.neutralize_charges = config.neutralize_charges
        self.atom_mlp = _build_mlp(config.hidden_dim, config.head_hidden_dims, 1)

    def forward(self, atom_features: torch.Tensor) -> torch.Tensor:
        q_latent = self.atom_mlp(atom_features).squeeze(-1)
        if self.neutralize_charges:
            q_latent = q_latent - q_latent.mean(dim=-1, keepdim=True)
        return q_latent


class SoftCoulombLongRange(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.long_range_strength = config.long_range_strength
        self.softening = config.long_range_softening

    def forward(self, positions: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        distances = _pairwise_distances(positions)
        kernel = torch.rsqrt(distances**2 + self.softening**2)
        pair_charge = charges[:, :, None] * charges[:, None, :]
        pair_energy = self.long_range_strength * pair_charge * kernel
        mask = _upper_triangle_mask(distances.shape[-1], distances.device)
        return pair_energy[:, mask].sum(dim=-1)


class ToyLESModel(nn.Module):
    """Minimal LES-style model with explicit SR and LR branches."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = LocalFeatureEncoder(config)
        self.short_head = ShortRangeHead(config)
        self.charge_head = LatentChargeHead(config)
        self.long_range = SoftCoulombLongRange(config)

    def forward(
        self,
        positions: torch.Tensor,
        types: torch.Tensor,
        compute_forces: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
            types = types.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        work_positions = positions.clone().requires_grad_(compute_forces)
        atom_features = self.encoder(work_positions, types)

        short_outputs = self.short_head(atom_features)
        latent_charges = self.charge_head(atom_features)
        energy_long = self.long_range(work_positions, latent_charges)
        energy_total = short_outputs["energy_short"] + energy_long

        outputs: Dict[str, torch.Tensor] = {
            "atom_features": atom_features,
            "latent_charges": latent_charges,
            "energy_short": short_outputs["energy_short"],
            "energy_long": energy_long,
            "energy_total": energy_total,
            "short_per_atom_energy": short_outputs["short_per_atom_energy"],
        }

        if compute_forces:
            forces = -torch.autograd.grad(
                energy_total.sum(),
                work_positions,
                create_graph=self.training,
            )[0]
            outputs["forces"] = forces

        if squeezed:
            squeezed_outputs: Dict[str, torch.Tensor] = {}
            for key, value in outputs.items():
                squeezed_outputs[key] = value.squeeze(0)
            return squeezed_outputs
        return outputs


class ShortRangeOnlyModel(nn.Module):
    """Ablation baseline that removes the latent-charge long-range branch."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = LocalFeatureEncoder(config)
        self.short_head = ShortRangeHead(config)

    def forward(
        self,
        positions: torch.Tensor,
        types: torch.Tensor,
        compute_forces: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
            types = types.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        work_positions = positions.clone().requires_grad_(compute_forces)
        atom_features = self.encoder(work_positions, types)
        short_outputs = self.short_head(atom_features)
        energy_total = short_outputs["energy_short"]

        outputs: Dict[str, torch.Tensor] = {
            "atom_features": atom_features,
            "energy_short": short_outputs["energy_short"],
            "energy_long": torch.zeros_like(energy_total),
            "energy_total": energy_total,
            "short_per_atom_energy": short_outputs["short_per_atom_energy"],
        }

        if compute_forces:
            forces = -torch.autograd.grad(
                energy_total.sum(),
                work_positions,
                create_graph=self.training,
            )[0]
            outputs["forces"] = forces

        if squeezed:
            squeezed_outputs: Dict[str, torch.Tensor] = {}
            for key, value in outputs.items():
                squeezed_outputs[key] = value.squeeze(0)
            return squeezed_outputs
        return outputs
