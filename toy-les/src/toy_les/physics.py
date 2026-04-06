from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class ToyPhysicsConfig:
    """Parameters for the analytic toy LES system."""

    q0: float = 1.0
    alpha_charge_transfer: float = 0.15
    charge_transfer_length: float = 1.2
    charge_cutoff: float = 2.0
    eps_sr_same: float = 2.0
    eps_sr_opposite: float = 1.2
    rho_sr_same: float = 0.7
    rho_sr_opposite: float = 0.85
    short_range_cutoff: float = 2.5
    k_lr: float = 1.0
    softening: float = 0.35

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> "ToyPhysicsConfig":
        return cls(**values)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _ensure_batch(positions: torch.Tensor, types: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if positions.ndim == 2:
        positions = positions.unsqueeze(0)
        types = types.unsqueeze(0)
        squeezed = True
    elif positions.ndim == 3:
        squeezed = False
    else:
        raise ValueError(f"positions must have shape [N, 3] or [B, N, 3], got {positions.shape}")
    return positions, types, squeezed


def _upper_triangle_mask(n_particles: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(n_particles, n_particles, dtype=torch.bool, device=device), diagonal=1)


def _pairwise_distances(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    displacements = positions[:, :, None, :] - positions[:, None, :, :]
    distances = torch.linalg.norm(displacements, dim=-1)
    return displacements, distances


def _smooth_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    x = distances / cutoff
    values = 1.0 - 6.0 * x**5 + 15.0 * x**4 - 10.0 * x**3
    return torch.where(distances < cutoff, values, torch.zeros_like(distances))


def _signed_types(types: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    positive = torch.ones_like(types, dtype=dtype)
    negative = -torch.ones_like(types, dtype=dtype)
    return torch.where(types > 0, positive, negative)


def compute_true_charges(
    positions: torch.Tensor,
    types: torch.Tensor,
    cfg: ToyPhysicsConfig,
) -> torch.Tensor:
    """
    Compute geometry-dependent true charges used only for generating labels.

    The induced part depends on the signed local environment, then we subtract the
    per-configuration mean to enforce exact charge neutrality.
    """

    positions, types, squeezed = _ensure_batch(positions, types)
    _, distances = _pairwise_distances(positions)
    sign = _signed_types(types, dtype=positions.dtype)
    eye = torch.eye(distances.shape[-1], dtype=torch.bool, device=distances.device).unsqueeze(0)

    env_weight = torch.exp(-(distances / cfg.charge_transfer_length) ** 2) * _smooth_cutoff(distances, cfg.charge_cutoff)
    env_weight = env_weight.masked_fill(eye, 0.0)

    induced = cfg.alpha_charge_transfer * torch.einsum("bij,bj->bi", env_weight, sign)
    q_raw = cfg.q0 * sign + induced
    q_true = q_raw - q_raw.mean(dim=-1, keepdim=True)

    if squeezed:
        return q_true.squeeze(0)
    return q_true


def short_range_energy(
    positions: torch.Tensor,
    types: torch.Tensor,
    cfg: ToyPhysicsConfig,
) -> torch.Tensor:
    """Smooth finite short-range repulsion."""

    positions, types, squeezed = _ensure_batch(positions, types)
    _, distances = _pairwise_distances(positions)
    same_type = types[:, :, None] == types[:, None, :]

    eps = torch.where(
        same_type,
        torch.full_like(distances, cfg.eps_sr_same),
        torch.full_like(distances, cfg.eps_sr_opposite),
    )
    rho = torch.where(
        same_type,
        torch.full_like(distances, cfg.rho_sr_same),
        torch.full_like(distances, cfg.rho_sr_opposite),
    )

    pair_energy = eps * torch.exp(-((distances / rho) ** 2)) * _smooth_cutoff(distances, cfg.short_range_cutoff)
    mask = _upper_triangle_mask(distances.shape[-1], distances.device)
    total = pair_energy[:, mask].sum(dim=-1)

    if squeezed:
        return total.squeeze(0)
    return total


def long_range_energy(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cfg: ToyPhysicsConfig,
) -> torch.Tensor:
    """Soft-Coulomb long-range energy with exact pair summation."""

    if positions.ndim == 2:
        positions = positions.unsqueeze(0)
        charges = charges.unsqueeze(0)
        squeezed = True
    elif positions.ndim == 3:
        squeezed = False
    else:
        raise ValueError(f"positions must have shape [N, 3] or [B, N, 3], got {positions.shape}")

    _, distances = _pairwise_distances(positions)
    kernel = torch.rsqrt(distances**2 + cfg.softening**2)
    pair_charge = charges[:, :, None] * charges[:, None, :]
    pair_energy = cfg.k_lr * pair_charge * kernel
    mask = _upper_triangle_mask(distances.shape[-1], distances.device)
    total = pair_energy[:, mask].sum(dim=-1)

    if squeezed:
        return total.squeeze(0)
    return total


def compute_energy_components(
    positions: torch.Tensor,
    types: torch.Tensor,
    cfg: ToyPhysicsConfig,
) -> Dict[str, torch.Tensor]:
    charges = compute_true_charges(positions, types, cfg)
    e_short = short_range_energy(positions, types, cfg)
    e_long = long_range_energy(positions, charges, cfg)
    e_total = e_short + e_long
    return {
        "true_charges": charges,
        "energy_short": e_short,
        "energy_long": e_long,
        "energy_total": e_total,
    }


def compute_energy_and_forces(
    positions: torch.Tensor,
    types: torch.Tensor,
    cfg: ToyPhysicsConfig,
) -> Dict[str, torch.Tensor]:
    """
    Return energy decomposition, true charges, and forces.

    Forces are obtained by differentiating the total energy, so they remain
    consistent with the geometry-dependent charge definition.
    """

    work_positions = positions.clone().detach().requires_grad_(True)
    components = compute_energy_components(work_positions, types, cfg)
    total_energy = components["energy_total"]
    grad = torch.autograd.grad(total_energy.sum(), work_positions, create_graph=False)[0]
    components["forces"] = -grad
    components["positions"] = work_positions.detach()
    components["types"] = types.detach()
    return components
