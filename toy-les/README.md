# toy-les

Minimal research code for understanding the `short-range + long-range + latent charge`
idea behind CACE + Latent Ewald Summation, without trying to reproduce the full
paper stack.

## Scope

This project uses an analytic toy system:

- particles live in a 3D box
- each particle has a binary type: negative (`0`) or positive (`1`)
- the total energy is

  `U(R) = U_short(R) + U_long(q(R), R)`

- `q(R)` is a hidden, geometry-dependent charge used to generate labels
- training is intended to use only total energy and forces
- true charges are saved for post-hoc evaluation, not for training

## Current Step

The first implementation pass focuses on:

1. a stable analytic toy physics definition
2. deterministic dataset generation
3. a clean project skeleton that we can extend with a minimal LES-style model

## Layout

- `configs/base.yaml`: default experiment and dataset settings
- `src/toy_les/physics.py`: analytic short-range, long-range, and true-charge definitions
- `src/toy_les/data_gen.py`: random configuration sampling and dataset export
- `scripts/generate_data.py`: command-line entry point for generating an `.npz` dataset

## Quick Start

Generate a small smoke-test dataset:

```bash
python toy-les/scripts/generate_data.py --preset smoke
```

Generate with the default config:

```bash
python toy-les/scripts/generate_data.py
```
