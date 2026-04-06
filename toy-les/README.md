# toy-les

Minimal research code for understanding the `short-range + long-range + latent charge`
idea behind CACE + Latent Ewald Summation, without trying to reproduce the full
paper stack.

## Goal

This repository is meant to answer a narrow question:

How little code do we need in order to study the LES data flow

`U(R) = U_short(R) + U_long(q(R), R)`

with training supervision only from:

- total energy
- forces

while keeping the latent charge `q(R)` hidden during training and available only
for post-hoc evaluation.

## Toy System

The default dataset is a 3D analytic particle system:

- each sample contains a fixed number of particles in a cubic box
- each particle has a binary type: negative (`0`) or positive (`1`)
- true charges are geometry-dependent and then neutralized to zero total charge
- the label energy is split into:
  - a smooth short-range repulsion
  - a soft-Coulomb long-range interaction built from the true charges
- forces are obtained by differentiating the total energy, so they stay
  consistent with the geometry-dependent charge definition

This keeps the LES structure while avoiding DFT, Ewald implementation details,
and large data dependencies.

## Repository Layout

- `configs/base.yaml`: default dataset, model, and training settings
- `src/toy_les/physics.py`: analytic toy physics and true-charge definition
- `src/toy_les/data_gen.py`: random configuration sampling and dataset export
- `src/toy_les/dataset.py`: `.npz` dataset reader
- `src/toy_les/model.py`: minimal `SR-only` and `SR+LR(latent charge)` models
- `src/toy_les/train.py`: training loop and ablation runner
- `scripts/generate_data.py`: dataset generation entry point
- `scripts/train.py`: single-model training entry point
- `scripts/run_ablation.py`: `SR-only` vs `SR+LR` comparison entry point

## Install

Create an environment and install the minimal dependencies:

```bash
pip install -r toy-les/requirements.txt
```

## Quick Start

Generate a smoke-test dataset:

```bash
python toy-les/scripts/generate_data.py --preset smoke
```

Train the LES-style model on that smoke dataset:

```bash
python toy-les/scripts/train.py \
  --dataset toy-les/data/processed/toy_les_smoke.npz \
  --model sr_lr \
  --epochs 5
```

Run the `SR-only` vs `SR+LR` ablation:

```bash
python toy-les/scripts/run_ablation.py \
  --dataset toy-les/data/processed/toy_les_smoke.npz \
  --epochs 5
```

Generate a larger default dataset:

```bash
python toy-les/scripts/generate_data.py
```

Then train with the default config:

```bash
python toy-les/scripts/train.py --model sr_lr
```

## What the Models Do

### `sr`

This ablation only uses:

- local geometry features
- a short-range energy head

It has no latent charge branch and no explicit long-range physics.

### `sr_lr`

This LES-style minimal model uses:

- the same local geometry encoder
- a short-range energy head
- a latent charge head
- an explicit soft-Coulomb long-range module
- forces from autograd on the total energy

This is the smallest model in the repository that preserves the core LES
structure.

## Outputs

Training writes results under `toy-les/outputs/runs/`:

- `<run_name>/best.pt`: best checkpoint by validation loss
- `<run_name>/metrics.json`: full epoch history and final metrics

The ablation script also writes:

- `toy-les/outputs/runs/ablation_seed<seed>.json`

## Current Scope and Limitations

- fixed-size particle systems only
- no periodic long-range summation yet
- no message passing yet
- no explicit plotting/evaluation script yet
- true charge is saved only for evaluation, not used as a loss target

That is intentional: this repository is trying to make the LES architecture easy
to inspect before it is made physically richer.

## License

This repository is released under the MIT License. If you want a different
license choice, it is easy to swap.
