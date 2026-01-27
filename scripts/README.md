# Scripts <!-- omit from toc -->

CLI entry points for the project. Each script owns one task's command-line
parsing and orchestration, while reusable data, model, and training logic stays
in `src/cenapad_mlpt/`.

Configuration choices belong in YAML files under `configs/`, and cluster
execution belongs in PBS scripts under `jobs/`. This keeps the command-line
interface, experiment settings, job submission, and scientific or engineering
code clearly separated.

- [1. Entry Points](#1-entry-points)
  - [1.1. Data Download](#11-data-download)
  - [1.2. Training](#12-training)
- [2. Project Philosophy](#2-project-philosophy)

## 1. Entry Points

### 1.1. Data Download

[`data.py`](data.py) downloads the MNIST training and test sets.

```bash
uv run scripts/data.py --data-root data
```

`--data-root` accepts either an absolute path or a path relative to the
directory where the command is run.

### 1.2. Training

[`train.py`](train.py) loads a YAML configuration, prepares the data and model,
starts training, and saves the configuration, logs, and checkpoints under the
requested output path.

```bash
uv run scripts/train.py \
		--config configs/toy.yaml \
		--output-path outputs/toy
```

The output path may also be absolute, which is useful for cluster scratch
storage.

## 2. Project Philosophy

Scripts should remain thin, task-specific entry points. When a new workflow is
needed, add a CLI script with its own arguments and orchestration rather than
coupling scripts to one another.

Use the project areas according to their responsibilities:

- `scripts/`: user-facing CLI entry points.
- `configs/`: YAML experiment and runtime configuration.
- `jobs/`: OpenPBS submission templates and cluster-specific setup.
- `src/cenapad_mlpt/`: reusable data, preprocessing, model, and training logic.

This separation makes local commands and cluster jobs use the same underlying
implementation while allowing each interface to evolve independently.
