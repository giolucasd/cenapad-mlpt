# CENAPAD Machine Learning Project Template <!-- omit from toc -->

`cenapad-mlpt` is a small PyTorch project template for machine learning experiments on CENAPAD-like HPC clusters. It provides a clean starting point for projects that need local development, GPU training, configuration files, and OpenPBS job submission.

The template uses [`uv`](https://docs.astral.sh/uv/) as its only Python
environment and dependency-management tool. This improves reproducibility by keeping dependencies in `pyproject.toml` and locking resolved versions in `uv.lock`, while the cluster's CUDA, drivers, and hardware remain part of the execution environment.

- [1. Project Structure](#1-project-structure)
- [2. Local Setup](#2-local-setup)
- [3. Download the Data](#3-download-the-data)
- [4. Configure and Run Training](#4-configure-and-run-training)
- [5. Run on an OpenPBS Cluster](#5-run-on-an-openpbs-cluster)
- [6. Reproducibility](#6-reproducibility)
- [7. Starting a New Project](#7-starting-a-new-project)

## 1. Project Structure

```text
configs/              YAML experiment configurations
data/                 Local dataset storage, not versioned
jobs/                 OpenPBS templates and cluster instructions
outputs/              Training artifacts, not versioned
scripts/              Command-line entry points
src/cenapad_mlpt/     Reusable data, model, and training code
pyproject.toml        Project metadata and dependencies
uv.lock               Locked dependency resolution
```

The project keeps responsibilities separate:

- `scripts/` contains thin CLI entry points.
- `configs/` contains experiment choices.
- `jobs/` contains cluster-specific submission setup.
- `src/cenapad_mlpt/` contains reusable implementation.

## 2. Local Setup

Install `uv` on your user account, then clone the repository and install the locked environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repository-url>
cd cenapad-mlpt
uv venv
uv sync
```

`uv sync` creates or updates the project virtual environment from `pyproject.toml` and `uv.lock`.

`uv` installs in user space and does not modify the Python installations provided by the cluster. It can manage any Python version supported by the project, without depending on the cluster's Python support. There is no significant performance drop compared with the cluster's native setup.

## 3. Download the Data

Download MNIST to the default `data/` directory:

```bash
uv run scripts/data.py
```

Choose another location with `--data-root`. Both absolute and relative paths are accepted; relative paths are resolved from the directory where the command is run:

```bash
uv run scripts/data.py --data-root /path/to/data
uv run scripts/data.py --data-root data
```

## 4. Configure and Run Training

Training is configured through YAML. The example configuration is [`configs/toy.yaml`](configs/toy.yaml); it defines the seed, dataset location, preprocessing resolution, model dimensions, optimizer settings, and training limits.

Run the example locally with:

```bash
uv run scripts/train.py \
    --config configs/toy.yaml \
    --output-path outputs/toy
```

The output directory contains the copied configuration, TensorBoard logs, and model checkpoints. `--output-path` accepts either an absolute path or a path relative to the directory where training is started.

## 5. Run on an OpenPBS Cluster

Complete the setup in section 2 on Lovelace while internet access is available. Jobs do not have internet access, so the environment and all dependencies must be installed before the job is submitted. No container runtime is required.

Copy the tracked job template to an untracked personal file and update values such as the username, queue, resource limits, log paths, and output paths:

```bash
mkdir -p jobs/untracked
cp jobs/train.pbs jobs/untracked/train.pbs
vim jobs/untracked/train.pbs # edit the file
qsub jobs/untracked/train.pbs
```

Use the cluster scratch area, such as `/work/username/`, for job execution and intermediate results. Cluster policy requires copying completed results to the user's home directory only after the job finishes. Do not train directly in home storage.

## 6. Reproducibility

This template provides a practical baseline for reproducibility:

- Python dependencies are declared in `pyproject.toml`.
- Resolved dependency versions are recorded in `uv.lock`.
- Experiment parameters are stored in YAML.
- Training uses a configured random seed.
- Each output directory receives a copy of the configuration used for the run.
- Checkpoints and TensorBoard logs are stored with the run outputs.

This does not fully reproduce the surrounding execution environment. Results may still depend on the cluster's operating system, GPU model, NVIDIA driver, CUDA compatibility, filesystem, and scheduler configuration. Record the Git revision, PBS job ID, hardware, and relevant software versions for important experiments.

## 7. Starting a New Project

Fork or clone this repository, then adapt the configuration and source code to the new task. Keep the separation between CLI, configuration, cluster jobs, and reusable implementation as the project grows.

Add new task-specific scripts under `scripts/`, new experiment configurations under `configs/`, and personal PBS copies under `jobs/untracked/` and `configs/untracked/`. Keep datasets, checkpoints, logs, and other runtime artifacts out of version control.
