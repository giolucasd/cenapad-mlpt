# Jobs (OpenPBS) <!-- omit from toc -->

This directory contains the OpenPBS job scripts used to run training on the
cluster. The project uses the native Python environment managed by `uv`; no
container runtime is required.

- [1. Job Template](#1-job-template)
- [2. Prepare the Lovelace Environment](#2-prepare-the-lovelace-environment)
- [3. Submit a Job](#3-submit-a-job)
- [4. Outputs and Logs](#4-outputs-and-logs)
- [5. Monitor a Job](#5-monitor-a-job)

## 1. Job Template

[`train.pbs`](train.pbs) runs the training entry point with the toy
configuration:

```bash
uv run --offline scripts/train.py \
    --config configs/toy.yaml \
    --output-path /work/username/outputs/toy-cnn
```

The PBS directives at the top of the script define the job name, queue,
walltime, and scheduler log files. Adjust them for the experiment and cluster
resources before submitting.

## 2. Prepare the Lovelace Environment

Install `uv` in the Lovelace environment if it is not already available:

```sh
cd ~
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Before submitting a job, install the project environment and dependencies from
the repository:

```bash
uv venv
uv sync
```

Jobs do not have internet access, so dependency installation must be completed
before submission.

## 3. Submit a Job

Create a personal, untracked copy of the template and replace user-specific
values such as the username, output paths, queue, and resource limits:

```bash
mkdir -p jobs/untracked
cp jobs/train.pbs jobs/untracked/train.pbs
```

Edit and submit the copy from the repository root:

```bash
qsub jobs/untracked/train.pbs
```

The job changes to `$PBS_O_WORKDIR`, so relative paths in the command are
resolved from the project directory.

## 4. Outputs and Logs

Use the cluster scratch area for all job execution and intermediate outputs,
for example `/work/username/`. The cluster policy requires copying results to
your home directory only after the job finishes. Do not train directly in your
home directory.

Training artifacts are written to the path passed with `--output-path`. Update
the template's `/work/username/outputs/toy-cnn` paths to use your own scratch
directory. The template copies completed results to the repository's
`outputs/toy-cnn` directory; adapt this destination if your personal workflow
uses a home-directory path after completion.

PBS standard output and error logs are configured with `#PBS -o` and `#PBS -e`
at the top of the job script.

## 5. Monitor a Job

List the current user's jobs with:

```bash
qstat -u "$USER"
```

After completion, inspect the configured PBS logs and the experiment output
directory for training results and checkpoints.
