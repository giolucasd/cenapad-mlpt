# Configuration Guide <!-- omit from toc -->

This directory contains YAML configuration files for training the project models.
The configuration is grouped by responsibility so that data loading, model
construction, optimization, and training settings can be changed independently.

- [1. Example Configuration](#1-example-configuration)
- [2. Parameter Reference](#2-parameter-reference)
  - [2.1. Global Settings (`global`)](#21-global-settings-global)
  - [2.2. Execution Settings (`execution`)](#22-execution-settings-execution)
  - [2.3. Dataset Settings (`dataset`)](#23-dataset-settings-dataset)
  - [2.4. Pipeline Settings (`pipeline`)](#24-pipeline-settings-pipeline)
  - [2.5. Model Settings (`model`)](#25-model-settings-model)
  - [2.6. Optimizer Settings (`optimizer`)](#26-optimizer-settings-optimizer)
  - [2.7. Training Settings (`training`)](#27-training-settings-training)

## 1. Example Configuration

The [toy.yaml](toy.yaml) configuration trains the current `simple_cnn` model
on MNIST:

```yaml
global:
	seed: 27
	model_name: simple_cnn

execution:
	batch_size: 128
	num_workers: 4

dataset:
	data_root: data

pipeline:
	resolution: 28

model:
	in_channels: 1
	num_classes: 10

optimizer:
	lr: 0.001
	weight_decay: 0.0001

training:
	max_epochs: 3
	resume_from: null
```

Run training from the repository root with:

```bash
uv run scripts/train.py \
		--config configs/toy.yaml \
		--output-path outputs/toy
```

## 2. Parameter Reference

### 2.1. Global Settings (`global`)

- **`seed`**: Random seed used for reproducibility.
- **`model_name`**: Name of the model registered in the model registry.

### 2.2. Execution Settings (`execution`)

- **`batch_size`**: Number of samples processed in each batch.
- **`num_workers`**: Number of worker processes used by the data loaders.

### 2.3. Dataset Settings (`dataset`)

- **`data_root`**: Location of the dataset. Relative paths are interpreted from
	the directory where the training command is started.

### 2.4. Pipeline Settings (`pipeline`)

- **`resolution`**: Height and width used by the example image resize
	transformation. The value is applied as `(resolution, resolution)`.

This section is intended to hold preprocessing settings as the data pipeline
grows.

### 2.5. Model Settings (`model`)

- **`in_channels`**: Number of channels in each input image.
- **`num_classes`**: Number of output classes.

The active model is selected by `global.model_name`. The current example uses
`simple_cnn`.

### 2.6. Optimizer Settings (`optimizer`)

The project currently uses AdamW for every model.

- **`lr`**: Learning rate.
- **`weight_decay`**: AdamW weight decay coefficient.

### 2.7. Training Settings (`training`)

- **`max_epochs`**: Maximum number of training epochs.
- **`resume_from`**: Optional checkpoint path. Set to `null` to start a new run.
