# HSPCT-DD

Implementation of **Hierarchical Shift-Aware Prototype Conditional Transport for Cross-Domain Fine-Grained Remote Sensing Image Classification**.

 All six transfer directions use the same implementation and the uniform paper hyperparameters. Separate comparison experiments, ablations, plotting code, and historical notebooks are not included.

## Contents

```text
HSPCT-DD.ipynb              Main method; one training cell, default L -> M
train_teacher.py            Required adversarial teacher; default L -> M
requirements.txt           Direct Python dependencies
dataset/
  images.tar.gz            Image archive (distributed separately from Git)
  SHA256SUMS.txt            SHA-256 checksum of the image archive
  split_counts.json        Number of samples in each split
  tasks/
    protocol.json          Original split and quality-control protocol
    high_vs_medium/        Shared hierarchy and two directed tasks
    high_vs_low/           Shared hierarchy and two directed tasks
    medium_vs_low/         Shared hierarchy and two directed tasks
checkpoints/<task>/last_model.pt   Generated adversarial teacher
results/<task>/seed_2026/          Created during training
```


## Installation


| Component | Version / hardware |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| NumPy | 2.2.6 |
| Pillow | 11.2.1 |
| JupyterLab | 4.4.2 |
| CUDA reported by PyTorch | 12.8 |
| GPU | NVIDIA GeForce RTX 5090 |


## Dataset preparation

## Prepare the teacher

HSPCT-DD requires a **pretrained, frozen adversarial teacher for the selected direction**. Train it first from the repository root:

```bash
python train_teacher.py
```

This defaults to Low-to-Medium, with 20 source-pretraining epochs, 20 adaptation epochs, 1000 steps per epoch, and the original domain-adversarial weight of 0.10. It automatically saves the final teacher to the path below.

```text
checkpoints/low_to_medium/last_model.pt
```

The main method reads this file directly; no manual copy is required. Both entry points use the same hierarchy. The notebook verifies `config.fine_names` when present and loads the compatible `model_state_dict`. Load only trusted checkpoints.

For another direction, change only the task argument:

```bash
python train_teacher.py --task medium_to_high
```

Alternatively, change the default `TASK_NAME` in `train_teacher.py`. Runtime options include `--data-root`, `--result-root`, and `--num-workers`. Keep the default output root for automatic handoff to the main method. Try `--num-workers 0` if data loading fails on your platform.

## Run the main method

Start JupyterLab **from the repository root**, then open the notebook:

```bash
python -m jupyterlab HSPCT-DD.ipynb
```

The default configuration is:

```python
DATA_ROOT = Path('dataset').resolve()
TASK_NAME = 'low_to_medium'
```

After preparing the data and the task-specific teacher, run the training cell. The notebook checks for the teacher before constructing datasets or downloading a backbone. Windows uses zero data-loader workers for notebook compatibility; other systems default to eight. Worker count is a runtime setting, not an algorithm hyperparameter.

### Switch direction

Only change `TASK_NAME`; dataset lists, hierarchy, teacher path, and output path are selected automatically:

| Direction | `TASK_NAME` |
| --- | --- |
| L to M (default) | `low_to_medium` |
| M to L | `medium_to_low` |
| H to M | `high_to_medium` |
| M to H | `medium_to_high` |
| L to H | `low_to_high` |
| H to L | `high_to_low` |

First train the teacher with the matching `--task`, then set the notebook's `TASK_NAME` to that direction. Each task has its own teacher under `checkpoints/<TASK_NAME>/last_model.pt`. No loss-weight or training-budget changes are needed. Restart the kernel between tasks. Repeating a task with the same seed writes to the same output directory; preserve earlier results separately if needed.

## Paper configuration

| Setting | Value |
| --- | --- |
| Random seed | 2026 |
| Backbone | ImageNet-pretrained ResNet-50 |
| Input | Aspect-ratio-preserving resize and padding to 224 x 224 |
| Source / target / evaluation batch sizes | 64 / 96 / 96 |
| Source pretraining / adaptation epochs | 20 / 20 |
| Optimization steps per epoch | **1000** in every direction |
| Coarse / fine transport weights | **0.001 / 0.01** |
| EMA-teacher momentum | **0.997** |
| Backbone / head learning rates | 0.0001 / 0.001 |
| Optimizer / weight decay | AdamW / 0.0001 |
| Final learning-rate ratio | 0.30 |
| Transport temperature | 0.10 |
| Hierarchy-consistency / distillation weights | 0.02 / 0.05 |
| Distillation temperature | 2.0 |
| Alignment ramp | 8 epochs |
| Coarse / fine confidence thresholds | 0.40 / 0.15 |
| Reliability lower bound | 0.05 |
| Prior epoch EMA | 0.80 |

These uniform settings replace the direction-specific values in the archived notebook. Training losses and network definitions are otherwise preserved; path handling and release documentation have been adapted for portability.

## Results

Outputs are saved in `results/<TASK_NAME>/seed_2026/`. For the paper's final-epoch reporting protocol, use `last_model.pt` and the top-level `target_test` fields of `metrics.json`. Fine and coarse macro accuracy are mean per-class recall, not overall accuracy or macro-F1. `train_log.csv` records epoch-wise metrics.

The inherited implementation also saves auxiliary `best_model.pt` and `best_target_test` monitoring outputs. These are not the final-epoch values used for reporting. The adversarial teacher input in this release explicitly points to `last_model.pt`.

## Data attribution and licensing

The classification crops are derived from FAIR1M 2.0. Redistribution and use of the underlying data remain subject to its original terms; a code license does not grant data rights. Confirm redistribution permission before making the image archive public. No additional code license has been assigned in this preparation step. Add the author-approved license and the verified paper citation/identifier before public release.
