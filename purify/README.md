# Algorithm 3 Purification

This folder contains the project-level inference pipeline for Algorithm 3:
inference-time dataset sanitization with the trained pixel-space CM purifier.

The purifier does not train a model and does not generate new poisons. It loads
the `.pth` checkpoint created by Algorithm 2, adds DDPM noise at a selected
`t_star`, runs one CM forward pass, and writes purified PNG images.

## Algorithm 3

Input:

```text
D_untrusted = poison images from dataset_generation/datasets/test
f_theta     = trained CM purifier loaded from .pth
t_star      = target DDPM noising timestep
```

For each minibatch:

```text
x_t_star = sqrt(alpha_bar_t_star) * x_untrusted
           + sqrt(1 - alpha_bar_t_star) * epsilon

x_hat = f_theta(x_t_star, t_star)
```

The output `x_hat` is saved as the sanitized image. This is one neural function
evaluation per image batch, not a multi-step diffusion chain.

This project version has no Canny map, no Stable Diffusion, and no LCM-LoRA.
Those parts belong to InstantPure and are not used here.

## Input Dataset

The held-out test dataset already contains `poisons/` folders:

```text
dataset_generation/datasets/test/WB_c0/poisons/
dataset_generation/datasets/test/BP_c0_g0/poisons/
```

Each case may also contain:

```text
clean/
target/
```

Purification consumes the existing `poisons/` images. It does not create or
modify the original dataset.

## Output Layout

Default output root:

```text
purify/outputs/test_purified/
```

Purified images are written with the same case and filename structure:

```text
purify/outputs/test_purified/<case>/poisons/<filename>.png
```

When `COPY_REFERENCE_DIRS=1`, which is the default, the runner also copies each
case's `clean/` and `target/` folders into the output tree for evaluation
context.

Each run writes:

```text
purify/outputs/test_purified/summary.json
```

The summary records the checkpoint path, input path, output path, resolved
`t_star`, batch size, seed, case counts, purified image count, elapsed time, and
throughput.

## Run Purification

On the university Slurm server, submit the runner instead of running long Python
jobs directly:

```bash
sbatch purify/run_purify_test.sh
```

Default runner settings:

```text
conda env:   purifying_poison
checkpoint:  consistency_model/checkpoints/cm_purifier.pth
input:       dataset_generation/datasets/test
output:      purify/outputs/test_purified
logs:        purify/logs/purify_test_<job_id>.log
GPU:         1
memory:      32GB
time:        48 hours
```

If the checkpoint is missing, the runner submits:

```text
consistency_model/run_cm_purifier_training.sh
```

Then it schedules purification with a Slurm `afterok` dependency so purification
starts only after training succeeds.

## Configuration

Common quick test:

```bash
MAX_IMAGES=32 LOG_STEPS=16 sbatch purify/run_purify_test.sh
```

Useful environment variables:

```text
CHECKPOINT_PATH       .pth checkpoint to load
TEST_DIR              test dataset root containing case folders
OUTPUT_DIR            purified output root
ENV_NAME              conda environment name
ROOT_REQUIREMENTS     requirements file used by the runner
T_STAR                DDPM timestep or fraction for inference noising
BATCH_SIZE            purification batch size
SEED                  seed for inference noise
LOG_STEPS             progress log interval in images
MAX_IMAGES            optional cap for a small run
COPY_REFERENCE_DIRS   1 to copy clean/target folders, 0 to skip
```

Examples:

```bash
T_STAR=150 BATCH_SIZE=128 sbatch purify/run_purify_test.sh
```

```bash
CHECKPOINT_PATH=consistency_model/checkpoints/cm_purifier_step50000.pth \
OUTPUT_DIR=purify/outputs/test_purified_step50000 \
sbatch purify/run_purify_test.sh
```

## Code Guide

Main files:

```text
dataset.py           discovers held-out cases and builds per-image records
purify_test.py       Algorithm 3 loop, logging, batching, saving, summary JSON
run_purify_test.sh   Slurm runner, env setup, checkpoint check, dependency submit
```

`purify_test.py` uses helpers from `consistency_model/cm_purifier`:

```text
load_purifier_from_checkpoint   rebuilds the model and loads EMA weights
load_image_tensor               loads RGB images into [-1, 1]
q_sample                        applies DDPM forward noising
save_image_tensor               saves purified tensors as PNG images
```

The Python stages logged during purification are:

```text
1. DISCOVERING TEST POISON CASES...
2. LOADING TRAINED CM PURIFIER...
3. PURIFYING POISON IMAGES...
4. COPYING CLEAN/TARGET REFERENCE DIRECTORIES...
5. WRITING SUMMARY...
```

Progress lines include processed image count, percent complete, images per
second, elapsed time, estimated remaining time, and CUDA memory usage.

## Checkpoint Behavior

The Slurm runner checks for:

```text
consistency_model/checkpoints/cm_purifier.pth
```

or the custom `CHECKPOINT_PATH` value. If the file exists, purification starts.
If the file is missing, the runner submits training first.

The Python module also checks the checkpoint path. If someone bypasses Slurm and
calls Python directly without a `.pth` file, it raises a clear `FileNotFoundError`
with instructions to run the training runner first.

## Troubleshooting

Missing `.pth`:

```text
Submit purify/run_purify_test.sh. It will submit training first and then
re-submit itself after training succeeds.
```

Missing CUDA:

```text
Check the Slurm GPU allocation and the CUDA section in purify/logs/.
The runner refuses CPU purification for this project workflow.
```

No `poisons/` folders:

```text
Confirm the input path is dataset_generation/datasets/test or a compatible
folder with <case>/poisons/ subdirectories.
```

Need a small sanity run:

```bash
MAX_IMAGES=8 LOG_STEPS=4 sbatch purify/run_purify_test.sh
```

Need only purified images, without copied references:

```bash
COPY_REFERENCE_DIRS=0 sbatch purify/run_purify_test.sh
```
