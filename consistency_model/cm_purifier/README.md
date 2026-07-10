# Pixel-Space CM Purifier

This package implements Algorithm 2 from the project README: poison-aware
Consistency Model training for targeted clean-label poison purification.

The implementation is a new pixel-space purifier. InstantPure is used only as a
reference for the research idea and coding style. This package does not import
from `consistency_model/InstantPure`, does not use LCM-LoRA, does not use Stable
Diffusion 1.5, and does not use Canny maps. The model is trained with normal
DDPM schedules on CIFAR-sized images.

## Algorithm 2

Training learns a purifier `f_theta` that maps a noised image back to the clean
image in one neural function evaluation.

Input dataset:

```text
dataset_generation/datasets/train/
  clean/
  poisons/
```

The two folders must contain matching filenames. Each pair gives:

```text
x_clean  = clean image
x_poison = poison or identity image
y        = clean class label parsed from the filename
a        = attack type parsed from the filename: clean, wb, or bp
```

The core poison-aware corruption is:

```text
delta = x_poison - x_clean
x_t_star = sqrt(alpha_bar_t) * x_clean
           + sqrt(1 - alpha_bar_t) * (epsilon + gamma_a * delta)
```

This is implemented in `train_one_step` in `train.py`:

```text
1. Load a clean/poison minibatch.
2. Compute delta = poison - clean.
3. Build gamma_a from attack type: clean, wb, or bp.
4. Sample epsilon and a DDIM training timestep.
5. Create x_t_star with the DDPM forward process.
6. Predict x_hat_0 = f_theta(x_t_star, t).
7. Use the DDIM solver to build a shallower teacher input.
8. Compute distillation, reconstruction, identity, and optional classifier loss.
9. Update the student model.
10. Update the EMA teacher.
```

The default loss is:

```text
loss = lambda_distill * loss_distill
     + lambda_rec     * loss_reconstruction
     + lambda_id      * loss_identity
     + lambda_cls     * loss_classifier
```

`lambda_cls` defaults to `0.0`, so no classifier is required unless the user
explicitly enables classifier preservation loss.

## Run Training

On the university Slurm server, submit the runner instead of running long Python
jobs directly:

```bash
sbatch consistency_model/run_cm_purifier_training.sh
```

Default runner settings:

```text
conda env:   purifying_poison
pair dir:    dataset_generation/datasets/train
checkpoint:  consistency_model/checkpoints/cm_purifier.pth
logs:        consistency_model/logs/cm_purifier_train_<job_id>.log
GPU:         1
memory:      64GB
time:        48 hours
```

The runner creates or repairs the conda environment, installs CUDA PyTorch, and
then installs the root `requirements.txt` without replacing the CUDA Torch
build.

Common Slurm overrides:

```bash
MAX_STEPS=1000 BATCH_SIZE=64 LOG_STEPS=10 \
sbatch consistency_model/run_cm_purifier_training.sh
```

Useful environment variables:

```text
PAIR_DIR        input pair dataset root
OUTPUT_PATH     checkpoint path to write
TEACHER_MODEL   diffusers DDPM model id or local path
ENV_NAME        conda environment name
CM_OUTPUT_MODE  CM output parameterization, default full_boundary
MAX_STEPS       total optimization steps
BATCH_SIZE      training batch size
NUM_WORKERS     dataloader workers
SAVE_STEPS      checkpoint save interval
LOG_STEPS       metric log interval
LEARNING_RATE   AdamW learning rate
EMA_DECAY       EMA teacher update decay
GAMMA_WB        poison residual strength for Witches' Brew pairs
GAMMA_BP        poison residual strength for Bullseye Polytope pairs
GAMMA_CLEAN     poison residual strength for clean identity pairs
```

For reference only, the Python module called by the runner is:

```bash
python -m consistency_model.cm_purifier.train \
  --pair-dir dataset_generation/datasets/train \
  --teacher-model google/ddpm-cifar10-32 \
  --backbone diffusers \
  --schedule-source diffusers \
  --out consistency_model/checkpoints/cm_purifier.pth \
  --device cuda
```

Use the Slurm runner for real training.

## Checkpoints

The output is a PyTorch `.pth` checkpoint. The file stores everything needed for
inference or resume:

```text
format        checkpoint format marker
model         student model weights
ema           EMA teacher weights, used by default for inference
optimizer     optimizer state for resume
global_step   saved training step
args          training arguments and model configuration
betas         DDPM beta schedule used during training
```

The `.pth` suffix is used because this is not just a raw state dict. It is a
training and inference bundle with weights, metadata, optimizer state, and the
schedule needed to reconstruct the purifier.

Inference loaders use `load_purifier_from_checkpoint` in `checkpoint.py`.
By default, inference loads the EMA weights. Passing `--use-student` to the
inference modules loads the student weights instead.

## Code Guide

Main files:

```text
dataset.py      loads matched clean/poison pairs and parses labels/attack types
schedules.py    builds DDPM beta, alpha, and sigma schedules
solver.py       implements the small deterministic DDIM step for teacher targets
model.py        wraps a diffusers or tiny denoiser as a clean-image CM predictor
losses.py       combines distillation, reconstruction, identity, and classifier losses
ema.py          creates and updates the EMA teacher model
checkpoint.py   saves training state and reloads trained purifiers
train.py        Algorithm 2 training loop and readable training logs
infer.py        low-level image-directory purifier helper
smoke_test.py   lightweight dataset/checkpoint sanity checks
```

Important value ranges:

```text
Loaded tensors:     [-1, 1]
Model predictions:  [-1, 1]
Saved PNG images:   [0, 1] converted to uint8 RGB
```

The model wrapper predicts a clean image. With the default `cm_output_mode` of
`full_boundary`, the wrapped denoiser predicts epsilon, converts it to `x_0`,
and applies the consistency-model boundary form `c_skip * x_t + c_out * x_0`.
The older `pred_x0` and `no_skip_boundary` modes remain available for ablations.

## Logs

Training logs are written by both the shell runner and Python training module.
The log body has section banners and compact metric lines like:

```text
Step: 100/50000 | 0.20% | lr: 0.000100 | Training loss is 0.1234 | distill: ... | rec: ... | id: ... | cls: ... | elapsed: ... | eta: ... | gpu: ...
```

The main log is:

```text
consistency_model/logs/cm_purifier_train_<job_id>.log
```

The error log is:

```text
consistency_model/logs/cm_purifier_train_err_<job_id>.log
```

## Low-Level Inference Helper

`infer.py` is a generic image-directory purifier. It is useful for debugging or
small experiments, but it is not the project-level Algorithm 3 runner.

```bash
python -m consistency_model.cm_purifier.infer \
  --checkpoint consistency_model/checkpoints/cm_purifier.pth \
  --input data/untrusted_images \
  --output data/sanitized_images \
  --t-star 200 \
  --batch-size 256
```

For held-out test cases under `dataset_generation/datasets/test`, use
`purify/run_purify_test.sh` instead.

## Smoke Check

After training creates the `.pth` checkpoint, submit:

```bash
sbatch consistency_model/smoke_test.sh
```

The smoke test is not Algorithm 3. It verifies that:

```text
1. The pair dataset has the expected shape.
2. The saved .pth checkpoint can be loaded.
3. Two real PNG images can pass through the purifier and be saved.
```

Expected smoke output is JSON with:

```text
dataset_check
checkpoint_two_image_smoke
```

The output images are written to:

```text
consistency_model/smoke_outputs/
```

## Troubleshooting

If training cannot find CUDA, check the Slurm allocation and the CUDA section in
the log. The runner refuses CPU training because full training is too slow.

If the dataset check fails, confirm that `clean/` and `poisons/` contain the
same filenames under `dataset_generation/datasets/train`.

If checkpoint loading fails in later purification, make sure the file exists at:

```text
consistency_model/checkpoints/cm_purifier.pth
```

or pass the same custom `OUTPUT_PATH` during training and `CHECKPOINT_PATH`
during purification.
