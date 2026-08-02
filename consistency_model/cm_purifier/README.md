# Pixel-Space CM Purifier

This package implements Algorithm 2 from the project README: poison-aware
Consistency Model training for targeted clean-label poison purification.

The implementation is a new pixel-space purifier. InstantPure is used only as a
reference for the research idea and coding style. This package does not import
from `consistency_model/InstantPure`, does not use LCM-LoRA, does not use Stable
Diffusion 1.5, and does not use Canny maps. The model is trained with normal
DDPM schedules on CIFAR-sized images. An optional frozen LPIPS network provides
perceptual supervision without adding a condition branch to the DDPM UNet.

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
8. Compute distillation, reconstruction, identity, optional LPIPS, and optional classifier loss.
9. Update the student model.
10. Update the EMA teacher.
```

The implemented loss is:

```text
loss = lambda_distill * loss_distill
     + lambda_rec     * loss_reconstruction
     + lambda_id      * loss_identity
     + lambda_lpips   * warmup(step) * loss_lpips
     + lambda_cls     * loss_classifier
```

Direct Python training keeps `lambda_lpips=0.0` and `lambda_cls=0.0` for
backward compatibility. The dedicated LPIPS Slurm runner enables
`lambda_lpips=0.10`; classifier loss remains disabled.

### What Each Loss Preserves

```text
loss_distill        matches the adjacent-step EMA consistency target
loss_reconstruction anchors every prediction to its paired clean image with L1
loss_identity       adds L1 pressure on clean identity pairs
loss_lpips          preserves perceptual features of every clean target
loss_classifier     optional label-semantic constraint, disabled by default
```

LPIPS complements the pixel losses rather than replacing them. LPIPS is less
sensitive than L1 to small pixel shifts and can preserve perceptually important
shape and texture, while L1 remains important for CIFAR-10 color and exact pixel
fidelity. It is not an explicit edge detector and it does not provide a Canny
condition to the UNet.

The LPIPS model is `lpips==0.1.4` with a frozen AlexNet backbone. Predictions
and clean targets are already in the required `[-1, 1]` range. They are
clamped to that range for LPIPS and bilinearly resized from `32x32` to `64x64`
before perceptual evaluation so later AlexNet feature maps do not collapse too
aggressively. LPIPS parameters never
receive gradients, but the LPIPS computation is not placed under `no_grad`:
the student prediction must receive the perceptual gradient.

The effective LPIPS weight increases linearly from zero to `0.10` over 2,000
optimization steps. On resume, the saved global step determines the next
warmup factor, so warmup does not restart.

## Run Training

On the university Slurm server, submit runners instead of running long Python
jobs directly. Baseline training remains available with:

```bash
sbatch consistency_model/run_cm_purifier_training.sh
```

The recommended LPIPS training command is:

```bash
sbatch consistency_model/run_cm_purifier_training_lpips.sh
```

The LPIPS wrapper executes `run_cm_purifier_training.sh` with `bash` inside the
same allocation. It does not submit a nested job.

Default runner settings:

```text
conda env:   purifying_poison
pair dir:    dataset_generation/datasets/train
baseline:    consistency_model/checkpoints/cm_purifier.pth
LPIPS:       consistency_model/checkpoints/cm_purifier_lpips.pth
logs:        consistency_model/logs/cm_purifier_train_<job_id>.log
GPU:         1
memory:      64GB
time:        48 hours
```

The runner creates or repairs the `purifying_poison` Conda environment, installs
CUDA PyTorch, and then installs the repository root `requirements.txt` without
replacing the CUDA Torch build. LPIPS is declared in that requirements file; it
is not installed manually into the active interactive environment. When LPIPS
is enabled, the runner constructs the selected LPIPS network before building
the CM, so missing package or pretrained-weight access fails early.

Common Slurm overrides:

```bash
MAX_STEPS=1000 BATCH_SIZE=64 LOG_STEPS=10 \
OUTPUT_PATH=consistency_model/checkpoints/cm_purifier_lpips_smoke.pth \
sbatch consistency_model/run_cm_purifier_training_lpips.sh
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
LAMBDA_LPIPS    target perceptual weight; LPIPS runner default 0.10
LPIPS_NET       LPIPS backbone: alex, vgg, or squeeze; default alex
LPIPS_IMAGE_SIZE square LPIPS evaluation size; default 64
LPIPS_WARMUP_STEPS linear weight warmup length; default 2000
```

For reference only, the Python module called by the runner is:

```bash
python -m consistency_model.cm_purifier.train \
  --pair-dir dataset_generation/datasets/train \
  --teacher-model google/ddpm-cifar10-32 \
  --backbone diffusers \
  --schedule-source diffusers \
  --out consistency_model/checkpoints/cm_purifier_lpips.pth \
  --lambda-lpips 0.10 \
  --lpips-net alex \
  --lpips-image-size 64 \
  --lpips-warmup-steps 2000 \
  --device cuda
```

Use the Slurm runner for real training.

## Checkpoints

The output is a PyTorch `.pth` checkpoint. LPIPS is a training-only frozen loss
network and is not embedded in the purifier. The file stores everything needed for
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

New LPIPS arguments are recorded under `args`, but the student/EMA model
architecture and state-dict format are unchanged. Old baseline checkpoints
remain loadable by the new inference code.

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
losses.py       combines distillation, L1, optional LPIPS, and classifier losses
ema.py          creates and updates the EMA teacher model
checkpoint.py   saves training state and reloads trained purifiers
train.py        Algorithm 2 training loop and readable training logs
infer.py        low-level image-directory purifier and private-noise helpers
smoke_test.py   lightweight dataset/checkpoint sanity checks
tests/          focused loss, warmup, and deterministic-noise tests
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
Step: 100/50000 | 0.20% | lr: 0.000100 | Training loss is ... | distill: ... | rec: ... | id: ... | lpips: ... | lpips weighted: ... | lpips weight: ... | cls: ... | elapsed: ... | eta: ... | gpu: ...
```

`lpips` is the raw perceptual distance. `lpips weight` is the current warmed-up
coefficient, and `lpips weighted` is the exact contribution added to the total.
Watching all three distinguishes failure to converge from a loss that is simply
weighted too strongly.

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
  --batch-size 256 \
  --seed 2026
```

Inference uses the standard DDPM corruption only:

```text
x_t = alpha_t * x + sigma_t * epsilon
```

There is no extra noise multiplier. `t_star` selects `alpha_t` and `sigma_t`
from the checkpoint schedule and is therefore the only purification-strength
control. A lower timestep usually preserves more image detail but may remove
less poison signal. Startup logs report the resolved timestep, alpha, sigma,
SNR, and noise seed.

Each `CMPurifier` owns a device-local `torch.Generator`. Downstream WB/BP
retraining may modify global Torch RNG state, but it cannot alter the next
purification noise draw. Identical seed, image ordering, and batch size therefore
produce the same Gaussian sequence across timestep runs.

For held-out test cases under `dataset_generation/datasets/test`, use
`purify/run_purify_test.sh` instead.

## Controlled `t_star` Benchmark Sweep

The complete one-job pipeline is:

```bash
sbatch runners/run_lpips_train_then_tstar_benchmarks.sh
```

It trains `cm_purifier_lpips.pth` once, then executes all four benchmark calls
sequentially in the same allocation. No called script submits another Slurm
job. Its output structure is:

```text
benchmark/outputs/slurm_<job_id>/
  t_star_050/
  t_star_100/
  t_star_150/
  t_star_200/
```

Each child contains `benchmark_results.csv`, `benchmark_results.jsonl`,
`run_config.json`, and the complete WB/BP case directories produced by a normal
single benchmark.

The sweep is configurable with an integer list:

```bash
T_STARS="50 100" sbatch runners/run_lpips_train_then_tstar_benchmarks.sh
```

For recovery after a timeout, reuse the checkpoint and select only unfinished
timesteps with `SKIP_TRAIN=1`, `T_STARS`, and the original
`BENCHMARK_OUTPUT_DIR`.

The independent-job alternative is to train the LPIPS model first:

```bash
sbatch consistency_model/run_cm_purifier_training_lpips.sh
```

After `consistency_model/checkpoints/cm_purifier_lpips.pth` exists, submit the
fixed-timestep benchmarks separately:

```bash
sbatch runners/run_benchmark_tstar_050.sh
sbatch runners/run_benchmark_tstar_100.sh
sbatch runners/run_benchmark_tstar_150.sh
sbatch runners/run_benchmark_tstar_200.sh
```

Do not run more than two of these GPU jobs concurrently. Every runner uses seed
`2026`, purification batch size `64`, one GPU, 64 GB host memory, and a 48-hour
limit. Each verifies the LPIPS checkpoint before environment setup and executes
`benchmark/run_benchmark.sh` directly in its current allocation. Missing
checkpoints cause an immediate error; no training or benchmark sub-job is
submitted.

Outputs are separated by timestep and job ID:

```text
benchmark/outputs/tstar_050_slurm_<job_id>/
benchmark/outputs/tstar_100_slurm_<job_id>/
benchmark/outputs/tstar_150_slurm_<job_id>/
benchmark/outputs/tstar_200_slurm_<job_id>/
```

Compare both attack removal and clean utility. The desirable setting has low
target attack success after purification while retaining clean accuracy close
to poisoned-set retraining. If target success is low at every timestep but clean
accuracy degrades as `t_star` grows, the purifier is removing the attack but
over-corrupting normal training images. If clean accuracy is retained at low
timesteps but target success remains high, the noise is too weak to remove the
poison signal.

## Smoke Check

After training creates the `.pth` checkpoint, submit:

```bash
CHECKPOINT_PATH=consistency_model/checkpoints/cm_purifier_lpips.pth \
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
consistency_model/checkpoints/cm_purifier_lpips.pth
```

or pass the same custom `OUTPUT_PATH` during training and `CHECKPOINT_PATH`
during purification.

If LPIPS import or AlexNet initialization fails, inspect the dependency section
near the beginning of the Slurm log. Confirm that `lpips==0.1.4` was installed
from the root `requirements.txt` and that pretrained AlexNet weights can be read
from the Torch cache or downloaded on the compute node.

If LPIPS loss becomes non-finite or dominates the total, stop the job and inspect
the raw and weighted LPIPS fields. Use a lower `LAMBDA_LPIPS` for the next run;
do not remove the reconstruction and identity anchors.
