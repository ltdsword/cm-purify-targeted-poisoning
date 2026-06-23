# Dataset Generation Chat Summary - 16/06

This file summarizes what we checked, changed, and decided for the CIFAR-10 poison-pair dataset generation pipeline.

## Goal

Generate a paired purifier-training dataset for the CM project:

- 10,000 Witches' Brew clean/poison pairs.
- 10,000 Bullseye Polytope clean/poison pairs.
- 10,000 clean identity pairs.
- Held-out WB/BP evaluation folders with clean bases, poisons, and one target per case.

The pipeline should be runnable on a fresh machine by pulling the repo and submitting only:

```bash
sbatch dataset_generation/runners/run_generation.sh
```

No manual precompute step should be required, except normal server/conda availability and network access for first-time downloads.

## Files Reviewed

Main pipeline:

- `dataset_generation/runners/run_generation.sh`
- `dataset_generation/scripts/dataset_generation.py`
- `dataset_generation/setup_datasets.py`
- `dataset_generation/PLAN.md`
- `dataset_generation/README.md`
- root `README.md`

Attack code:

- `dataset_generation/poisoning-gradient-matching/`
- `dataset_generation/BullseyePoison/README.md`
- `dataset_generation/BullseyePoison/craft_poisons_transfer.py`
- `dataset_generation/BullseyePoison/utils.py`
- `dataset_generation/BullseyePoison/trainer.py`

Logs inspected during debugging:

- `dataset_generation/logs/poison_pipeline_71780.log`
- `dataset_generation/logs/poison_pipeline_err_71780.log`
- `dataset_generation/logs/poison_pipeline_71866.log`
- `dataset_generation/logs/poison_pipeline_err_71866.log`
- `dataset_generation/logs/poison_pipeline_72009.log`
- `dataset_generation/logs/poison_pipeline_err_72009.log`
- `slurm-72057.out`
- `slurm-72130.out`
- `slurm-72173.out`
- `slurm-72247.out`

## Server And Runner Fixes

`run_generation.sh` now:

- Finds the repo path robustly from `SLURM_SUBMIT_DIR` or from the runner location.
- Creates logs under `dataset_generation/logs/`.
- Writes both main and stderr logs:
  - `dataset_generation/logs/poison_pipeline_${SLURM_JOB_ID}.log`
  - `dataset_generation/logs/poison_pipeline_err_${SLURM_JOB_ID}.log`
- Uses the conda environment `purifying_poison`.
- Creates/repairs that conda env if missing.
- Forces `PYTHONNOUSERSITE=1` to avoid leaking user-site packages.
- Pins NumPy to `<2` to avoid the PyTorch/NumPy ABI crash.
- Validates:
  - NumPy 1.x
  - PyTorch `2.2.2+cu118`
  - Torchvision `0.17.2+cu118`
  - CUDA availability
  - a real CUDA allocation/model forward pass before poison generation
- Excludes `gpu03`, because logs showed CUDA allocation failures on that node.
- Runs sequentially:
  1. `setup_clean`
  2. `craft_wb`
  3. `craft_bp`

Important note: the `transformers` warning about PyTorch `>=2.4` is noisy but not relevant to WB/BP generation. The real NumPy issue was fixed by pinning NumPy `<2`.

## Dataset Split Design

We follow `dataset_generation/PLAN.md`.

For each CIFAR-10 class, after deterministic shuffle with seed `121`:

```text
[0..499]       WB train case 1
[500..999]     WB train case 2
[1000..1499]   WB eval case
[1500..2499]   BP train pool: 100 groups x 10 images
[2500..2519]   BP eval pool: 2 groups x 10 images
[2520..3519]   Clean identity training pool: 1000 images
[3520..4999]   Reserve
```

This avoids overlap between WB, BP, clean identity, and eval base images.

Generated counts after `setup_clean`:

- train clean PNGs: `30000`
- test clean PNGs: `5200`
- test targets: `30`
- WB setups: `30`
- BP setups: `1020`

## Naming And Folder Layout

The old naming/layout caused confusion:

- `WB_c0_eval`
- `WB_c0_WB_eval`
- `BP` files appearing before BP actually ran
- duplicated method names like `wb_c0_WB_train1_base_...`
- clean/poison filename mismatches

The cleaned layout is:

```text
dataset_generation/datasets/train/clean/
dataset_generation/datasets/train/poisons/
dataset_generation/datasets/test/WB_c{class}/clean/
dataset_generation/datasets/test/WB_c{class}/poisons/
dataset_generation/datasets/test/WB_c{class}/target/
dataset_generation/datasets/test/BP_c{class}_g{group}/clean/
dataset_generation/datasets/test/BP_c{class}_g{group}/poisons/
dataset_generation/datasets/test/BP_c{class}_g{group}/target/
```

Current filename rules:

- clean identity: `clean_c{class}_{image_idx}.png`
- WB: `wb_c{class}_{image_idx}.png`
- BP: `bp_c{class}_g{group}_{image_idx}.png`
- target: `target_c{target_class}_{target_idx}.png`

The clean and poison files for a case now share the same filename stem, so pairing is direct.

## WB Behavior

WB generation is already sequential and restart-friendly at the case/output level:

1. If all expected poison PNGs already exist, the WB case is skipped.
2. If `poisons.pickle` exists but PNG export was interrupted, PNGs are exported without recrafting.
3. Otherwise, the WB case is crafted from scratch.

WB had already completed in the latest successful partial run.

## BP Failure In `slurm-72247.out`

BP failed with:

```text
FileNotFoundError: datasets/CIFAR10_TRAIN_Split.pth
```

The BP script runs with `cwd=dataset_generation/BullseyePoison`, so the path it expects is:

```text
dataset_generation/BullseyePoison/datasets/CIFAR10_TRAIN_Split.pth
```

That file was missing during the failed run.

## BP Dataset ZIP / PTH Decision

The BullseyePoison README links an official `datasets.zip`.

The zip contains:

```text
datasets/CIFAR10_TRAIN_Split.pth
datasets/cifar-10-batches-py/
datasets/cifar-10-python.tar.gz
datasets/102flowers/
datasets/epfl-gims08/
```

So the official archive has the correct path shape. However, after checking the content, the official split has only about `200` usable CIFAR images per class in the relevant split. That is not enough for our `PLAN.md`, which needs BP base positions:

- train: `[1500..2499]`
- eval: `[2500..2519]`

Therefore, using the official split directly would not support our plan.

The current solution is to generate our own BP-compatible split at the same expected path:

```text
dataset_generation/BullseyePoison/datasets/CIFAR10_TRAIN_Split.pth
```

The file is a compact full-CIFAR split:

- `others`: all 50,000 CIFAR-10 train images, ordered class-by-class using our deterministic per-class shuffle
- `clean_train`: same data as `others`
- `target`: CIFAR-10 test images
- `format`: `cm_custom_compact_v1`

Current file size:

```text
338400551 bytes
```

This is intentionally located inside `BullseyePoison/datasets/`, because BP uses relative paths from the BP folder.

## BP Compatibility Patch

Original BP utilities expected:

```python
torch.load("datasets/CIFAR10_TRAIN_Split.pth")[subset]
```

with a list-like structure of `(image, label)` pairs.

Our generated `.pth` is more compact, so `dataset_generation/BullseyePoison/utils.py` was patched to support both:

- the old official list-like format
- the new compact dict format with `data` and `targets`

Patched functions:

- `load_image_label_subset`
- `iter_image_label_subset`
- `fetch_target`
- `fetch_all_target_cls`
- `fetch_poison_bases`

Compatibility check performed:

- `fetch_poison_bases(0, 10, "others", ..., start_idx=1500)` returns the expected BP base positions.
- The fetched BP base image matches the clean PNG generated by `setup_clean` with only tiny float conversion difference.

Important limit: the custom `.pth` is compatible with the patched BP generation path. Untouched BP scripts that directly call `torch.load(...)[subset]` and assume the old list format may still need the same helper if we use them later.

## BP Checkpoint And Model Folder Fix

The BP model checkpoint download from Google Drive extracts as `model-chks-release` or similar, but the BP code expects:

```text
dataset_generation/BullseyePoison/model-chks/
```

`dataset_generation.py` now:

- checks `BullseyePoison/model-chks`
- moves downloaded/extracted checkpoint files from `model-chks-release` or `model_chks_release` into `model-chks`
- creates a compatibility alias when possible
- downloads from Google Drive only if checkpoints are missing

The needed checkpoint currently exists:

```text
dataset_generation/BullseyePoison/model-chks/ckpt-ResNet18-4800.t7
```

## BP Restart / Resume Design

BP is now designed to run sequentially like WB, with stronger recovery inside a case.

For every BP setup `bp_i`:

1. If all expected poison PNGs exist, skip the case.
2. Else, if `benchmark_results/bp_i/poisons.pickle` exists, export PNGs from it.
3. Else, search for the latest internal BP checkpoint:

```text
dataset_generation/BullseyePoison/benchmark_results/bp_i/{BP_MODE}/{BP_POISON_ITERS}/{target_index}/poison_XXXXX.pth
```

4. If a checkpoint exists, call BP with:

```bash
--resume-poison-ite XXXXX
```

5. Otherwise, start that BP case from scratch.

BP saves a checkpoint every 50 iterations and at the final iteration.

`craft_poisons_transfer.py` was also patched so that, when called by our dataset-generation wrapper with `BP_EXPORT_DIR`, resumed runs keep writing into the same stable case directory instead of redirecting into a `-resume` directory. This makes repeated Slurm restarts deterministic and easy to reason about.

This gives the behavior we wanted:

- completed BP cases are skipped
- exported but not copied cases are copied
- interrupted current cases resume from their latest `poison_XXXXX.pth`
- later cases have not started yet

## Commands To Run

Normal submit:

```bash
sbatch dataset_generation/runners/run_generation.sh
```

Submit a second job only after the first succeeds:

```bash
jid1=$(sbatch --parsable dataset_generation/runners/run_generation.sh)
jid2=$(sbatch --parsable --dependency=afterok:${jid1} dataset_generation/runners/run_generation.sh)
```

Optional shorter BP test before full run:

```bash
BP_POISON_ITERS=200 sbatch dataset_generation/runners/run_generation.sh
```

The full default is:

```text
BP_MODE=mean
BP_POISON_ITERS=1500
```

## What Does Not Need Manual Clearing

You do not need to delete previous artifacts before resubmitting.

The pipeline is intended to be idempotent:

- clean setup regenerates deterministic config and clean images
- WB skips existing exported poisons
- BP skips existing exported poisons
- BP resumes partial checkpoints

Only delete artifacts when you intentionally want to change the split, naming convention, attack parameters, or poison iteration count.

## Remaining Risks Before The Next Long Run

The big remaining risk is runtime, not path correctness.

There are `1020` BP cases. Even with resume, full `1500`-iteration BP for every case may require many 48-hour jobs. Chained Slurm submissions with `afterok` are recommended.

Also, if `BP_POISON_ITERS` changes between runs, the checkpoint directory changes because the iteration count is part of the path. That is intentional. Do not mix `BP_POISON_ITERS=200` test artifacts with a full `BP_POISON_ITERS=1500` run unless you understand that they are separate outputs.

## Validation Already Done

Checks performed after the changes:

- `setup_clean` completed successfully.
- Dataset counts matched the plan.
- WB config has `30` setups.
- BP config has `1020` setups.
- BP custom `.pth` exists at the path BP expects.
- BP custom `.pth` was tested through patched BP utility functions.
- Model checkpoint folder exists under `BullseyePoison/model-chks`.
- Test folder layout now uses one folder per eval case:
  - `WB_c{class}/clean,poisons,target`
  - `BP_c{class}_g{group}/clean,poisons,target`
- Train clean names are consistent with future poison output names.

## Current Important Modified Files

- `dataset_generation/runners/run_generation.sh`
- `dataset_generation/scripts/dataset_generation.py`
- `dataset_generation/BullseyePoison/utils.py`
- `dataset_generation/BullseyePoison/craft_poisons_transfer.py`
- `dataset_generation/setup_datasets.py`
- `SUMMARY_16_06.md`

