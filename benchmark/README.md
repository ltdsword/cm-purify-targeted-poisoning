# CM Purification Benchmark

This folder benchmarks whether the trained CM purifier reduces targeted poison
success after retraining.

The benchmark is intentionally separate from `purify/`: `purify/` exposes the
image-to-image purifier function, while `benchmark/` builds full tampered train
sets, purifies them, retrains victims, and writes metrics.

## What It Runs

For each held-out case in `dataset_generation/datasets/test`:

1. Build a full tampered CIFAR-10 train set.
2. Replace the clean base images with the poison images for that case.
3. Purify the full tampered train set with `purify.CMPurifier`.
4. Retrain/evaluate the poisoned train set.
5. Retrain/evaluate the purified train set.
6. Write the requested comparison table.

The benchmark purifies the whole train set because the realistic setting does
not know which images are poisoned.

## WB And BP Are Different

WB and BP do not share retraining code.

WB cases:

```text
WB_c<class>
```

use the Witches' Brew / `poisoning-gradient-matching` Forest stack:

```text
Kettle benchmark metadata
ResNet18 victim
Forest validation target checks
```

For poisoned WB training, only the poison base indices receive deltas. For
purified WB training, every CIFAR train image receives a purified-minus-clean
delta because the whole train set was sanitized.

BP cases:

```text
BP_c<class>_g<group>
```

use the BullseyePoison transfer-learning evaluator style:

```text
compact CIFAR10_TRAIN_Split.pth
pretrained ResNet18 victim by default
linear-head retraining by default
BP target prediction as attack success
```

The ResNet18 default matches `dataset_generation/scripts/dataset_generation.py`,
where BP poisons are generated with `--substitute-nets ResNet18` and
`--target-net ResNet18`.

BP setup indices are class-relative in the compact split, so the benchmark maps
them to flat split indices before building the BP retraining dataset.

## Output

Default root:

```text
benchmark/outputs/<run_id>/
```

Per case:

```text
<case>/
  poisoned_train/<label>/<index>.png
  purified_train/<label>/<index>.png
  purify/<original_poison_filename>.png
  target/<target_filename>.png
  summary.json
```

Run-level metrics:

```text
benchmark_results.csv
benchmark_results.jsonl
run_config.json
```

CSV columns:

```text
Case
Target
Attack
Clean Accuracy (Poison)
Target Acc (Poison)
Clean Acc (Purified)
Target Acc (Purified)
```

## Slurm Usage

Submit through Slurm:

```bash
sbatch benchmark/run_benchmark.sh
```

Default runner settings:

```text
conda env:   purifying_poison
checkpoint:  consistency_model/checkpoints/cm_purifier.pth
test dir:    dataset_generation/datasets/test
output dir:  benchmark/outputs
logs:        benchmark/logs/benchmark_<job_id>.log
GPU:         1
memory:      64GB
time:        48 hours
```

If the checkpoint is missing, the runner submits
`consistency_model/run_cm_purifier_training.sh` first and schedules the
benchmark with a Slurm `afterok` dependency.

Useful small checks:

```bash
MAX_CASES=1 SKIP_RETRAIN=1 sbatch benchmark/run_benchmark.sh
```

```bash
ATTACK_FILTER=WB MAX_CASES=1 WB_EPOCHS=1 sbatch benchmark/run_benchmark.sh
```

## Configuration

Environment variables accepted by `run_benchmark.sh`:

```text
CHECKPOINT_PATH
TEST_DIR
OUTPUT_DIR
RUN_ID
ATTACK_FILTER        all, WB, or BP
CASE_FILTER          comma-separated names/globs, e.g. WB_c0,BP_c0_g0
MAX_CASES
T_STAR
BATCH_SIZE           default 64 for full-train purification
SEED
LOG_STEPS
SKIP_PURIFY          1 skips CM purification
SKIP_RETRAIN         1 skips victim retraining/evaluation
OVERWRITE_ARTIFACTS  1 clears each case output dir first
WB_EPOCHS            optional Forest epoch override
WB_DRYRUN            1 enables Forest dryrun
BP_VICTIM_NET        default ResNet18
BP_CHECKPOINT_NAME   default ckpt-%s-4800.t7
BP_RETRAIN_EPOCHS    default 60
BP_RETRAIN_BSIZE     default 64
```

Python entrypoint:

```bash
python -m benchmark.run_benchmark --help
```

Use the Slurm runner for real jobs on the university server; direct Python is
only for help text or tiny local static checks.
