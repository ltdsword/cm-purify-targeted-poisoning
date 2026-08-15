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

## Attack-specific evaluation

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

Narcissus cases use `NS_c<class>`. For each case the benchmark replaces the 500 stored target-class indices in the official 50,000-image CIFAR-10 training set. It trains independent CIFAR ResNet-18 victims from scratch for poisoned and CM-defended data using the same seed and configuration: SGD (learning rate 0.1, momentum 0.9, weight decay `5e-4`), batch size 128, 200 epochs, milestones 100 and 150, and `RandomCrop(32, padding=4)`. Victim checkpoints resume independently under `victim_checkpoints/poison/` and `victim_checkpoints/purified/`.

NS reports natural accuracy on all 10,000 clean queries, target-class accuracy on 1,000 clean target-class queries, and ASR on 9,000 triggered non-target queries. The same held-out trigger is used for train poisons and triggered queries at scale 1.0. Test queries are never purified or magnified.

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
  victim_checkpoints/          # NS only
```

Run-level metrics:

```text
benchmark_results.csv
benchmark_results.jsonl
run_config.json
narcissus_summary.json         # when NS results are available
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

The legacy columns remain unchanged. Additional standardized natural-accuracy, attack-success/ASR, target-class-accuracy, and purification-timing columns are appended. Timing covers only the full load-purify-write phase and records image count, wall time, throughput, batch statistics, device, timestep, inference seed, and purifier checkpoint hash. Reused completed artifacts retain their original measurements and are marked `reused`.

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
ATTACK_FILTER        all, WB, BP, or NS
CASE_FILTER          comma-separated names/globs, e.g. WB_c0,BP_c0_g0,NS_c2
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
NS_PROFILE           final or smoke
NS_VICTIM_EPOCHS     default 200 final, 2 smoke
NS_VICTIM_BATCH_SIZE default 128
NS_VICTIM_WORKERS    default 8
NS_VICTIM_SEED       locked to 62000 in final mode
NS_VICTIM_CHECKPOINT_INTERVAL default 1 epoch
NS_VICTIM_RESUME     default 1; set 0 to ignore an existing victim checkpoint
NS_TRIGGERED_TEST_LIMIT unset for final; smoke defaults to 500
```

Example diagnostic run (not reportable):

```bash
ATTACK_FILTER=NS CASE_FILTER=NS_c2 NS_PROFILE=smoke TEST_DIR=dataset_generation/datasets/test_smoke RUN_ID=ns_smoke sbatch benchmark/run_benchmark.sh
```

For reportable results use `NS_PROFILE=final`, all ten NS cases, 200 victim epochs, and all 9,000 triggered queries per class.
Set a stable `RUN_ID` when resubmitting an expired job; materialized data, purification timing, and the two victim checkpoint trees are then discovered under the same run directory and resumed. A new automatic Slurm run ID intentionally starts a separate run.

Python entrypoint:

```bash
python -m benchmark.run_benchmark --help
```

Use the Slurm runner for real jobs on the university server; direct Python is
only for help text or tiny local static checks.
