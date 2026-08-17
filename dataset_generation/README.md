# Dataset Generation on the Slurm Server

This directory generates the CIFAR-10 purifier-training bank and held-out evaluation cases for Witches' Brew (WB), Bullseye Polytope (BP), and Narcissus (NS). The supported execution path is the Slurm runner:

```text
dataset_generation/runners/run_generation.sh
```

Do not launch `dataset_generation.py` directly on a login node. The Slurm runner allocates a GPU, prepares the software environment, verifies CUDA, creates the CIFAR-10 split, and invokes the requested generation stages.

## 1. Final dataset layout

For each CIFAR-10 semantic class, the fixed shuffled positions are allocated as follows:

| Shuffled class positions | Purpose |
|---|---|
| `[0, 999]` | WB purifier-training pairs: two blocks of 500 |
| `[1000, 1499]` | Held-out WB evaluation and NS evaluation bases |
| `[1500, 2499]` | BP purifier-training pairs: 100 groups of 10 |
| `[2500, 2519]` | Held-out BP evaluation: two groups of 10 |
| `[2520, 3519]` | Clean identity pairs |
| `[3520, 4519]` | NS purifier-training pairs |
| `[4520, 4999]` | Reserve |

The final bank under `dataset_generation/datasets/train/` contains:

```text
10,000 clean identity pairs
10,000 WB pairs
10,000 BP pairs
10,000 NS pairs
---------------------------
40,000 total pairs
```

NS and WB evaluation intentionally share the clean base indices at positions `[1000, 1499]`. These positions never appear in the purifier-training allocation, so this does not introduce purifier-training/evaluation leakage. The shared allocation is recorded in the NS manifests.

## 2. Server prerequisites

The server must provide:

- Slurm with `sbatch` and a GPU partition supporting `--gres=gpu:1`;
- Conda available from the login and compute-node shell;
- an NVIDIA driver compatible with CUDA 11.8;
- outbound access to PyPI and the PyTorch package index while the environment is prepared;
- sufficient persistent storage for CIFAR-10, Tiny ImageNet, poison checkpoints, 40,000 paired images, and held-out cases;
- `wget` or `curl`, plus `unzip`, for staging Tiny ImageNet.

Submit jobs from the repository root—the directory containing `requirements.txt`, `dataset_generation/`, and `consistency_model/`. The runner locates the repository from `SLURM_SUBMIT_DIR` and changes to that root before starting the pipeline.

### Automatic dependency installation

No separate Python environment command is required. On every submission, `run_generation.sh`:

1. creates or reuses the Conda environment `purifying_poison` with Python 3.10;
2. installs or verifies `torch==2.2.2+cu118` and `torchvision==0.17.2+cu118`;
3. constrains NumPy to version 1.x with `numpy<2`;
4. installs the dataset-generation dependencies from the root `requirements.txt`, including Pillow, tqdm, LPIPS, LMDB, EfficientNet-PyTorch, pandas, SciPy, OpenCV, Matplotlib, Google Cloud clients, and gdown;
5. checks that PyTorch can see the allocated CUDA device and perform a GPU allocation.

The runner intentionally excludes the Hugging Face diffusion packages because dataset generation does not use them. Those packages are installed by the CM-training and benchmark runners when required.

### Automatically obtained artifacts

- CIFAR-10 is downloaded by the setup stage into `dataset_generation/datasets/` when absent.
- BullseyePoison checkpoints are reused from `dataset_generation/BullseyePoison/model-chks/` when present. If the directory is empty, the BP stage downloads and extracts the checkpoint archive through gdown.
- WB does not require a separately downloaded victim checkpoint. It runs Forest with `--vruns 0`; Forest records the poison-crafting model initialization seed, and no post-crafting victim is trained during generation.

## 3. Install Tiny ImageNet

Narcissus requires the Tiny ImageNet training split as its proxy out-of-distribution (POOD) dataset. It is not downloaded by the runner, and there is deliberately no fallback dataset.

On the server login node, choose a persistent storage directory outside the Git repository. Replace every `USERNAME` placeholder below with the server account name (for example, `ndthuc03`):

```bash
TINY_IMAGENET_PARENT=/media02/USERNAME/datasets
mkdir -p "${TINY_IMAGENET_PARENT}"
cd "${TINY_IMAGENET_PARENT}"

wget -c http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip

NARCISSUS_POOD_ROOT="${TINY_IMAGENET_PARENT}/tiny-imagenet-200/train"
test -d "${NARCISSUS_POOD_ROOT}"

TINY_CLASS_COUNT="$(find "${NARCISSUS_POOD_ROOT}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
test "${TINY_CLASS_COUNT}" -eq 200
echo "Tiny ImageNet ready: ${NARCISSUS_POOD_ROOT} (${TINY_CLASS_COUNT} classes)"
```

If `wget` is unavailable, use this download command instead:

```bash
curl -L --retry 5 --continue-at - \
  http://cs231n.stanford.edu/tiny-imagenet-200.zip \
  --output tiny-imagenet-200.zip
```

The value passed as `NARCISSUS_POOD_ROOT` must be the `train` directory itself. Its immediate children must be the 200 class directories expected by `torchvision.datasets.ImageFolder`:

```text
tiny-imagenet-200/
└── train/
    ├── n01443537/
    │   └── images/
    ├── n01629819/
    │   └── images/
    └── ... 198 additional class directories
```

Do not pass the parent `tiny-imagenet-200/` directory, the ZIP file, or the validation directory.

## 4. Preflight checks for the server checkout

Before submitting a job, confirm that the server has the complete updated source tree:

```bash
cd /media02/USERNAME/cm-purify-targeted-poisoning

test -f requirements.txt
test -f dataset_generation/__init__.py
test -f dataset_generation/Narcissus/__init__.py
test -f dataset_generation/Narcissus/integration.py
test -f dataset_generation/Narcissus/dataset_builder.py
test -f dataset_generation/scripts/dataset_generation.py
test -f dataset_generation/runners/run_generation.sh
```

The following error indicates that the server is running an older or incomplete checkout:

```text
ModuleNotFoundError: No module named 'dataset_generation.Narcissus.dataset_builder';
'dataset_generation' is not a package
```

The corrected version requires both `dataset_generation/__init__.py` and the repository-root path bootstrap near the beginning of `dataset_generation/scripts/dataset_generation.py`. Synchronize those updated files—and the complete `dataset_generation/Narcissus/` directory—to the server before resubmitting. Merely installing Tiny ImageNet will not resolve this import error.

## 5. Slurm submissions

All commands below must be issued from the repository root. `--export=ALL` preserves the normal server environment and passes the pipeline controls explicitly to the compute job.

### Recommended complete workflow

Generate WB first:

```bash
sbatch --export=ALL,RUN_WB=1,RUN_BP=0,RUN_NARCISSUS=0 \
  dataset_generation/runners/run_generation.sh
```

After WB completes, generate BP:

```bash
sbatch --export=ALL,RUN_WB=0,RUN_BP=1,RUN_NARCISSUS=0 \
  dataset_generation/runners/run_generation.sh
```

After BP completes, generate all final Narcissus triggers, training pairs, evaluation cases, and validation metadata. Replace the POOD path with the installed location:

```bash
sbatch --export=ALL,RUN_WB=0,RUN_BP=0,RUN_NARCISSUS=1,NS_STAGE=all,NS_PROFILE=final,NS_CLASSES=all,NS_TRIGGER_KIND=both,NARCISSUS_POOD_ROOT=/media02/USERNAME/datasets/tiny-imagenet-200/train \
  dataset_generation/runners/run_generation.sh
```

Narcissus trigger generation is expensive. If the 48-hour allocation expires, submit the exact same command again. The runner reuses valid final triggers and resumes compatible checkpoints from:

```text
dataset_generation/Narcissus/checkpoints/train/class_XX/state.pt
dataset_generation/Narcissus/checkpoints/eval/class_XX/state.pt
```

Do not delete these directories between submissions. A checkpoint generated with different settings is rejected rather than silently mixed with the current run.

### Diagnostic smoke submission

The smoke profile uses target class 2, one short optimization stage, 50 poisoned training images, 500 triggered queries, and profile-specific output directories. It is only a pipeline diagnostic and must not be reported as an experimental result.

```bash
sbatch --export=ALL,RUN_WB=0,RUN_BP=0,RUN_NARCISSUS=1,NS_STAGE=all,NS_PROFILE=smoke,NS_CLASSES=2,NS_TRIGGER_KIND=both,NARCISSUS_POOD_ROOT=/media02/USERNAME/datasets/tiny-imagenet-200/train \
  dataset_generation/runners/run_generation.sh
```

Smoke artifacts are isolated under:

```text
dataset_generation/datasets/train_smoke/
dataset_generation/datasets/test_smoke/
dataset_generation/datasets/narcissus/smoke/
dataset_generation/Narcissus/checkpoints/smoke/
```

### Resume or regenerate a selected Narcissus class

Resume both triggers and all exports for class 4:

```bash
sbatch --export=ALL,RUN_WB=0,RUN_BP=0,RUN_NARCISSUS=1,NS_STAGE=all,NS_PROFILE=final,NS_CLASSES=4,NS_TRIGGER_KIND=both,NARCISSUS_POOD_ROOT=/media02/USERNAME/datasets/tiny-imagenet-200/train \
  dataset_generation/runners/run_generation.sh
```

Resume only the evaluation trigger for class 4:

```bash
sbatch --export=ALL,RUN_WB=0,RUN_BP=0,RUN_NARCISSUS=1,NS_STAGE=triggers,NS_PROFILE=final,NS_CLASSES=4,NS_TRIGGER_KIND=eval,NARCISSUS_POOD_ROOT=/media02/USERNAME/datasets/tiny-imagenet-200/train \
  dataset_generation/runners/run_generation.sh
```

Export the NS bank after all training triggers already exist:

```bash
sbatch --export=ALL,RUN_WB=0,RUN_BP=0,RUN_NARCISSUS=1,NS_STAGE=bank,NS_PROFILE=final,NS_CLASSES=all \
  dataset_generation/runners/run_generation.sh
```

Export held-out cases after all evaluation triggers already exist:

```bash
sbatch --export=ALL,RUN_WB=0,RUN_BP=0,RUN_NARCISSUS=1,NS_STAGE=eval,NS_PROFILE=final,NS_CLASSES=all \
  dataset_generation/runners/run_generation.sh
```

Validate completed final artifacts without regenerating triggers:

```bash
sbatch --export=ALL,RUN_WB=0,RUN_BP=0,RUN_NARCISSUS=1,NS_STAGE=validate,NS_PROFILE=final,NS_CLASSES=all \
  dataset_generation/runners/run_generation.sh
```

The POOD path is required only for `NS_STAGE=triggers` and `NS_STAGE=all`. Bank export, evaluation export, and validation load previously generated trigger artifacts and therefore do not access Tiny ImageNet.

## 6. Slurm controls

| Variable | Default | Meaning |
|---|---:|---|
| `RUN_WB` | `1` | Generate or export WB poisons |
| `RUN_BP` | `1` | Generate or export BP poisons |
| `RUN_NARCISSUS` | `1` | Run the selected Narcissus stage |
| `NARCISSUS_POOD_ROOT` | unset | Absolute Tiny ImageNet `train/` path |
| `NS_STAGE` | `all` | `triggers`, `bank`, `eval`, `validate`, or `all` |
| `NS_PROFILE` | `final` | `final` or diagnostic `smoke` |
| `NS_CLASSES` | `all` | `all`, one class such as `4`, or a comma-separated list |
| `NS_TRIGGER_KIND` | `both` | `train`, `eval`, or `both` |
| `NS_CHECKPOINT_INTERVAL` | `10` | Trigger-stage checkpoint interval |
| `NS_BATCH_SIZE` | `350` | Narcissus surrogate/warm-up/trigger batch size |
| `NS_NUM_WORKERS` | `8` | DataLoader workers |
| `ENV_NAME` | `purifying_poison` | Conda environment name |

The final profile locks the reportable protocol to `8/255`, 200 surrogate epochs, five warm-up epochs, and 1,000 trigger rounds. Use the smoke profile rather than weakening final settings.

## 7. Output and logs

Final NS cases have the following schema:

```text
dataset_generation/datasets/test/NS_c0/
├── clean/
├── poisons/
├── trigger.pt
├── trigger.json
├── poison_indices.json
├── train_manifest.jsonl
├── triggered_test_manifest.jsonl
└── metadata.json
```

The final trigger store is:

```text
dataset_generation/datasets/narcissus/triggers/train/class_00.pt
dataset_generation/datasets/narcissus/triggers/eval/class_00.pt
```

The locked seed namespaces are:

```text
CM pair selection:        11000
training trigger:         12000 + target class
evaluation trigger:       22000 + target class
evaluation poison order:  32000 + target class
CM inference:             52000
victim training:          62000
```

Every trigger is stored with JSON metadata and a SHA-256 hash. Validation requires the training and evaluation hashes to differ for every class.

Slurm output is copied into timestamped project logs:

```text
dataset_generation/logs/poison_pipeline_<job_id>.log
dataset_generation/logs/poison_pipeline_err_<job_id>.log
```

When a job fails, inspect the error log first. The beginning of the main log also records the resolved repository path, CUDA visibility, selected NS stage/profile/classes, environment name, and package versions.
