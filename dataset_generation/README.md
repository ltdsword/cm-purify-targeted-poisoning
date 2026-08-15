# Dataset Generation

This directory generates the CIFAR-10 purifier bank and held-out cases for Witches' Brew (WB), Bullseye Polytope (BP), and Narcissus (NS).

## 🗂️ Dataset Split Plan

The final supervised bank contains 40,000 paired images (poisoned or identity input to clean target).

For each CIFAR-10 class (5,000 images), the partition is:
- `[0..999]`: Witches' Brew (WB) training cases (2 cases × 500 images)
- `[1000..1499]`: WB evaluation cases (1 case × 500 images)
- `[1500..2499]`: Bullseye Polytope (BP) training (100 groups × 10 images)
- `[2500..2519]`: BP evaluation (2 groups × 10 images)
- `[2520..3519]`: Clean CM training pool (1,000 clean pairs)
- `[3520..4519]`: NS CM pool (100 images per source class for each of 10 target-class triggers)
- `[4520..4999]`: reserve

NS evaluation intentionally reuses WB evaluation base positions `[1000..1499]`. This does not overlap CM training, but the WB and NS held-out cases share clean base indices; manifests record this fact.

**Total Training Bank:**
- 10,000 WB pairs
- 10,000 BP pairs
- 10,000 Clean pairs
- 10,000 NS pairs
- **Total: 40,000 pairs (4,000 per semantic class)**

## ⚙️ How to Generate the Dataset

1. **Install Prerequisites**: Make sure your conda/python environment is activated and requirements are installed.

2. **Download Required Model Checkpoints**:
   Before generating Bullseye Polytope poisons, you must download the pre-trained victim models/checkpoints into the `BullseyePoison/model-chks` directory. You can download and extract them via `gdown` using the Google Drive ID `1TwxNbJ1arDNQrBJdt5AFeaAbKC65HOko`.

   ```bash
   pip install gdown

   # Download the zip file
   gdown 1TwxNbJ1arDNQrBJdt5AFeaAbKC65HOko -O model_chks_release.zip

   # Unzip it
   unzip model_chks_release.zip
   
   # Move the models inside to BullseyePoison/model-chks (Ensure destination exists)
   mkdir -p BullseyePoison/model-chks
   mv model_chks_release/* BullseyePoison/model-chks/
   rm -rf model_chks_release.zip model_chks_release
   ```
   *(Note: The `dataset_generation.py` script automatically verifies and performs this download step if the folder is missing).*

3. **Initialize the splits and baseline sets**:
   This step sets up the data splits according to `PLAN.md` and generates clean references.
   ```bash
   python scripts/dataset_generation.py --mode setup_clean
   ```

4. **Craft Witches' Brew Poisons**:
   ```bash
   python scripts/dataset_generation.py --mode craft_wb
   ```

   WB uses Forest with `--vruns 0`: it trains the poison-crafting model but does not retrain a post-crafting victim. Forest already records the crafting model initialization seed in its result table, so no additional victim-seed artifact is added here.

5. **Craft Bullseye Polytope Poisons**:
   ```bash
   python scripts/dataset_generation.py --mode craft_bp
   ```

6. **Generate Narcissus artifacts**:

   Tiny ImageNet is mandatory and is never downloaded automatically. Point to its 200-class `ImageFolder` training directory:

   ```bash
   export NARCISSUS_POOD_ROOT=/path/to/tiny-imagenet-200/train
   python scripts/dataset_generation.py --mode craft_ns --pood-root "$NARCISSUS_POOD_ROOT" --ns-profile final
   ```

   Resumable stages are `craft_ns_triggers`, `craft_ns_bank`, `craft_ns_eval`, and `validate_ns`. Use `--ns-classes 2` for one class and `--ns-trigger-kind train|eval|both` for trigger jobs. Separate checkpoints live at `Narcissus/checkpoints/{train|eval}/class_XX/state.pt`; incompatible configurations are rejected.

   A diagnostic smoke run uses class 2, 50 poisons, 500 triggered queries, and short optimization stages. It writes to `datasets/train_smoke`, `datasets/test_smoke`, and profile-specific trigger/checkpoint directories so it cannot overwrite final artifacts:

   ```bash
   python scripts/dataset_generation.py --mode craft_ns --ns-profile smoke --ns-classes 2 --pood-root "$NARCISSUS_POOD_ROOT"
   ```

Each `datasets/test/NS_cN/` contains `clean/`, `poisons/`, `trigger.pt`, `poison_indices.json`, `train_manifest.jsonl`, `triggered_test_manifest.jsonl`, and `metadata.json`. Smoke outputs are not reportable final results.

The locked seed namespaces are: CM pair selection 11000, training trigger `12000+class`, evaluation trigger `22000+class`, evaluation poison order `32000+class`, CM inference 52000, and victim training 62000. Trigger tensors, SHA-256 hashes, resolved seeds, source indices, semantic labels, scales, and paths are stored in machine-readable metadata.

For Slurm, `runners/run_generation.sh` accepts `RUN_WB`, `RUN_BP`, `RUN_NARCISSUS`, `NARCISSUS_POOD_ROOT`, `NS_CLASSES`, `NS_STAGE` (`triggers`, `bank`, `eval`, `validate`, or `all`), `NS_PROFILE`, `NS_TRIGGER_KIND`, and checkpoint/batch/worker variables. For example, resume only evaluation-trigger class 4 with `RUN_WB=0 RUN_BP=0 NS_STAGE=triggers NS_TRIGGER_KIND=eval NS_CLASSES=4 sbatch dataset_generation/runners/run_generation.sh`.

*(Note: Use `scripts/test_generation.py` to perform a quick dry-run syntax/path evaluation before launching heavy generation jobs).*
