# Dataset Generation

This directory contains the pipeline for generating a purification training bank from targeted data poisoning attacks (Witches' Brew and Bullseye Polytope) on CIFAR-10. This generated dataset is used to train a Consistency Model (CM) as a defense/purifier.

## 🗂️ Dataset Split Plan

As detailed in `PLAN.md`, we generate a supervised dataset of 30,000 paired images (poisoned -> clean), specifically avoiding overlap between base images of different methods to maintain diversity.

For each CIFAR-10 class (5,000 images), the partition is:
- `[0..999]`: Witches' Brew (WB) training cases (2 cases × 500 images)
- `[1000..1499]`: WB evaluation cases (1 case × 500 images)
- `[1500..2499]`: Bullseye Polytope (BP) training (100 groups × 10 images)
- `[2500..2519]`: BP evaluation (2 groups × 10 images)
- `[2520..3519]`: Clean CM training pool (1,000 clean pairs)

**Total Training Bank:**
- 10,000 WB pairs
- 10,000 BP pairs
- 10,000 Clean pairs
- **Total: 30,000 pairs**

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

5. **Craft Bullseye Polytope Poisons**:
   ```bash
   python scripts/dataset_generation.py --mode craft_bp
   ```

*(Note: Use `scripts/test_generation.py` to perform a quick dry-run syntax/path evaluation before launching heavy generation jobs).*
