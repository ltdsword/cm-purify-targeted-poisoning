# DiffPure SLURM Usage Guide

This guide explains how to run DiffPure experiments on your university SLURM cluster.

## ✅ Pre-flight Checklist

All required files are in place:
- ✅ **ImageNet LMDB**: `dataset/imagenet_lmdb/val.lmdb` (6.4 GB)
- ✅ **Score SDE (CIFAR-10)**: `pretrained/score_sde/cifar10_ddpmpp_deep_continuous/checkpoint_8.pth`
- ✅ **Guided Diffusion (ImageNet)**: `pretrained/guided_diffusion/256x256_diffusion_uncond.pt`
- ✅ **CIFAR-10 Classifiers**:
  - `pretrained/cifar10/wresnet-76-10/weights-best.pt`
  - `pretrained/cifar10/resnet-50/weights.pt`
  - `pretrained/cifar10/wrn-70-16-dropout/weights.pt`

**Note**: CIFAR-10 dataset will auto-download when you run experiments.

---

## 🚀 Quick Start

### Step 1: Set Up Environment (One-time setup)

```bash
cd /media02/ndthuc03/ltdsword/DiffPure
bash setup_env.sh
```

This creates a conda environment called `diffpure` with all dependencies.

### Step 2: Edit Run Configuration

Open `run.sh` and modify the CONFIGURATION section:

```bash
# Dataset: "cifar10" or "imagenet"
DATASET="cifar10"

# Attack type: "autoattack_linf_rand", "autoattack_linf_stand", etc.
ATTACK_TYPE="autoattack_linf_rand"

# Classifier
CLASSIFIER="wideresnet-28-10"

# Random seeds
SEED_ID=121
DATA_SEED=0
```

### Step 3: Submit Job

```bash
sbatch run.sh
```

### Step 4: Monitor Job

```bash
# Check job status
squeue -u $USER

# View live logs
tail -f logs/diffpure_<job_id>.log

# Check for errors
tail -f logs/diffpure_err_<job_id>.log
```

---

## 📊 Available Experiments

### CIFAR-10 Experiments

| Attack Type | Classifier Options | Description |
|-------------|-------------------|-------------|
| `autoattack_linf_rand` | `wideresnet-28-10`, `wideresnet-70-16`, `resnet-50` | AutoAttack L∞ (Rand version) |
| `autoattack_linf_stand` | `wideresnet-28-10`, `wideresnet-70-16`, `resnet-50` | AutoAttack L∞ (Standard version) |
| `autoattack_l2_rand` | `wideresnet-28-10`, `wideresnet-70-16`, `resnet-50` | AutoAttack L2 (Rand version) |
| `autoattack_l2_stand` | `wideresnet-28-10`, `wideresnet-70-16`, `resnet-50` | AutoAttack L2 (Standard version) |
| `stadv` | `resnet-50` only | StAdv attack |
| `bpda` | `wideresnet-28-10` only | BPDA+EOT attack |

**Recommended seeds for paper reproduction:**
- `SEED_ID`: 121, 122, 123 (3 seeds for error bars)
- `DATA_SEED`: 0-7 (8 seeds for CIFAR-10)

### ImageNet Experiments

| Attack Type | Classifier Options | Description |
|-------------|-------------------|-------------|
| `autoattack_linf_rand` | `resnet50`, `wideresnet-50-2`, `deit-s` | AutoAttack L∞ (Rand version) |
| `autoattack_linf_stand` | `resnet50`, `wideresnet-50-2`, `deit-s` | AutoAttack L∞ (Standard version) |

**Recommended seeds for paper reproduction:**
- `SEED_ID`: 121, 122, 123 (3 seeds for error bars)
- `DATA_SEED`: 0-31 (32 seeds for ImageNet)

---

## 📝 Example Configurations

### Example 1: CIFAR-10 AutoAttack L∞ (WideResNet-28-10)

```bash
DATASET="cifar10"
ATTACK_TYPE="autoattack_linf_rand"
CLASSIFIER="wideresnet-28-10"
SEED_ID=121
DATA_SEED=0
```

### Example 2: CIFAR-10 AutoAttack L2 (ResNet-50)

```bash
DATASET="cifar10"
ATTACK_TYPE="autoattack_l2_rand"
CLASSIFIER="resnet-50"
SEED_ID=121
DATA_SEED=0
```

### Example 3: ImageNet AutoAttack L∞ (DeiT-S)

```bash
DATASET="imagenet"
ATTACK_TYPE="autoattack_linf_rand"
CLASSIFIER="deit-s"
SEED_ID=121
DATA_SEED=0
```

### Example 4: CIFAR-10 BPDA+EOT Attack

```bash
DATASET="cifar10"
ATTACK_TYPE="bpda"
CLASSIFIER="wideresnet-28-10"
SEED_ID=121
DATA_SEED=0
```

---

## 📁 Output Files

Results are saved to:
```
/media02/ndthuc03/ltdsword/DiffPure/exp_results/
```

Logs are saved to:
```
/media02/ndthuc03/ltdsword/DiffPure/logs/
├── diffpure_<job_id>.log       # Standard output
└── diffpure_err_<job_id>.log   # Error output
```

---

## 🔄 Running Multiple Experiments

To reproduce paper results, you need to run multiple seeds. Here's a batch submission script:

```bash
#!/bin/bash
# batch_submit.sh - Submit multiple experiments

cd /media02/ndthuc03/ltdsword/DiffPure

# CIFAR-10: 3 seeds × 8 data_seeds = 24 jobs
for seed_id in 121 122 123; do
    for data_seed in 0 1 2 3 4 5 6 7; do
        # Modify run.sh parameters
        sed -i "s/^SEED_ID=.*/SEED_ID=$seed_id/" run.sh
        sed -i "s/^DATA_SEED=.*/DATA_SEED=$data_seed/" run.sh
        
        # Submit job
        sbatch run.sh
        
        # Wait a bit to avoid overwhelming scheduler
        sleep 2
    done
done
```

---

## ⚙️ Resource Requirements

### CIFAR-10
- **GPU**: 1 GPU (16-32 GB memory)
- **RAM**: 32 GB
- **Time**: ~1-4 hours per experiment
- **CPUs**: 8 cores

### ImageNet
- **GPU**: 1 GPU (32 GB memory recommended)
- **RAM**: 32 GB
- **Time**: ~4-12 hours per experiment
- **CPUs**: 8 cores

If you need more GPUs for ImageNet experiments, edit the SLURM directives in `run.sh`:

```bash
#SBATCH --gres=gpu:4  # Use 4 GPUs instead of 1
```

---

## 🐛 Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size in the experiment scripts:
```bash
# Edit the corresponding script in run_scripts/
# Change --adv_batch_size to a smaller value (e.g., 32 instead of 64)
```

### Issue: Environment not found

**Solution**: 
```bash
conda activate diffpure
# If doesn't exist, run setup again:
bash setup_env.sh
```

### Issue: Import errors

**Solution**: Reinstall dependencies:
```bash
conda activate diffpure
bash setup_env.sh
```

### Issue: Job stuck in queue

**Solution**: Check partition availability:
```bash
sinfo
squeue -u $USER
```

---

## 📊 Expected Results

After running experiments, you should see output like:

```
Clean accuracy: 94.8%
Robust accuracy (purified): 79.2%
Attack success rate: 18.5%
```

To get final paper numbers, average across all seeds:
- Min of (Rand version, Standard version) for worst-case performance
- Average across SEED_ID for error bars

---

## 📞 Support

For questions about:
- **DiffPure method**: See [official repo](https://github.com/NVlabs/DiffPure)
- **SLURM issues**: Contact your university cluster support
- **Dataset/model issues**: Check `README.md` and `run.md`

---

**Last Updated**: February 22, 2026
