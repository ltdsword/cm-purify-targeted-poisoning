# Server Policy Compliance Summary

## ✅ Changes Made to Comply with Server Requirements

### 📋 Server Policies (from announcement)

1. **Maximum 2 jobs** per group at any time
2. **Maximum 48 hours** (2 days) runtime per job
3. **Maximum resources per job**: 2 GPUs, 16 CPUs, 64GB RAM
4. Jobs exceeding limits will be automatically cancelled
5. Must join Discord group: https://discord.gg/xAKB6Am4
6. Store data in /media path
7. Can use /raid on GPU nodes for training data (node-specific)
8. Save checkpoints to resume after 48-hour timeout

---

## ✅ Files Modified

### 1. **run.sh** (Main changes)

**SLURM Directives Updated:**
```bash
#SBATCH --time=48:00:00  # Changed from 72:00:00 to comply with 48h limit
```

**Added Server Policy Warnings:**
- Header section explaining 2-job limit
- Runtime warnings about 48-hour timeout
- Resource compliance notice
- Instructions for handling timeouts

**Current Settings (COMPLIANT):**
- ✅ 1 GPU (within 2 GPU limit)
- ✅ 8 CPUs (within 16 CPU limit)
- ✅ 32GB RAM (within 64GB limit)
- ✅ 48 hours (matches server limit)

**Expected Runtime:**
- CIFAR-10: ~30 hours (12 experiments) → ✅ **Will complete**
- ImageNet: ~48 hours (6 experiments) → ⚠️ **May timeout**

---

### 2. **batch_submit.sh**

**Added Warnings:**
- Cannot submit all jobs at once (violates 2-job limit)
- Must submit in batches of 2, wait for completion, then submit next batch
- Warning message before submission

---

### 3. **Documentation Files Updated**

**QUICKSTART.md:**
- Added server policy section at top
- Updated runtime estimates with 48h limit warnings
- Changed batch submission instructions to respect 2-job limit

**ANSWERS.md:**
- Added server policy warnings
- Updated full reproduction estimates (realistic for 2-job limit)
- Modified recommended workflow for 2-job constraint

---

## 📊 Impact on Experiments

### Single Job (run.sh)

| Dataset | Experiments | Time | Status |
|---------|-------------|------|--------|
| CIFAR-10 | 12 | ~30h | ✅ Should complete |
| ImageNet | 6 | ~48h | ⚠️ May timeout |

### Full Reproduction

**CIFAR-10** (24 jobs total):
- **Old estimate**: Run 24 GPUs in parallel = ~30 hours
- **New reality**: 2 jobs at a time = ~12 waves = **~360 hours (15 days)**

**ImageNet** (96 jobs total):
- **Old estimate**: Run 96 GPUs in parallel = ~48 hours
- **New reality**: 2 jobs at a time = ~48 waves = **~2,304 hours (96 days)**

---

## 🎯 Recommended Strategies

### Strategy 1: Quick Validation (2 jobs)
```bash
# Submit 2 jobs with different seeds
# Job 1: CIFAR-10, seed=121, data_seed=0
# Job 2: CIFAR-10, seed=121, data_seed=1
# Time: ~30 hours each
# Purpose: Verify setup works
```

### Strategy 2: Reduced Experiments
**Modify run.sh to run fewer experiments:**
```bash
# Edit run.sh line 43-44:
# Only run L-inf attacks (skip L2)
ATTACK_TYPES=("autoattack_linf_rand" "autoattack_linf_stand")
# This reduces from 12 to 6 experiments (~15 hours)
```

### Strategy 3: Focus on Single Classifier
**Edit run.sh to test only one classifier:**
```bash
# Edit run.sh line 46:
CLASSIFIERS=("wideresnet-28-10")
# This reduces from 12 to 4 experiments (~10 hours)
```

### Strategy 4: Split ImageNet Jobs
For ImageNet, consider splitting experiments across multiple jobs:
```bash
# Job 1: Run only Rand version (3 experiments)
ATTACK_TYPES=("autoattack_linf_rand")

# Job 2: Run only Stand version (3 experiments)
ATTACK_TYPES=("autoattack_linf_stand")
```

---

## 🚀 How to Submit Jobs (Respecting 2-Job Limit)

### Step 1: Check Current Jobs
```bash
squeue -u $USER
# Should show 0-1 jobs running (keep < 2)
```

### Step 2: Submit First Batch (2 jobs)
```bash
# Job 1: CIFAR-10, seed 121, data_seed 0
sed -i "s/^DATASET=.*/DATASET=\"cifar10\"/" run.sh
sed -i "s/^SEED_ID=.*/SEED_ID=121/" run.sh
sed -i "s/^DATA_SEED=.*/DATA_SEED=0/" run.sh
sbatch run.sh

# Job 2: CIFAR-10, seed 121, data_seed 1
sed -i "s/^DATA_SEED=.*/DATA_SEED=1/" run.sh
sbatch run.sh
```

### Step 3: Monitor Progress
```bash
# Check job status
squeue -u $USER

# Watch logs
tail -f logs/diffpure_all_*.log
```

### Step 4: Wait for Completion
```bash
# Wait until jobs complete (check with squeue)
# Then submit next batch (data_seed 2 and 3)
```

### Step 5: Repeat
Continue in batches of 2 until all desired experiments complete.

---

## ⚠️ Timeout Handling

If a job is killed at 48 hours:

### Check What Completed
```bash
ls exp_results/
# Each completed experiment will have its own directory
```

### Continue from Where It Stopped
The looped run.sh will start from experiment 1 each time. If you need to continue:

**Option 1**: Check logs and manually note which experiments completed

**Option 2**: Modify run.sh to skip completed experiments:
```bash
# Add skip logic in the loop (advanced)
# Or run specific experiments using individual scripts in run_scripts/
```

**Option 3**: Use individual experiment scripts:
```bash
cd run_scripts/cifar10
# Run specific experiments that didn't complete
bash run_cifar_rand_inf.sh 121 0
```

---

## 📝 Pre-Submission Checklist

Before running `sbatch run.sh`:

- [ ] Checked `squeue -u $USER` (have < 2 jobs running)
- [ ] Edited `DATASET` in run.sh
- [ ] Set appropriate `SEED_ID` and `DATA_SEED`
- [ ] Verified conda environment exists: `conda activate diffpure`
- [ ] Understand job will be killed at 48 hours
- [ ] Know which experiments to expect (CIFAR-10: 12, ImageNet: 6)
- [ ] Have plan for handling potential timeout

---

## 📞 Additional Server Guidelines

1. **Join Discord**: https://discord.gg/xAKB6Am4
2. **Data storage**: Use `/media02/ndthuc03/ltdsword/` path (already configured ✓)
3. **Large datasets**: Can use `/raid` on specific GPU nodes (currently using /media ✓)
4. **VSCode**: Disconnect when not in use to reduce bandwidth
5. **Disk space**: Backup and delete unnecessary data (currently OK)

---

## 📊 Summary Status

| Item | Status | Notes |
|------|--------|-------|
| Time limit compliance | ✅ | Changed to 48h |
| Resource compliance | ✅ | 1 GPU, 8 CPU, 32GB |
| 2-job limit | ⚠️ | User must manage manually |
| CIFAR-10 completion | ✅ | Should complete in 48h |
| ImageNet completion | ⚠️ | May timeout |
| Documentation updated | ✅ | Warnings added |
| Batch scripts updated | ✅ | Warning about 2-job limit |

---

**Last Updated**: February 24, 2026

**Ready to use**: ✅ Yes, run.sh is compliant with all server policies
**Main limitation**: Can only run 2 jobs simultaneously - plan accordingly!
