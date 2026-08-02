#!/bin/bash
#SBATCH --job-name=cm_train_then_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

# Purpose: Walk upward from a starting directory until the project root is found.
# Input: a starting directory path.
# Output: repository root printed to stdout, or non-zero exit when not found.
find_repo_dir() {
    local start_dir="$1"
    local dir
    dir="$(cd "${start_dir}" && pwd)"

    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/requirements.txt" && -f "${dir}/consistency_model/cm_purifier/train.py" && -f "${dir}/benchmark/run_benchmark.py" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done

    return 1
}

# Purpose: Convert a 0/1 environment variable into an optional argparse flag.
# Input: variable value, expected flag name, and output array name.
# Output: appends the flag to the named array when the variable value is 1.
append_flag_if_enabled() {
    local value="$1"
    local flag="$2"
    local array_name="$3"
    if [[ "${value}" == "1" ]]; then
        eval "${array_name}+=(\"${flag}\")"
    fi
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && REPO_DIR="$(find_repo_dir "${SLURM_SUBMIT_DIR}")"; then
    :
else
    RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(find_repo_dir "${RUNNER_DIR}")"
fi

RUNNERS_DIR="${REPO_DIR}/runners"
CONSISTENCY_DIR="${REPO_DIR}/consistency_model"
BENCHMARK_DIR="${REPO_DIR}/benchmark"

PAIR_DIR="${PAIR_DIR:-${REPO_DIR}/dataset_generation/datasets/train}"
TEST_DIR="${TEST_DIR:-${REPO_DIR}/dataset_generation/datasets/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CONSISTENCY_DIR}/checkpoints/cm_purifier.pth}"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${BENCHMARK_DIR}/outputs}"
ENV_NAME="${ENV_NAME:-purifying_poison}"
ROOT_REQUIREMENTS="${ROOT_REQUIREMENTS:-${REPO_DIR}/requirements.txt}"

TEACHER_MODEL="${TEACHER_MODEL:-google/ddpm-cifar10-32}"
CM_OUTPUT_MODE="${CM_OUTPUT_MODE:-full_boundary}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${BATCH_SIZE:-128}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-50000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
TRAIN_LOG_STEPS="${TRAIN_LOG_STEPS:-${LOG_STEPS:-100}}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EMA_DECAY="${EMA_DECAY:-0.9999}"
GAMMA_WB="${GAMMA_WB:-1.0}"
GAMMA_BP="${GAMMA_BP:-1.0}"
GAMMA_CLEAN="${GAMMA_CLEAN:-0.0}"
LAMBDA_LPIPS="${LAMBDA_LPIPS:-0.0}"
LPIPS_NET="${LPIPS_NET:-alex}"
LPIPS_IMAGE_SIZE="${LPIPS_IMAGE_SIZE:-64}"
LPIPS_WARMUP_STEPS="${LPIPS_WARMUP_STEPS:-2000}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

RUN_ID="${RUN_ID:-}"
ATTACK_FILTER="${ATTACK_FILTER:-all}"
CASE_FILTER="${CASE_FILTER:-}"
MAX_CASES="${MAX_CASES:-}"
T_STAR="${T_STAR:-200}"
T_STARS="${T_STARS:-}"
BENCHMARK_BATCH_SIZE="${BENCHMARK_BATCH_SIZE:-64}"
BENCHMARK_SEED="${BENCHMARK_SEED:-2026}"
BENCHMARK_LOG_STEPS="${BENCHMARK_LOG_STEPS:-1024}"
SKIP_PURIFY="${SKIP_PURIFY:-0}"
SKIP_RETRAIN="${SKIP_RETRAIN:-0}"
OVERWRITE_ARTIFACTS="${OVERWRITE_ARTIFACTS:-0}"
WB_EPOCHS="${WB_EPOCHS:-}"
WB_DRYRUN="${WB_DRYRUN:-0}"
BP_VICTIM_NET="${BP_VICTIM_NET:-ResNet18}"
BP_CHECKPOINT_NAME="${BP_CHECKPOINT_NAME:-ckpt-%s-4800.t7}"
BP_RETRAIN_EPOCHS="${BP_RETRAIN_EPOCHS:-60}"
BP_RETRAIN_BSIZE="${BP_RETRAIN_BSIZE:-64}"

LOG_DIR="${RUNNERS_DIR}/logs"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
MAIN_LOG="${LOG_DIR}/train_then_benchmark_${JOB_ID}.log"
ERR_LOG="${LOG_DIR}/train_then_benchmark_err_${JOB_ID}.log"

mkdir -p "${LOG_DIR}" "$(dirname "${CHECKPOINT_PATH}")" "${BENCHMARK_OUTPUT_DIR}"
exec > >(tee -a "${MAIN_LOG}") 2> >(tee -a "${ERR_LOG}" >&2)

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Repository: ${REPO_DIR}"
echo "Pair directory: ${PAIR_DIR}"
echo "Test directory: ${TEST_DIR}"
echo "Checkpoint path: ${CHECKPOINT_PATH}"
echo "Benchmark output directory: ${BENCHMARK_OUTPUT_DIR}"
echo "LPIPS weight/network: ${LAMBDA_LPIPS}/${LPIPS_NET}"
echo "LPIPS image size/warmup: ${LPIPS_IMAGE_SIZE}/${LPIPS_WARMUP_STEPS}"
echo "Conda environment: ${ENV_NAME}"
echo "Root requirements: ${ROOT_REQUIREMENTS}"
echo "Logs directory: ${LOG_DIR}"
echo "Main log: ${MAIN_LOG}"
echo "Error log: ${ERR_LOG}"
echo "Started at: $(date -Is)"
echo "SLURM job GPUs: ${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM step GPUs: ${SLURM_STEP_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES before setup: ${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "Training settings:"
echo "  teacher model: ${TEACHER_MODEL}"
echo "  CM output mode: ${CM_OUTPUT_MODE}"
echo "  batch size: ${TRAIN_BATCH_SIZE}"
echo "  max steps: ${MAX_STEPS}"
echo "  log steps: ${TRAIN_LOG_STEPS}"
echo "  skip train: ${SKIP_TRAIN}"
echo "Benchmark settings:"
echo "  run id: ${RUN_ID:-<auto>}"
echo "  attack filter: ${ATTACK_FILTER}"
echo "  case filter: ${CASE_FILTER:-<none>}"
echo "  max cases: ${MAX_CASES:-<none>}"
echo "  t star: ${T_STAR}"
echo "  t star sweep: ${T_STARS:-<disabled>}"
echo "  batch size: ${BENCHMARK_BATCH_SIZE}"
echo "  seed: ${BENCHMARK_SEED}"
echo "  log steps: ${BENCHMARK_LOG_STEPS}"

if [[ ! -d "${PAIR_DIR}" ]]; then
    echo "ERROR: missing training pair dataset: ${PAIR_DIR}" >&2
    echo "Generate datasets first with: sbatch dataset_generation/runners/run_generation.sh" >&2
    exit 1
fi

if [[ ! -d "${TEST_DIR}" ]]; then
    echo "ERROR: missing held-out test dataset: ${TEST_DIR}" >&2
    echo "Generate datasets first with: sbatch dataset_generation/runners/run_generation.sh" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is required but was not found on PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
CONDA_BASE="$(conda info --base)"
ENV_DIR="${CONDA_BASE}/envs/${ENV_NAME}"
ENV_PYTHON="${ENV_DIR}/bin/python"

if [[ ! -d "${ENV_DIR}/conda-meta" ]]; then
    echo "Creating conda environment '${ENV_NAME}'."
    conda create -y --name "${ENV_NAME}" python=3.10 pip
elif [[ ! -x "${ENV_PYTHON}" ]]; then
    echo "Conda environment '${ENV_NAME}' is missing Python; repairing it."
    conda install -y --name "${ENV_NAME}" python=3.10 pip
fi
conda activate "${ENV_NAME}"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

REQ_NO_TORCH="$(mktemp)"
PIP_CONSTRAINTS="$(mktemp)"
trap 'rm -f "${REQ_NO_TORCH}" "${PIP_CONSTRAINTS}"' EXIT
grep -vE '^[[:space:]]*(torch|torchvision)([<>=!~ ].*)?$' "${ROOT_REQUIREMENTS}" > "${REQ_NO_TORCH}"
cat > "${PIP_CONSTRAINTS}" <<'EOF'
numpy<2
EOF

"${ENV_PYTHON}" -m pip install --upgrade pip
"${ENV_PYTHON}" - <<'PY' || "${ENV_PYTHON}" -m pip install --force-reinstall --constraint "${PIP_CONSTRAINTS}" torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
import numpy as np
import torch
import torchvision

assert np.__version__.startswith("1."), np.__version__
assert torch.__version__.startswith("2.2.2"), torch.__version__
assert "+cu118" in torch.__version__, torch.__version__
assert torchvision.__version__.startswith("0.17.2"), torchvision.__version__
PY
"${ENV_PYTHON}" -m pip install --constraint "${PIP_CONSTRAINTS}" -r "${REQ_NO_TORCH}"

"${ENV_PYTHON}" - <<'PY'
from diffusers import DDPMScheduler, UNet2DModel
import accelerate
import diffusers
import huggingface_hub
import safetensors
import transformers

expected = {
    "diffusers": (diffusers.__version__, "0.30.3"),
    "transformers": (transformers.__version__, "4.44.2"),
    "accelerate": (accelerate.__version__, "0.33.0"),
    "huggingface_hub": (huggingface_hub.__version__, "0.24.7"),
    "safetensors": (safetensors.__version__, "0.4.5"),
}
for package, (actual, wanted) in expected.items():
    if actual != wanted:
        raise SystemExit(f"ERROR: {package}=={actual}, expected {wanted}. Re-run dependency installation.")
print("HF stack OK:")
for package, (actual, _) in expected.items():
    print(f"  {package}: {actual}")
print(f"  diffusers classes: {DDPMScheduler.__name__}, {UNet2DModel.__name__}")
PY

"${ENV_PYTHON}" - <<'PY'
import os
import sys
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("ERROR: PyTorch cannot access CUDA; refusing to run train-then-benchmark on CPU.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
try:
    tensor = torch.zeros((1,), device="cuda")
    torch.cuda.synchronize()
except RuntimeError as exc:
    sys.exit(f"ERROR: CUDA allocation test failed before pipeline: {exc}")
PY

cd "${REPO_DIR}"

if [[ "${SKIP_TRAIN}" == "1" ]]; then
    echo "=============================="
    echo "1. SKIPPING ALGORITHM 2 TRAINING..."
    echo "=============================="
    if [[ ! -f "${CHECKPOINT_PATH}" && "${SKIP_PURIFY}" != "1" ]]; then
        echo "ERROR: SKIP_TRAIN=1 but checkpoint is missing: ${CHECKPOINT_PATH}" >&2
        exit 1
    fi
else
    echo "=============================="
    echo "1. RUNNING ALGORITHM 2 TRAINING..."
    echo "=============================="
    TRAIN_ARGS=(
        -u -m consistency_model.cm_purifier.train
        --pair-dir "${PAIR_DIR}"
        --teacher-model "${TEACHER_MODEL}"
        --backbone diffusers
        --cm-output-mode "${CM_OUTPUT_MODE}"
        --schedule-source diffusers
        --out "${CHECKPOINT_PATH}"
        --device cuda
        --batch-size "${TRAIN_BATCH_SIZE}"
        --num-workers "${NUM_WORKERS}"
        --max-steps "${MAX_STEPS}"
        --save-steps "${SAVE_STEPS}"
        --log-steps "${TRAIN_LOG_STEPS}"
        --learning-rate "${LEARNING_RATE}"
        --ema-decay "${EMA_DECAY}"
        --gamma-wb "${GAMMA_WB}"
        --gamma-bp "${GAMMA_BP}"
        --gamma-clean "${GAMMA_CLEAN}"
        --lambda-lpips "${LAMBDA_LPIPS}"
        --lpips-net "${LPIPS_NET}"
        --lpips-image-size "${LPIPS_IMAGE_SIZE}"
        --lpips-warmup-steps "${LPIPS_WARMUP_STEPS}"
    )
    "${ENV_PYTHON}" "${TRAIN_ARGS[@]}"
fi

if [[ ! -f "${CHECKPOINT_PATH}" && "${SKIP_PURIFY}" != "1" ]]; then
    echo "ERROR: expected checkpoint was not created: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

if [[ -n "${T_STARS}" ]]; then
    NORMALIZED_T_STARS="${T_STARS//,/ }"
    read -r -a BENCHMARK_T_STAR_VALUES <<< "${NORMALIZED_T_STARS}"
else
    BENCHMARK_T_STAR_VALUES=("${T_STAR}")
fi

if [[ "${#BENCHMARK_T_STAR_VALUES[@]}" -eq 0 ]]; then
    echo "ERROR: no benchmark timesteps were configured." >&2
    exit 1
fi

BENCHMARK_INDEX=0
for CURRENT_T_STAR in "${BENCHMARK_T_STAR_VALUES[@]}"; do
    BENCHMARK_INDEX=$((BENCHMARK_INDEX + 1))
    CURRENT_RUN_ID="${RUN_ID}"
    if [[ -n "${T_STARS}" ]]; then
        if [[ ! "${CURRENT_T_STAR}" =~ ^[0-9]+$ ]]; then
            echo "ERROR: T_STARS accepts integer DDPM timesteps, got '${CURRENT_T_STAR}'." >&2
            exit 1
        fi
        printf -v CURRENT_RUN_ID "t_star_%03d" "$((10#${CURRENT_T_STAR}))"
    fi

    BENCHMARK_ARGS=(
        -u -m benchmark.run_benchmark
        --checkpoint "${CHECKPOINT_PATH}"
        --test-dir "${TEST_DIR}"
        --output-dir "${BENCHMARK_OUTPUT_DIR}"
        --attack-filter "${ATTACK_FILTER}"
        --t-star "${CURRENT_T_STAR}"
        --batch-size "${BENCHMARK_BATCH_SIZE}"
        --device cuda
        --seed "${BENCHMARK_SEED}"
        --log-steps "${BENCHMARK_LOG_STEPS}"
        --bp-victim-net "${BP_VICTIM_NET}"
        --bp-checkpoint-name "${BP_CHECKPOINT_NAME}"
        --bp-retrain-epochs "${BP_RETRAIN_EPOCHS}"
        --bp-retrain-bsize "${BP_RETRAIN_BSIZE}"
    )

    if [[ -n "${CURRENT_RUN_ID}" ]]; then
        BENCHMARK_ARGS+=(--run-id "${CURRENT_RUN_ID}")
    fi
    if [[ -n "${CASE_FILTER}" ]]; then
        BENCHMARK_ARGS+=(--case-filter "${CASE_FILTER}")
    fi
    if [[ -n "${MAX_CASES}" ]]; then
        BENCHMARK_ARGS+=(--max-cases "${MAX_CASES}")
    fi
    append_flag_if_enabled "${SKIP_PURIFY}" "--skip-purify" BENCHMARK_ARGS
    append_flag_if_enabled "${SKIP_RETRAIN}" "--skip-retrain" BENCHMARK_ARGS
    append_flag_if_enabled "${OVERWRITE_ARTIFACTS}" "--overwrite-artifacts" BENCHMARK_ARGS
    if [[ -n "${WB_EPOCHS}" ]]; then
        BENCHMARK_ARGS+=(--wb-epochs "${WB_EPOCHS}")
    fi
    append_flag_if_enabled "${WB_DRYRUN}" "--wb-dryrun" BENCHMARK_ARGS

    echo "=============================="
    echo "2.${BENCHMARK_INDEX}. RUNNING BENCHMARK AT T_STAR=${CURRENT_T_STAR} IN SAME JOB..."
    echo "Output run ID: ${CURRENT_RUN_ID:-<auto>}"
    echo "=============================="
    "${ENV_PYTHON}" "${BENCHMARK_ARGS[@]}"
done

echo "=============================="
echo "DONE! Checkpoint: ${CHECKPOINT_PATH}"
echo "Benchmark outputs: ${BENCHMARK_OUTPUT_DIR}"
echo "Finished at: $(date -Is)"
echo "=============================="
