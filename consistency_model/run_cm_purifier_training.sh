#!/bin/bash
#SBATCH --job-name=cm_purifier_train
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
        if [[ -f "${dir}/requirements.txt" && -f "${dir}/consistency_model/cm_purifier/train.py" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done

    return 1
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && REPO_DIR="$(find_repo_dir "${SLURM_SUBMIT_DIR}")"; then
    :
else
    RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(find_repo_dir "${RUNNER_DIR}")"
fi

CONSISTENCY_DIR="${REPO_DIR}/consistency_model"
PAIR_DIR="${PAIR_DIR:-${REPO_DIR}/dataset_generation/datasets/train}"
OUTPUT_PATH="${OUTPUT_PATH:-${CONSISTENCY_DIR}/checkpoints/cm_purifier.pth}"
TEACHER_MODEL="${TEACHER_MODEL:-google/ddpm-cifar10-32}"
CM_OUTPUT_MODE="${CM_OUTPUT_MODE:-full_boundary}"
ENV_NAME="${ENV_NAME:-purifying_poison}"
ROOT_REQUIREMENTS="${ROOT_REQUIREMENTS:-${REPO_DIR}/requirements.txt}"
LOG_DIR="${CONSISTENCY_DIR}/logs"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
MAIN_LOG="${LOG_DIR}/cm_purifier_train_${JOB_ID}.log"
ERR_LOG="${LOG_DIR}/cm_purifier_train_err_${JOB_ID}.log"

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PATH}")"
exec > >(tee -a "${MAIN_LOG}") 2> >(tee -a "${ERR_LOG}" >&2)

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Repository: ${REPO_DIR}"
echo "Pair directory: ${PAIR_DIR}"
echo "Output checkpoint: ${OUTPUT_PATH}"
echo "CM output mode: ${CM_OUTPUT_MODE}"
echo "Conda environment: ${ENV_NAME}"
echo "Root requirements: ${ROOT_REQUIREMENTS}"
echo "Logs directory: ${LOG_DIR}"
echo "Main log: ${MAIN_LOG}"
echo "Error log: ${ERR_LOG}"
echo "Started at: $(date -Is)"
echo "SLURM job GPUs: ${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM step GPUs: ${SLURM_STEP_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES before setup: ${CUDA_VISIBLE_DEVICES:-<unset>}"

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
import os
import sys
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("ERROR: PyTorch cannot access CUDA; refusing to run CM purifier training on CPU.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
try:
    tensor = torch.zeros((1,), device="cuda")
    torch.cuda.synchronize()
except RuntimeError as exc:
    sys.exit(f"ERROR: CUDA allocation test failed before training: {exc}")
PY

cd "${REPO_DIR}"

echo "=============================="
echo "TRAINING CM PURIFIER..."
echo "=============================="
"${ENV_PYTHON}" -u -m consistency_model.cm_purifier.train \
    --pair-dir "${PAIR_DIR}" \
    --teacher-model "${TEACHER_MODEL}" \
    --backbone diffusers \
    --cm-output-mode "${CM_OUTPUT_MODE}" \
    --schedule-source diffusers \
    --out "${OUTPUT_PATH}" \
    --device cuda \
    --batch-size "${BATCH_SIZE:-128}" \
    --num-workers "${NUM_WORKERS:-8}" \
    --max-steps "${MAX_STEPS:-50000}" \
    --save-steps "${SAVE_STEPS:-5000}" \
    --log-steps "${LOG_STEPS:-100}" \
    --learning-rate "${LEARNING_RATE:-1e-4}" \
    --ema-decay "${EMA_DECAY:-0.9999}" \
    --gamma-wb "${GAMMA_WB:-1.0}" \
    --gamma-bp "${GAMMA_BP:-1.0}" \
    --gamma-clean "${GAMMA_CLEAN:-0.0}"

echo "=============================="
echo "DONE! Checkpoint saved to ${OUTPUT_PATH}"
echo "Finished at: $(date -Is)"
echo "=============================="
