#!/bin/bash
#SBATCH --job-name=cm_purify_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu03

set -euo pipefail

# Purpose: Walk upward from a starting directory until the project root is found.
# Input: a starting directory path.
# Output: repository root printed to stdout, or non-zero exit when not found.
find_repo_dir() {
    local start_dir="$1"
    local dir
    dir="$(cd "${start_dir}" && pwd)"

    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/requirements.txt" && -f "${dir}/purify/purify_test.py" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done

    return 1
}

# Purpose: Submit CM training and then submit this purification runner as a dependent Slurm job.
# Input: training runner path and this runner path.
# Output: Slurm jobs are submitted; the current job exits before purification begins.
submit_training_then_purify() {
    local train_script="$1"
    local self_script="$2"

    if ! command -v sbatch >/dev/null 2>&1; then
        echo "ERROR: checkpoint is missing and sbatch is unavailable; cannot submit training job." >&2
        exit 1
    fi

    echo "Checkpoint is missing: ${CHECKPOINT_PATH}"
    echo "Submitting CM training first: ${train_script}"
    local train_submit
    train_submit="$(
        sbatch \
            --export=ALL,OUTPUT_PATH="${CHECKPOINT_PATH}",ENV_NAME="${ENV_NAME}",ROOT_REQUIREMENTS="${ROOT_REQUIREMENTS}" \
            "${train_script}"
    )"
    echo "${train_submit}"
    local train_job_id
    train_job_id="$(echo "${train_submit}" | awk '{print $NF}')"
    if [[ -z "${train_job_id}" ]]; then
        echo "ERROR: could not parse training Slurm job id from: ${train_submit}" >&2
        exit 1
    fi

    echo "Submitting this purification job after training job ${train_job_id} succeeds."
    sbatch \
        --dependency=afterok:"${train_job_id}" \
        --export=ALL,TEST_DIR="${TEST_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",ENV_NAME="${ENV_NAME}",ROOT_REQUIREMENTS="${ROOT_REQUIREMENTS}",T_STAR="${T_STAR}",BATCH_SIZE="${BATCH_SIZE}",SEED="${SEED}",LOG_STEPS="${LOG_STEPS}",MAX_IMAGES="${MAX_IMAGES}",COPY_REFERENCE_DIRS="${COPY_REFERENCE_DIRS}" \
        "${self_script}"
    echo "Exiting current purification job; dependent purification job will run after training."
    exit 0
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && REPO_DIR="$(find_repo_dir "${SLURM_SUBMIT_DIR}")"; then
    :
else
    RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(find_repo_dir "${RUNNER_DIR}")"
fi

PURIFY_DIR="${REPO_DIR}/purify"
CONSISTENCY_DIR="${REPO_DIR}/consistency_model"
TRAIN_SCRIPT="${CONSISTENCY_DIR}/run_cm_purifier_training.sh"
SELF_SCRIPT="${PURIFY_DIR}/run_purify_test.sh"
TEST_DIR="${TEST_DIR:-${REPO_DIR}/dataset_generation/datasets/test}"
OUTPUT_DIR="${OUTPUT_DIR:-${PURIFY_DIR}/outputs/test_purified}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CONSISTENCY_DIR}/checkpoints/cm_purifier.pth}"
ENV_NAME="${ENV_NAME:-purifying_poison}"
ROOT_REQUIREMENTS="${ROOT_REQUIREMENTS:-${REPO_DIR}/requirements.txt}"
T_STAR="${T_STAR:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-2026}"
LOG_STEPS="${LOG_STEPS:-256}"
MAX_IMAGES="${MAX_IMAGES:-}"
COPY_REFERENCE_DIRS="${COPY_REFERENCE_DIRS:-1}"
LOG_DIR="${PURIFY_DIR}/logs"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
MAIN_LOG="${LOG_DIR}/purify_test_${JOB_ID}.log"
ERR_LOG="${LOG_DIR}/purify_test_err_${JOB_ID}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
exec > >(tee -a "${MAIN_LOG}") 2> >(tee -a "${ERR_LOG}" >&2)

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Repository: ${REPO_DIR}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Test directory: ${TEST_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Conda environment: ${ENV_NAME}"
echo "Root requirements: ${ROOT_REQUIREMENTS}"
echo "Logs directory: ${LOG_DIR}"
echo "Main log: ${MAIN_LOG}"
echo "Error log: ${ERR_LOG}"
echo "Started at: $(date -Is)"
echo "SLURM job GPUs: ${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM step GPUs: ${SLURM_STEP_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES before setup: ${CUDA_VISIBLE_DEVICES:-<unset>}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    submit_training_then_purify "${TRAIN_SCRIPT}" "${SELF_SCRIPT}"
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
import os
import sys
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("ERROR: PyTorch cannot access CUDA; refusing to purify on CPU.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
try:
    tensor = torch.zeros((1,), device="cuda")
    torch.cuda.synchronize()
except RuntimeError as exc:
    sys.exit(f"ERROR: CUDA allocation test failed before purification: {exc}")
PY

cd "${REPO_DIR}"

PURIFY_ARGS=(
    -u -m purify.purify_test
    --checkpoint "${CHECKPOINT_PATH}"
    --input "${TEST_DIR}"
    --output "${OUTPUT_DIR}"
    --t-star "${T_STAR}"
    --batch-size "${BATCH_SIZE}"
    --device cuda
    --seed "${SEED}"
    --log-steps "${LOG_STEPS}"
)

if [[ -n "${MAX_IMAGES}" ]]; then
    PURIFY_ARGS+=(--max-images "${MAX_IMAGES}")
fi

if [[ "${COPY_REFERENCE_DIRS}" == "1" ]]; then
    PURIFY_ARGS+=(--copy-reference-dirs)
fi

echo "=============================="
echo "PURIFYING HELD-OUT TEST POISONS..."
echo "=============================="
"${ENV_PYTHON}" "${PURIFY_ARGS[@]}"

echo "=============================="
echo "DONE! Purified outputs are in ${OUTPUT_DIR}"
echo "Finished at: $(date -Is)"
echo "=============================="
