#!/bin/bash
#SBATCH --job-name=cm_purifier_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

find_repo_dir() {
    local start_dir="$1"
    local dir
    dir="$(cd "${start_dir}" && pwd)"

    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/requirements.txt" && -f "${dir}/consistency_model/cm_purifier/smoke_test.py" ]]; then
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
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CONSISTENCY_DIR}/checkpoints/cm_purifier.pth}"
SMOKE_INPUT_DIR="${SMOKE_INPUT_DIR:-${CONSISTENCY_DIR}/smoke_inputs}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${CONSISTENCY_DIR}/smoke_outputs}"
ENV_NAME="${ENV_NAME:-purifying_poison}"
LOG_DIR="${CONSISTENCY_DIR}/logs"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
MAIN_LOG="${LOG_DIR}/cm_purifier_smoke_${JOB_ID}.log"
ERR_LOG="${LOG_DIR}/cm_purifier_smoke_err_${JOB_ID}.log"

mkdir -p "${LOG_DIR}" "${SMOKE_INPUT_DIR}" "${SMOKE_OUTPUT_DIR}"
exec > >(tee -a "${MAIN_LOG}") 2> >(tee -a "${ERR_LOG}" >&2)

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Repository: ${REPO_DIR}"
echo "Pair directory: ${PAIR_DIR}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Smoke input directory: ${SMOKE_INPUT_DIR}"
echo "Smoke output directory: ${SMOKE_OUTPUT_DIR}"
echo "Logs directory: ${LOG_DIR}"
echo "Started at: $(date -Is)"
echo "SLURM job GPUs: ${SLURM_JOB_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES before setup: ${CUDA_VISIBLE_DEVICES:-<unset>}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "ERROR: checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
    echo "Submit consistency_model/run_cm_purifier_training.sh first, or set CHECKPOINT_PATH." >&2
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
    echo "ERROR: conda environment '${ENV_NAME}' does not exist." >&2
    echo "Run consistency_model/run_cm_purifier_training.sh first to create/install the environment." >&2
    exit 1
fi
conda activate "${ENV_NAME}"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

cd "${REPO_DIR}"

echo "=============================="
echo "LOADING CHECKPOINT AND PURIFYING TWO SMOKE IMAGES..."
echo "=============================="
"${ENV_PYTHON}" -u -m consistency_model.cm_purifier.smoke_test \
    --pair-dir "${PAIR_DIR}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --tmp-image-dir "${SMOKE_INPUT_DIR}" \
    --tmp-output-dir "${SMOKE_OUTPUT_DIR}"

echo "=============================="
echo "DONE! Smoke outputs are in ${SMOKE_OUTPUT_DIR}"
echo "Finished at: $(date -Is)"
echo "=============================="
