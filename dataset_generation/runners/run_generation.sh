#!/bin/bash
#SBATCH --job-name=generate_poisons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      
#SBATCH --mem=32G              
#SBATCH --gres=gpu:1           
#SBATCH --time=48:00:00        

set -euo pipefail

find_repo_dir() {
    local start_dir="$1"
    local dir
    dir="$(cd "${start_dir}" && pwd)"

    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/requirements.txt" && -f "${dir}/dataset_generation/scripts/dataset_generation.py" ]]; then
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

DATASET_GENERATION_DIR="${REPO_DIR}/dataset_generation"
SCRIPT_PATH="${DATASET_GENERATION_DIR}/scripts/dataset_generation.py"
LOG_DIR="${DATASET_GENERATION_DIR}/logs"
ENV_NAME="purifying_poison"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
MAIN_LOG="${LOG_DIR}/poison_pipeline_${JOB_ID}.log"
ERR_LOG="${LOG_DIR}/poison_pipeline_err_${JOB_ID}.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${MAIN_LOG}") 2> >(tee -a "${ERR_LOG}" >&2)

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Repository: ${REPO_DIR}"
echo "Logs directory: ${LOG_DIR}"
echo "Main log: ${MAIN_LOG}"
echo "Error log: ${ERR_LOG}"
echo "Started at: $(date -Is)"

# Load conda environment
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is required but was not found on PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y --name "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"
python -m pip install -r "${REPO_DIR}/requirements.txt"

export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"

# Run the single MAIN script to execute both Setup and Poison Generation
echo "=============================="
echo "1. PREPARING CLEAN DATASETS..."
echo "=============================="
python -u "${SCRIPT_PATH}" --mode setup_clean

echo "=============================="
echo "2. GENERATING WTICHES BREW POISONS..."
echo "=============================="
python -u "${SCRIPT_PATH}" --mode craft_wb

echo "=============================="
echo "3. GENERATING BULLSEYE POLYTOPE POISONS..."
echo "=============================="
python -u "${SCRIPT_PATH}" --mode craft_bp

echo "=============================="
echo "DONE! Results are in ${DATASET_GENERATION_DIR}/datasets/"
echo "Finished at: $(date -Is)"
echo "=============================="
