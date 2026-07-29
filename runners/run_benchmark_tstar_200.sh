#!/bin/bash
#SBATCH --job-name=cm_bench_t200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

# Purpose: Walk upward from a directory until the project root is found.
# Input: starting directory path.
# Output: repository root on stdout, or non-zero when no root is found.
find_repo_dir() {
    local start_dir="$1"
    local dir
    dir="$(cd "${start_dir}" && pwd)"
    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/requirements.txt" && -f "${dir}/benchmark/run_benchmark.sh" ]]; then
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
    WRAPPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(find_repo_dir "${WRAPPER_DIR}")"
fi
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_DIR}/consistency_model/checkpoints/cm_purifier_lpips.pth}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "ERROR: missing LPIPS purifier checkpoint: ${CHECKPOINT_PATH}" >&2
    echo "Submit consistency_model/run_cm_purifier_training_lpips.sh first." >&2
    exit 1
fi

export T_STAR=200
export BATCH_SIZE="${BATCH_SIZE:-64}"
export SEED="${SEED:-2026}"
export RUN_ID="${RUN_ID:-tstar_200_slurm_${SLURM_JOB_ID:-local}}"

# Execute the benchmark in this allocation; never submit a nested Slurm job.
exec bash "${REPO_DIR}/benchmark/run_benchmark.sh"
