#!/bin/bash
#SBATCH --job-name=cm_purifier_lpips
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
        if [[ -f "${dir}/requirements.txt" && -f "${dir}/consistency_model/run_cm_purifier_training.sh" ]]; then
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

CONSISTENCY_DIR="${REPO_DIR}/consistency_model"

# Keep the original checkpoint as a baseline and enable the reviewed LPIPS setup.
export OUTPUT_PATH="${OUTPUT_PATH:-${CONSISTENCY_DIR}/checkpoints/cm_purifier_lpips.pth}"
export LAMBDA_LPIPS="${LAMBDA_LPIPS:-0.10}"
export LPIPS_NET="${LPIPS_NET:-alex}"
export LPIPS_IMAGE_SIZE="${LPIPS_IMAGE_SIZE:-64}"
export LPIPS_WARMUP_STEPS="${LPIPS_WARMUP_STEPS:-2000}"

# Run inside this allocation. The base runner is executed with bash, not sbatch.
exec bash "${CONSISTENCY_DIR}/run_cm_purifier_training.sh"
