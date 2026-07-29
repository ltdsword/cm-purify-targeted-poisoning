#!/bin/bash
#SBATCH --job-name=cm_lpips_tstar_sweep
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
        if [[ -f "${dir}/requirements.txt" && -f "${dir}/runners/run_train_then_benchmark.sh" ]]; then
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

RUNNER_DIR="${REPO_DIR}/runners"
JOB_TOKEN="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
SWEEP_RUN_ID="${SWEEP_RUN_ID:-slurm_${JOB_TOKEN}}"

# Train the LPIPS strategy once and preserve the original baseline checkpoint.
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_DIR}/consistency_model/checkpoints/cm_purifier_lpips.pth}"
export LAMBDA_LPIPS="${LAMBDA_LPIPS:-0.10}"
export LPIPS_NET="${LPIPS_NET:-alex}"
export LPIPS_IMAGE_SIZE="${LPIPS_IMAGE_SIZE:-64}"
export LPIPS_WARMUP_STEPS="${LPIPS_WARMUP_STEPS:-2000}"

# Run all requested timesteps sequentially under one parent output directory.
export T_STARS="${T_STARS:-150 200}"
export BENCHMARK_BATCH_SIZE="${BENCHMARK_BATCH_SIZE:-64}"
export BENCHMARK_SEED="${BENCHMARK_SEED:-2026}"
export BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${REPO_DIR}/benchmark/outputs/${SWEEP_RUN_ID}}"

# Execute the complete pipeline in this allocation; never submit a nested job.
exec bash "${RUNNER_DIR}/run_train_then_benchmark.sh"
