#!/bin/bash
#SBATCH --job-name=cm_train_then_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
        if [[ -f "${dir}/requirements.txt" && -f "${dir}/consistency_model/run_cm_purifier_training.sh" && -f "${dir}/benchmark/run_benchmark.sh" ]]; then
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

RUNNERS_DIR="${REPO_DIR}/runners"
CONSISTENCY_DIR="${REPO_DIR}/consistency_model"
BENCHMARK_DIR="${REPO_DIR}/benchmark"
TRAIN_SCRIPT="${CONSISTENCY_DIR}/run_cm_purifier_training.sh"
BENCHMARK_SCRIPT="${BENCHMARK_DIR}/run_benchmark.sh"

PAIR_DIR="${PAIR_DIR:-${REPO_DIR}/dataset_generation/datasets/train}"
TEST_DIR="${TEST_DIR:-${REPO_DIR}/dataset_generation/datasets/test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CONSISTENCY_DIR}/checkpoints/cm_purifier.pth}"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${BENCHMARK_DIR}/outputs}"
ENV_NAME="${ENV_NAME:-purifying_poison}"
ROOT_REQUIREMENTS="${ROOT_REQUIREMENTS:-${REPO_DIR}/requirements.txt}"
CM_OUTPUT_MODE="${CM_OUTPUT_MODE:-full_boundary}"
LOG_DIR="${RUNNERS_DIR}/logs"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
MAIN_LOG="${LOG_DIR}/train_then_benchmark_${JOB_ID}.log"
ERR_LOG="${LOG_DIR}/train_then_benchmark_err_${JOB_ID}.log"

RUN_ID="${RUN_ID:-}"
ATTACK_FILTER="${ATTACK_FILTER:-all}"
CASE_FILTER="${CASE_FILTER:-}"
MAX_CASES="${MAX_CASES:-}"
T_STAR="${T_STAR:-200}"
BENCHMARK_BATCH_SIZE="${BENCHMARK_BATCH_SIZE:-64}"
BENCHMARK_SEED="${BENCHMARK_SEED:-2026}"
BENCHMARK_LOG_STEPS="${BENCHMARK_LOG_STEPS:-1024}"
OVERWRITE_ARTIFACTS="${OVERWRITE_ARTIFACTS:-0}"
WB_EPOCHS="${WB_EPOCHS:-}"
WB_DRYRUN="${WB_DRYRUN:-0}"
BP_VICTIM_NET="${BP_VICTIM_NET:-ResNet18}"
BP_CHECKPOINT_NAME="${BP_CHECKPOINT_NAME:-ckpt-%s-4800.t7}"
BP_RETRAIN_EPOCHS="${BP_RETRAIN_EPOCHS:-60}"
BP_RETRAIN_BSIZE="${BP_RETRAIN_BSIZE:-64}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${MAIN_LOG}") 2> >(tee -a "${ERR_LOG}" >&2)

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Repository: ${REPO_DIR}"
echo "Training script: ${TRAIN_SCRIPT}"
echo "Benchmark script: ${BENCHMARK_SCRIPT}"
echo "Pair directory: ${PAIR_DIR}"
echo "Test directory: ${TEST_DIR}"
echo "Checkpoint path: ${CHECKPOINT_PATH}"
echo "CM output mode: ${CM_OUTPUT_MODE}"
echo "Benchmark output directory: ${BENCHMARK_OUTPUT_DIR}"
echo "Conda environment: ${ENV_NAME}"
echo "Logs directory: ${LOG_DIR}"
echo "Main log: ${MAIN_LOG}"
echo "Error log: ${ERR_LOG}"
echo "Started at: $(date -Is)"

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

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is required for this chained Slurm runner." >&2
    exit 1
fi

echo "=============================="
echo "1. SUBMITTING ALGORITHM 2 TRAINING..."
echo "=============================="
TRAIN_SUBMIT="$(
    sbatch \
        --export=ALL,PAIR_DIR="${PAIR_DIR}",OUTPUT_PATH="${CHECKPOINT_PATH}",ENV_NAME="${ENV_NAME}",ROOT_REQUIREMENTS="${ROOT_REQUIREMENTS}",CM_OUTPUT_MODE="${CM_OUTPUT_MODE}" \
        "${TRAIN_SCRIPT}"
)"
echo "${TRAIN_SUBMIT}"
TRAIN_JOB_ID="$(echo "${TRAIN_SUBMIT}" | awk '{print $NF}')"
if [[ -z "${TRAIN_JOB_ID}" ]]; then
    echo "ERROR: could not parse training Slurm job id from: ${TRAIN_SUBMIT}" >&2
    exit 1
fi

echo "=============================="
echo "2. SUBMITTING BENCHMARK AFTER TRAINING..."
echo "=============================="
BENCHMARK_SUBMIT="$(
    sbatch \
        --dependency=afterok:"${TRAIN_JOB_ID}" \
        --export=ALL,CHECKPOINT_PATH="${CHECKPOINT_PATH}",TEST_DIR="${TEST_DIR}",OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR}",RUN_ID="${RUN_ID}",ENV_NAME="${ENV_NAME}",ROOT_REQUIREMENTS="${ROOT_REQUIREMENTS}",ATTACK_FILTER="${ATTACK_FILTER}",CASE_FILTER="${CASE_FILTER}",MAX_CASES="${MAX_CASES}",T_STAR="${T_STAR}",BATCH_SIZE="${BENCHMARK_BATCH_SIZE}",SEED="${BENCHMARK_SEED}",LOG_STEPS="${BENCHMARK_LOG_STEPS}",SKIP_PURIFY=0,SKIP_RETRAIN=0,OVERWRITE_ARTIFACTS="${OVERWRITE_ARTIFACTS}",WB_EPOCHS="${WB_EPOCHS}",WB_DRYRUN="${WB_DRYRUN}",BP_VICTIM_NET="${BP_VICTIM_NET}",BP_CHECKPOINT_NAME="${BP_CHECKPOINT_NAME}",BP_RETRAIN_EPOCHS="${BP_RETRAIN_EPOCHS}",BP_RETRAIN_BSIZE="${BP_RETRAIN_BSIZE}" \
        "${BENCHMARK_SCRIPT}"
)"
echo "${BENCHMARK_SUBMIT}"
BENCHMARK_JOB_ID="$(echo "${BENCHMARK_SUBMIT}" | awk '{print $NF}')"
if [[ -z "${BENCHMARK_JOB_ID}" ]]; then
    echo "ERROR: could not parse benchmark Slurm job id from: ${BENCHMARK_SUBMIT}" >&2
    exit 1
fi

echo "=============================="
echo "CHAIN SUBMITTED"
echo "Training job:  ${TRAIN_JOB_ID}"
echo "Benchmark job: ${BENCHMARK_JOB_ID} depends on afterok:${TRAIN_JOB_ID}"
echo "Finished at: $(date -Is)"
echo "=============================="
