#!/bin/bash
#SBATCH --job-name=cm_benchmark
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
        if [[ -f "${dir}/requirements.txt" && -f "${dir}/benchmark/run_benchmark.py" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done

    return 1
}

# Purpose: Submit CM training and then submit this benchmark runner as a dependent Slurm job.
# Input: training runner path and this runner path.
# Output: Slurm jobs are submitted; current job exits before benchmark begins.
submit_training_then_benchmark() {
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

    echo "Submitting benchmark after training job ${train_job_id} succeeds."
    sbatch \
        --dependency=afterok:"${train_job_id}" \
        --export=ALL,CHECKPOINT_PATH="${CHECKPOINT_PATH}",TEST_DIR="${TEST_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",RUN_ID="${RUN_ID}",ENV_NAME="${ENV_NAME}",ROOT_REQUIREMENTS="${ROOT_REQUIREMENTS}",ATTACK_FILTER="${ATTACK_FILTER}",CASE_FILTER="${CASE_FILTER}",MAX_CASES="${MAX_CASES}",T_STAR="${T_STAR}",BATCH_SIZE="${BATCH_SIZE}",SEED="${SEED}",LOG_STEPS="${LOG_STEPS}",SKIP_PURIFY="${SKIP_PURIFY}",SKIP_RETRAIN="${SKIP_RETRAIN}",OVERWRITE_ARTIFACTS="${OVERWRITE_ARTIFACTS}",WB_EPOCHS="${WB_EPOCHS}",WB_DRYRUN="${WB_DRYRUN}",BP_VICTIM_NET="${BP_VICTIM_NET}",BP_CHECKPOINT_NAME="${BP_CHECKPOINT_NAME}",BP_RETRAIN_EPOCHS="${BP_RETRAIN_EPOCHS}",BP_RETRAIN_BSIZE="${BP_RETRAIN_BSIZE}" \
        "${self_script}"
    echo "Exiting current benchmark job; dependent benchmark job will run after training."
    exit 0
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && REPO_DIR="$(find_repo_dir "${SLURM_SUBMIT_DIR}")"; then
    :
else
    RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(find_repo_dir "${RUNNER_DIR}")"
fi

BENCHMARK_DIR="${REPO_DIR}/benchmark"
CONSISTENCY_DIR="${REPO_DIR}/consistency_model"
TRAIN_SCRIPT="${CONSISTENCY_DIR}/run_cm_purifier_training.sh"
SELF_SCRIPT="${BENCHMARK_DIR}/run_benchmark.sh"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CONSISTENCY_DIR}/checkpoints/cm_purifier.pth}"
TEST_DIR="${TEST_DIR:-${REPO_DIR}/dataset_generation/datasets/test}"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCHMARK_DIR}/outputs}"
RUN_ID="${RUN_ID:-}"
ENV_NAME="${ENV_NAME:-purifying_poison}"
ROOT_REQUIREMENTS="${ROOT_REQUIREMENTS:-${REPO_DIR}/requirements.txt}"
ATTACK_FILTER="${ATTACK_FILTER:-all}"
CASE_FILTER="${CASE_FILTER:-}"
MAX_CASES="${MAX_CASES:-}"
T_STAR="${T_STAR:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-2026}"
LOG_STEPS="${LOG_STEPS:-1024}"
SKIP_PURIFY="${SKIP_PURIFY:-0}"
SKIP_RETRAIN="${SKIP_RETRAIN:-0}"
OVERWRITE_ARTIFACTS="${OVERWRITE_ARTIFACTS:-0}"
WB_EPOCHS="${WB_EPOCHS:-}"
WB_DRYRUN="${WB_DRYRUN:-0}"
BP_VICTIM_NET="${BP_VICTIM_NET:-ResNet18}"
BP_CHECKPOINT_NAME="${BP_CHECKPOINT_NAME:-ckpt-%s-4800.t7}"
BP_RETRAIN_EPOCHS="${BP_RETRAIN_EPOCHS:-60}"
BP_RETRAIN_BSIZE="${BP_RETRAIN_BSIZE:-64}"
LOG_DIR="${BENCHMARK_DIR}/logs"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
MAIN_LOG="${LOG_DIR}/benchmark_${JOB_ID}.log"
ERR_LOG="${LOG_DIR}/benchmark_err_${JOB_ID}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
exec > >(tee -a "${MAIN_LOG}") 2> >(tee -a "${ERR_LOG}" >&2)

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Repository: ${REPO_DIR}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Test directory: ${TEST_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Run ID: ${RUN_ID:-<auto>}"
echo "Conda environment: ${ENV_NAME}"
echo "Root requirements: ${ROOT_REQUIREMENTS}"
echo "Logs directory: ${LOG_DIR}"
echo "Main log: ${MAIN_LOG}"
echo "Error log: ${ERR_LOG}"
echo "Started at: $(date -Is)"
echo "SLURM job GPUs: ${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM step GPUs: ${SLURM_STEP_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES before setup: ${CUDA_VISIBLE_DEVICES:-<unset>}"

if [[ ! -f "${CHECKPOINT_PATH}" && "${SKIP_PURIFY}" != "1" ]]; then
    submit_training_then_benchmark "${TRAIN_SCRIPT}" "${SELF_SCRIPT}"
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
    sys.exit("ERROR: PyTorch cannot access CUDA; refusing to run benchmark on CPU.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
try:
    tensor = torch.zeros((1,), device="cuda")
    torch.cuda.synchronize()
except RuntimeError as exc:
    sys.exit(f"ERROR: CUDA allocation test failed before benchmark: {exc}")
PY

cd "${REPO_DIR}"

BENCHMARK_ARGS=(
    -u -m benchmark.run_benchmark
    --checkpoint "${CHECKPOINT_PATH}"
    --test-dir "${TEST_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --attack-filter "${ATTACK_FILTER}"
    --t-star "${T_STAR}"
    --batch-size "${BATCH_SIZE}"
    --device cuda
    --seed "${SEED}"
    --log-steps "${LOG_STEPS}"
    --bp-victim-net "${BP_VICTIM_NET}"
    --bp-checkpoint-name "${BP_CHECKPOINT_NAME}"
    --bp-retrain-epochs "${BP_RETRAIN_EPOCHS}"
    --bp-retrain-bsize "${BP_RETRAIN_BSIZE}"
)

if [[ -n "${RUN_ID}" ]]; then
    BENCHMARK_ARGS+=(--run-id "${RUN_ID}")
fi
if [[ -n "${CASE_FILTER}" ]]; then
    BENCHMARK_ARGS+=(--case-filter "${CASE_FILTER}")
fi
if [[ -n "${MAX_CASES}" ]]; then
    BENCHMARK_ARGS+=(--max-cases "${MAX_CASES}")
fi
if [[ "${SKIP_PURIFY}" == "1" ]]; then
    BENCHMARK_ARGS+=(--skip-purify)
fi
if [[ "${SKIP_RETRAIN}" == "1" ]]; then
    BENCHMARK_ARGS+=(--skip-retrain)
fi
if [[ "${OVERWRITE_ARTIFACTS}" == "1" ]]; then
    BENCHMARK_ARGS+=(--overwrite-artifacts)
fi
if [[ -n "${WB_EPOCHS}" ]]; then
    BENCHMARK_ARGS+=(--wb-epochs "${WB_EPOCHS}")
fi
if [[ "${WB_DRYRUN}" == "1" ]]; then
    BENCHMARK_ARGS+=(--wb-dryrun)
fi

echo "=============================="
echo "RUNNING CM PURIFICATION BENCHMARK..."
echo "=============================="
"${ENV_PYTHON}" "${BENCHMARK_ARGS[@]}"

echo "=============================="
echo "DONE! Benchmark outputs are in ${OUTPUT_DIR}"
echo "Finished at: $(date -Is)"
echo "=============================="
