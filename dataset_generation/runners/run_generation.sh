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
ENV_NAME="${ENV_NAME:-purifying_poison}"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
MAIN_LOG="${LOG_DIR}/poison_pipeline_${JOB_ID}.log"
ERR_LOG="${LOG_DIR}/poison_pipeline_err_${JOB_ID}.log"
RUN_WB="${RUN_WB:-1}"
RUN_BP="${RUN_BP:-1}"
RUN_NARCISSUS="${RUN_NARCISSUS:-1}"
if [[ -n "${NARCISSUS_POOD_ROOT:-}" ]]; then
    TINY_IMAGENET_ROOT="${TINY_IMAGENET_ROOT:-$(dirname "${NARCISSUS_POOD_ROOT}")}"
    TINY_IMAGENET_PARENT="${TINY_IMAGENET_PARENT:-$(dirname "${TINY_IMAGENET_ROOT}")}"
elif [[ -n "${TINY_IMAGENET_ROOT:-}" ]]; then
    TINY_IMAGENET_PARENT="${TINY_IMAGENET_PARENT:-$(dirname "${TINY_IMAGENET_ROOT}")}"
    NARCISSUS_POOD_ROOT="${TINY_IMAGENET_ROOT}/train"
else
    TINY_IMAGENET_PARENT="${TINY_IMAGENET_PARENT:-${DATASET_GENERATION_DIR}/datasets}"
    TINY_IMAGENET_ROOT="${TINY_IMAGENET_ROOT:-${TINY_IMAGENET_PARENT}/tiny-imagenet-200}"
    NARCISSUS_POOD_ROOT="${TINY_IMAGENET_ROOT}/train"
fi
TINY_IMAGENET_URL="${TINY_IMAGENET_URL:-http://cs231n.stanford.edu/tiny-imagenet-200.zip}"
NS_CLASSES="${NS_CLASSES:-all}"
NS_STAGE="${NS_STAGE:-all}"
NS_PROFILE="${NS_PROFILE:-final}"
NS_TRIGGER_KIND="${NS_TRIGGER_KIND:-both}"
NS_CHECKPOINT_INTERVAL="${NS_CHECKPOINT_INTERVAL:-10}"
NS_BATCH_SIZE="${NS_BATCH_SIZE:-350}"
NS_NUM_WORKERS="${NS_NUM_WORKERS:-8}"

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
echo "SLURM job GPUs: ${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM step GPUs: ${SLURM_STEP_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES before setup: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Narcissus: run=${RUN_NARCISSUS}, stage=${NS_STAGE}, classes=${NS_CLASSES}, profile=${NS_PROFILE}"
echo "Tiny ImageNet root: ${TINY_IMAGENET_ROOT}"
echo "Narcissus POOD root: ${NARCISSUS_POOD_ROOT}"

tiny_imagenet_class_count() {
    if [[ ! -d "${NARCISSUS_POOD_ROOT}" ]]; then
        echo 0
        return
    fi
    find "${NARCISSUS_POOD_ROOT}" -mindepth 1 -maxdepth 1 -type d | wc -l
}

tiny_imagenet_image_count() {
    if [[ ! -d "${NARCISSUS_POOD_ROOT}" ]]; then
        echo 0
        return
    fi
    find "${NARCISSUS_POOD_ROOT}" -type f -name '*.JPEG' | wc -l
}

tiny_imagenet_is_valid() {
    local class_count
    local image_count
    class_count="$(tiny_imagenet_class_count)"
    image_count="$(tiny_imagenet_image_count)"
    [[ "${class_count}" -eq 200 && "${image_count}" -eq 100000 ]]
}

ensure_tiny_imagenet() {
    if tiny_imagenet_is_valid; then
        echo "Tiny ImageNet already installed; skipping download (${NARCISSUS_POOD_ROOT}, 200 classes, 100000 images)."
        return
    fi

    local archive_path
    archive_path="${TINY_IMAGENET_PARENT}/tiny-imagenet-200.zip"
    mkdir -p "${TINY_IMAGENET_PARENT}"

    echo "Tiny ImageNet is missing or incomplete at ${NARCISSUS_POOD_ROOT}."
    echo "Downloading ${TINY_IMAGENET_URL} to ${archive_path}."
    if command -v wget >/dev/null 2>&1; then
        wget --continue --tries=5 --timeout=60 --output-document="${archive_path}" "${TINY_IMAGENET_URL}"
    elif command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 5 --continue-at - \
            "${TINY_IMAGENET_URL}" --output "${archive_path}"
    else
        echo "ERROR: Tiny ImageNet installation requires wget or curl." >&2
        exit 1
    fi

    if ! command -v unzip >/dev/null 2>&1; then
        echo "ERROR: Tiny ImageNet installation requires unzip." >&2
        exit 1
    fi
    echo "Extracting Tiny ImageNet under ${TINY_IMAGENET_PARENT}."
    unzip -q -o "${archive_path}" -d "${TINY_IMAGENET_PARENT}"

    if ! tiny_imagenet_is_valid; then
        echo "ERROR: Tiny ImageNet installation is invalid." >&2
        echo "Expected 200 class directories and 100000 JPEG images under ${NARCISSUS_POOD_ROOT}." >&2
        echo "Found $(tiny_imagenet_class_count) classes and $(tiny_imagenet_image_count) JPEG images." >&2
        exit 1
    fi
    echo "Tiny ImageNet installation complete: ${NARCISSUS_POOD_ROOT} (200 classes, 100000 images)."
}

if [[ "${RUN_NARCISSUS}" == "1" ]]; then
    case "${NS_STAGE}" in
        triggers|bank|eval|validate|all) ;;
        *) echo "ERROR: NS_STAGE must be triggers, bank, eval, validate, or all." >&2; exit 1 ;;
    esac
    if [[ "${NS_STAGE}" == "triggers" || "${NS_STAGE}" == "all" ]]; then
        ensure_tiny_imagenet
    fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

# Load conda environment
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

REQ_NO_TORCH="$(mktemp)"
PIP_CONSTRAINTS="$(mktemp)"
trap 'rm -f "${REQ_NO_TORCH}" "${PIP_CONSTRAINTS}"' EXIT
grep -vE '^[[:space:]]*(torch|torchvision|diffusers|transformers|accelerate|huggingface_hub)([<>=!~ ].*)?$' "${REPO_DIR}/requirements.txt" > "${REQ_NO_TORCH}"
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
import sys
import numpy as np
import torch
import torchvision
from torch import nn

print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA_VISIBLE_DEVICES: {__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("ERROR: PyTorch cannot access CUDA; refusing to run slow CPU poison generation.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
torch.from_numpy(np.zeros((1,), dtype=np.float32))
try:
    tensor = torch.zeros((1,), device="cuda")
    model = nn.Linear(1, 1).to("cuda")
    model(tensor.reshape(1, 1))
    torch.cuda.synchronize()
except RuntimeError as exc:
    sys.exit(f"ERROR: CUDA allocation test failed before poison generation: {exc}")
PY

export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"

# Run the single MAIN script to execute both Setup and Poison Generation
echo "=============================="
echo "1. PREPARING CLEAN DATASETS..."
echo "=============================="
"${ENV_PYTHON}" -u "${SCRIPT_PATH}" --mode setup_clean

echo "=============================="
echo "2. GENERATING WITCHES BREW POISONS..."
echo "=============================="
if [[ "${RUN_WB}" == "1" ]]; then
    "${ENV_PYTHON}" -u "${SCRIPT_PATH}" --mode craft_wb
fi

echo "=============================="
echo "3. GENERATING BULLSEYE POLYTOPE POISONS..."
echo "=============================="
if [[ "${RUN_BP}" == "1" ]]; then
    "${ENV_PYTHON}" -u "${SCRIPT_PATH}" --mode craft_bp
fi

if [[ "${RUN_NARCISSUS}" == "1" ]]; then
    case "${NS_STAGE}" in
        triggers) NS_MODE="craft_ns_triggers" ;;
        bank) NS_MODE="craft_ns_bank" ;;
        eval) NS_MODE="craft_ns_eval" ;;
        validate) NS_MODE="validate_ns" ;;
        all) NS_MODE="craft_ns" ;;
        *) echo "ERROR: NS_STAGE must be triggers, bank, eval, validate, or all." >&2; exit 1 ;;
    esac
    if [[ "${NS_STAGE}" == "triggers" || "${NS_STAGE}" == "all" ]]; then
        if ! tiny_imagenet_is_valid; then
            echo "ERROR: Tiny ImageNet became unavailable at ${NARCISSUS_POOD_ROOT}." >&2
            exit 1
        fi
        NS_POOD_ARGS=(--pood-root "${NARCISSUS_POOD_ROOT}")
    else
        NS_POOD_ARGS=()
    fi
    echo "=============================="
    echo "4. RUNNING NARCISSUS STAGE ${NS_STAGE}..."
    echo "=============================="
    "${ENV_PYTHON}" -u "${SCRIPT_PATH}" \
        --mode "${NS_MODE}" \
        --ns-classes "${NS_CLASSES}" \
        --ns-profile "${NS_PROFILE}" \
        --ns-trigger-kind "${NS_TRIGGER_KIND}" \
        --ns-device cuda \
        --ns-checkpoint-interval "${NS_CHECKPOINT_INTERVAL}" \
        --ns-batch-size "${NS_BATCH_SIZE}" \
        --ns-num-workers "${NS_NUM_WORKERS}" \
        "${NS_POOD_ARGS[@]}"
fi

echo "=============================="
echo "DONE! Results are in ${DATASET_GENERATION_DIR}/datasets/"
echo "Finished at: $(date -Is)"
echo "=============================="
