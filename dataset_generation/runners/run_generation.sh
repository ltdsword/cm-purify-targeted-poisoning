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
echo "SLURM job GPUs: ${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM step GPUs: ${SLURM_STEP_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES before setup: ${CUDA_VISIBLE_DEVICES:-<unset>}"

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
"${ENV_PYTHON}" -u "${SCRIPT_PATH}" --mode craft_wb

echo "=============================="
echo "3. GENERATING BULLSEYE POLYTOPE POISONS..."
echo "=============================="
"${ENV_PYTHON}" -u "${SCRIPT_PATH}" --mode craft_bp

echo "=============================="
echo "DONE! Results are in ${DATASET_GENERATION_DIR}/datasets/"
echo "Finished at: $(date -Is)"
echo "=============================="
