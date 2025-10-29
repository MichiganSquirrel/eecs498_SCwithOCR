#!/bin/bash

#SBATCH --account=eecs498f25s007_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=07:55:00
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=seeclick_deepenc_ft
#SBATCH --output=/scratch/eecs498f25s007_class_root/eecs498f25s007_class/%u/seeclick_runs/%x-%j.out

set -euo pipefail

# ---------- user-config (edit these or override via sbatch --export) ----------
REPO_DIR=${REPO_DIR:-/home/yikun/eecs498/eecs498_SCwithOCR}
# Path to DeepSeek-OCR codebase containing "DeepSeek-OCR-vllm"
DEEPENCODER_PATH=${DEEPENCODER_PATH:-"${REPO_DIR}/external/DeepSeek-OCR/DeepSeek-OCR-vllm"}

# Data/model configuration
QWEN_PATH=${QWEN_PATH:-"Qwen/Qwen-VL-Chat"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen-VL-Chat"}   # or your local SeeClick-pretrain path
DATA_PATH=${DATA_PATH:-"/abs/path/to/mind2web_train_sft.json"}

# Training output to scratch
SCRATCH_BASE=${SCRATCH_BASE:-/scratch/eecs498f25s007_class_root/eecs498f25s007_class/${USER}/seeclick_runs}
OUT_DIR=${OUT_DIR:-${SCRATCH_BASE}/checkpoints}
SAVE_NAME=${SAVE_NAME:-sft_deepencoder_single}

# Optional training knobs
MAX_LEN=${MAX_LEN:-704}
MB_SIZE=${MB_SIZE:-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
EPOCHS=${EPOCHS:-3}
LR=${LR:-2e-5}
GA_STEPS=${GA_STEPS:-8}

# ------------------------------------------------

mkdir -p "${SCRATCH_BASE}" "${OUT_DIR}/${SAVE_NAME}"
echo "[INFO] Job scratch: ${SCRATCH_BASE}"

# Load modules (Anaconda + CUDA 11.8 for cu118)
module load python3.10-anaconda/2023.03
module load cuda/11.8.0

# Enable conda in batch shell
eval "$(conda shell.bash hook)"

# Create/activate env
ENV_NAME=seeclick-env
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"

# Install PyTorch GPU stack (cu118)
pip install --index-url https://download.pytorch.org/whl/cu118 \
  "torch==2.1.0+cu118" "torchvision==0.16.0+cu118" "torchaudio==2.1.0+cu118" --force-reinstall

# Install project deps (filter out conflicting GPU stack lines)
REQ_SRC="${REPO_DIR}/requirements_agent.txt"
REQ_TMP="${SCRATCH_BASE}/req_agent_filtered.txt"
grep -v -E '^(deepspeed==|nvidia-.*|triton==|torch==|torchvision==|torchaudio==)' "${REQ_SRC}" > "${REQ_TMP}"
pip install -r "${REQ_TMP}"

# Build/install DeepSpeed (requires nvcc)
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
pip install deepspeed==0.12.5

# Extract deepencoder weights if not present
ENC_DIR="${REPO_DIR}/encoder_weights"
if [ ! -f "${ENC_DIR}/sam_encoder.pth" ] || [ ! -f "${ENC_DIR}/clip_encoder.pth" ] || [ ! -f "${ENC_DIR}/projector.pth" ]; then
  mkdir -p "${ENC_DIR}"
  python "${REPO_DIR}/scripts/extract_deepencoder_weights.py" \
    --model deepseek-ai/DeepSeek-OCR \
    --out "${ENC_DIR}"
fi

# Ensure DeepSeek-OCR code present for adapter import
if [ ! -d "${DEEPENCODER_PATH}" ]; then
  echo "[INFO] DeepSeek-OCR not found at ${DEEPENCODER_PATH}, cloning..."
  mkdir -p "$(dirname "${DEEPENCODER_PATH}")"
  git clone --depth=1 https://github.com/deepseek-ai/DeepSeek-OCR.git "${REPO_DIR}/external/DeepSeek-OCR" || true
  if [ ! -d "${DEEPENCODER_PATH}" ]; then
    echo "[ERROR] DeepSeek-OCR code not found after clone. Set DEEPENCODER_PATH to your local path."
    exit 1
  fi
fi

# Quick device self-check
python - << 'PY'
import torch
print('[CHECK] torch:', torch.__version__, '| cuda:', torch.version.cuda, '| available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[CHECK] device:', torch.cuda.get_device_name(0))
else:
    raise SystemExit('[ERROR] No GPU visible in this job. Check partition/gpus settings.')
PY

cd "${REPO_DIR}"

# Launch finetuning with deepencoder replacement
DEEPENCODER_PATH="${DEEPENCODER_PATH}" \
SAM_CKPT="${ENC_DIR}/sam_encoder.pth" \
CLIP_CKPT="${ENC_DIR}/clip_encoder.pth" \
PROJ_CKPT="${ENC_DIR}/projector.pth" \
MODEL_PATH="${MODEL_PATH}" \
QWEN_PATH="${QWEN_PATH}" \
DATA_PATH="${DATA_PATH}" \
OUT_DIR="${OUT_DIR}" \
SAVE_NAME="${SAVE_NAME}" \
MAX_LEN=${MAX_LEN} \
MB_SIZE=${MB_SIZE} \
SAVE_INTERVAL=${SAVE_INTERVAL} \
EPOCHS=${EPOCHS} \
LR=${LR} \
GA_STEPS=${GA_STEPS} \
bash finetune/finetune_single_gpu.sh

echo "[DONE] Output: ${OUT_DIR}/${SAVE_NAME}"


