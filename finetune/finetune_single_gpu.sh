#!/bin/bash
set -e
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Defaults (override via env or edit here)
SAVE_NAME=${SAVE_NAME:-sft_deepencoder_single}
MAX_LEN=${MAX_LEN:-704}
MB_SIZE=${MB_SIZE:-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
EPOCHS=${EPOCHS:-3}
LR=${LR:-2e-5}
GA_STEPS=${GA_STEPS:-8}
DATA_PATH=${DATA_PATH:-data/mind2web_train_sft.json}
QWEN_PATH=${QWEN_PATH:-Qwen/Qwen-VL-Chat}
MODEL_PATH=${MODEL_PATH:-SeeClick-pretrain}
OUT_DIR=${OUT_DIR:-checkpoint_qwen}

# Deepencoder params (paths should be absolute or relative)
REPLACE_ENCODER=${REPLACE_ENCODER:-true}
DEEPENCODER_PATH=${DEEPENCODER_PATH:-/path/to/DeepSeek-OCR/DeepSeek-OCR-vllm}
SAM_CKPT=${SAM_CKPT:-}
CLIP_CKPT=${CLIP_CKPT:-}
PROJ_CKPT=${PROJ_CKPT:-}
FREEZE_SAM=${FREEZE_SAM:-true}
FREEZE_CLIP=${FREEZE_CLIP:-false}

mkdir -p "$OUT_DIR/$SAVE_NAME"

torchrun --nproc_per_node 1 finetune/finetune.py \
  --model_name_or_path "$MODEL_PATH" \
  --qwen_path "$QWEN_PATH" \
  --data_path "$DATA_PATH" \
  --bf16 True \
  --fix_vit False \
  --output_dir "$OUT_DIR/$SAVE_NAME" \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${MB_SIZE} \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps ${GA_STEPS} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps ${SAVE_INTERVAL} \
  --save_total_limit 10 \
  --learning_rate ${LR} \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --report_to "tensorboard" \
  --model_max_length ${MAX_LEN} \
  --lazy_preprocess True \
  --use_lora \
  --gradient_checkpointing \
  --deepspeed finetune/ds_config_zero2.json \
  $( [ "$REPLACE_ENCODER" = true ] && echo --replace_encoder ) \
  --deepencoder_path "$DEEPENCODER_PATH" \
  --sam_checkpoint "$SAM_CKPT" \
  --clip_checkpoint "$CLIP_CKPT" \
  --projector_checkpoint "$PROJ_CKPT" \
  --freeze_sam ${FREEZE_SAM} \
  --freeze_clip ${FREEZE_CLIP}

echo "Training done. Output: $OUT_DIR/$SAVE_NAME"


