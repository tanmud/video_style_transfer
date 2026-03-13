#!/bin/bash

ONLY_MOTION=false
for arg in "$@"; do
  if [ "$arg" == "--only-motion" ]; then
    ONLY_MOTION=true
    echo "WARNING: Only training motion (expecting already trained spatial)"
  fi
done

if [ "$ONLY_MOTION" = false ]; then
  bash train.sh
fi

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export MOTION_ADAPTER="guoyww/animatediff-motion-adapter-sdxl-beta"

# WandB
export WANDB_NAME="animatediff_male_biker"
export WANDB_MODE="offline"

# Data
export INSTANCE_DIR="/work/10572/tmudali/vista/video_style_transfer/instance_videos/male_biker"
export OUTPUT_DIR="models/male_biker_video"
export NUM_FRAMES=8
export RESOLUTION=1024

# Stage-1 UnZipLoRA outputs (required for Stage-2)
export UNZIPLORA_CONTENT="models/male_biker_image/male_biker_image_content"
export UNZIPLORA_STYLE="models/male_biker_image/male_biker_image_style"
export UNZIPLORA_CONTENT_WEIGHTS="models/male_biker_image/male_biker_image_merger_content.pth"
export UNZIPLORA_STYLE_WEIGHTS="models/male_biker_image/male_biker_image_merger_style.pth"

# Training
export STEPS=4000
export LEARNING_RATE=1e-5
export PROMPT="A male biker in cartoon style biking on the street"
export GRAD_ACC_STEPS=4
export MIXED_PRECISION="bf16"
export WARMUP_STEPS=100

accelerate launch --mixed_precision=$MIXED_PRECISION train_animatediff.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --name=$WANDB_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --unziplora_content_path="${UNZIPLORA_CONTENT}" \
  --unziplora_style_path="${UNZIPLORA_STYLE}" \
  --unziplora_content_weight_path="${UNZIPLORA_CONTENT_WEIGHTS}" \
  --unziplora_style_weight_path="${UNZIPLORA_STYLE_WEIGHTS}" \
  --motion_adapter_path=$MOTION_ADAPTER \
  --resolution=$RESOLUTION \
  --num_frames=$NUM_FRAMES \
  --train_batch_size=1 \
  --gradient_accumulation_steps=$GRAD_ACC_STEPS \
  --enable-gradient_checkpointing \
  --learning_rate="${LEARNING_RATE}" \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=$WARMUP_STEPS \
  --max_train_steps=$STEPS \
  --checkpointing_steps=500 \
  --mixed_precision=$MIXED_PRECISION \
  --seed="0"
