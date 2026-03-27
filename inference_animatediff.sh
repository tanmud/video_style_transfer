#!bin/bash

# Paths
export STAGE2_DIR="models/male_biker_video"
export SAVE_DIR="output/male_biker"
export CHECKPOINT="checkpoint-final"

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export MOTION_ADAPTER_PATH="$STAGE2_DIR/$CHECKPOINT"

# Stage-1 UnZipLoRA outputs
export UNZIPLORA_CONTENT="models/male_biker_image/male_biker_image_content"
export UNZIPLORA_STYLE="models/male_biker_image/male_biker_image_style"
export UNZIPLORA_CONTENT_WEIGHTS="models/male_biker_image/male_biker_image_merger_content.pth"
export UNZIPLORA_STYLE_WEIGHTS="models/male_biker_image/male_biker_image_merger_style.pth"

# Generation settings
export NUM_FRAMES=16
export NUM_INFERENCE_STEPS=75
export GUIDANCE_SCALE=7.5
export HEIGHT=1024
export WIDTH=1024
export FPS=8
export MIXED_PRECISION="bf16"

# Three generation modes:
# type=both  
export INSTANCE_PROMPT="A male biker in cartoon style biking on the street"
# type=content 
export CONTENT_PROMPT="A male biker biking in a snowy landscape"
# type=style  
export STYLE_PROMPT="A dog running in cartoon style"

python inference_animatediff.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --unziplora_content_path="${UNZIPLORA_CONTENT}" \
  --unziplora_style_path="${UNZIPLORA_STYLE}" \
  --unziplora_content_weight_path="${UNZIPLORA_CONTENT_WEIGHTS}" \
  --unziplora_style_weight_path="${UNZIPLORA_STYLE_WEIGHTS}" \
  --motion_adapter_path="$MOTION_ADAPTER_PATH" \
  --instance_prompt="${INSTANCE_PROMPT}" \
  --content_prompt="${CONTENT_PROMPT}" \
  --style_prompt="${STYLE_PROMPT}" \
  --save_dir="$SAVE_DIR" \
  --num_frames=$NUM_FRAMES \
  --num_inference_steps=$NUM_INFERENCE_STEPS \
  --guidance_scale=$GUIDANCE_SCALE \
  --height=$HEIGHT \
  --width=$WIDTH \
  --fps=$FPS \
  --mixed_precision=$MIXED_PRECISION