#!/bin/bash

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export RANK=64

export OUTPUT_DIR="models/male_biker/male_biker"
export SAVE_DIR="output/male_biker_videos/"

export VALIDATION_PROMPT="a male biker riding through a forest in cartoon style"
export VALIDATION_CONTENT="a male biker riding through a forest"
export VALIDATION_STYLE="a person riding through a forest in cartoon style"

python infer_video.py \
  --output_dir="$OUTPUT_DIR" \
  --rank="${RANK}" \
  --with_unziplora \
  --save_dir="$SAVE_DIR" \
  --validation_prompt="${VALIDATION_PROMPT}" \
  --validation_prompt_content_forward="${VALIDATION_CONTENT}" \
  --validation_prompt_style_forward="${VALIDATION_STYLE}" \
  --num_frames=16 \
  --fps=8 \
  --seed=0
