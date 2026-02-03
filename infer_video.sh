export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

export RANK=64

export NUM=2  # Just 2 videos

export NUM_FRAMES=16

export FPS=8

export OUTPUT_DIR="models/instance_videos/male_biker"

export SAVE_DIR="output/male_biker_test/"

# ============================================
# Training was: "male biker in cartoon style on the street"
# Testing:      "male biker in cartoon style on Mars"
# ============================================

export VALID_PROMPT="A male biker in cartoon style biking on Mars"

export VALID_STYLE="A biker in cartoon style on Mars"

export VALID_CONTENT="A male biker biking on Mars"

# Run inference
python infer_video.py \
  --output_dir="$OUTPUT_DIR" \
  --rank="${RANK}" \
  --num="${NUM}" \
  --num_frames="${NUM_FRAMES}" \
  --fps="${FPS}" \
  --with_unziplora \
  --save_dir="$SAVE_DIR" \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_prompt_style_forward="${VALID_STYLE}" \
  --validation_prompt_content_forward="${VALID_CONTENT}" \
  --save_frames \
  --inference_steps=50
