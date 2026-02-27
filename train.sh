export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

export PYTHONUNBUFFERED=1 # flush output on log file

# Hyperparameters
export RANK=64
export CONTENT_LR=0.00005
export STYLE_LR=0.00005
export weight_lr=0.005
export similarity_lambda=0.5
export period_sample_epoch=3
export sampled_column_ratio=0.1

# WandB
export WANDB_NAME="unziplora_male_biker"
export WANDB_MODE="offline"

# Input: point at a directory. If it contains an .mp4, frames are extracted
# automatically (requires the DreamBoothDataset patch above).
# Or pre-extract manually:
#   ffmpeg -i male_biker.mp4 -vf fps=2 instance_data/male_biker/frame_%04d.jpg
export INSTANCE_VIDEO="instance_videos/male_biker/male_biker.mp4"
export NUM_INSTANCE_FRAMES=1   # 1 = middle frame; use 3-5 for style variety

# Output — Stage 2 will look for:
#   models/male_biker_image/male_biker_image_content/
#   models/male_biker_image/male_biker_image_style/
#   models/male_biker_image/male_biker_image_mergercontent.pth
#   models/male_biker_image/male_biker_image_mergerstyle.pth
export OUTPUT_DIR="models/male_biker_image/male_biker_image"

export STEPS=600

# Prompts
export PROMPT="A male biker in cartoon style biking on the street"
export CONTENT_FORWARD_PROMPT="A male biker"
export STYLE_FORWARD_PROMPT="A biker in cartoon style"
export VALID_CONTENT="A male biker biking on a park"
export VALID_PROMPT="A male biker in cartoon style biking on a park"
export VALID_STYLE="A biker in cartoon style biking on a park"
export VALID_CONTENT_PROMPT="a photo of a male biker in a park"
export VALID_STYLE_PROMPT="A dog running in cartoon style"

accelerate launch train_unziplora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --name=$WANDB_NAME \
  --instance_video="${INSTANCE_VIDEO}" \
  --num_instance_frames=$NUM_INSTANCE_FRAMES \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --content_forward_prompt="${CONTENT_FORWARD_PROMPT}" \
  --style_forward_prompt="${STYLE_FORWARD_PROMPT}" \
  --rank="${RANK}" \
  --resolution=1024 \
  --train_batch_size=1 \
  --content_learning_rate="${CONTENT_LR}" \
  --style_learning_rate="${STYLE_LR}" \
  --weight_learning_rate="$weight_lr" \
  --similarity_lambda="$similarity_lambda" \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps="${STEPS}" \
  --checkpointing_steps=500 \
  --mixed_precision="no" \
  --seed="0" \
  --validation_content="${VALID_CONTENT}" \
  --validation_style="${VALID_STYLE}" \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_prompt_style="${VALID_STYLE_PROMPT}" \
  --validation_prompt_content="${VALID_CONTENT_PROMPT}" \
  --sample_times=$period_sample_epoch \
  --column_ratio=$sampled_column_ratio  \
  2>&1 | tee -a train_log.txt