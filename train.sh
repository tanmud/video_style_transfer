export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

# Hyper parameters
export period_sample_epoch=3
export sampled_column_ratio=0.1

# For weight similarity
export CONTENT_LR=0.00005
export STYLE_LR=0.00005
export weight_lr=0.005
export similarity_lambda=0.5
export RANK=64

export WANDB_NAME="unziplora"
export WANDB_MODE="offline"

export NUM_FRAMES=16  # Number of frames to sample per video
export RESOLUTION=512  # Lower resolution for video (memory)

export INSTANCE_DIR="male_biker/male_biker"
export OUTPUT_DIR="models/male_biker/male_biker"
export STEPS=600

# Training prompt
export PROMPT="A male biker in cartoon style biking on the street"
export CONTENT_FORWARD_PROMPT="A male biker"
export STYLE_FORWARD_PROMPT="A biker in cartoon style"
# For validation
export VALID_CONTENT="A male biker biking on a park"
export VALID_PROMPT="A male biker in cartoon style biking on a park"
export VALID_STYLE="A biker in cartoon style biking on a park"

# for content validation
export VALID_CONTENT_PROMPT="a video of a biker in park"

# for style validation
export VALID_STYLE_PROMPT="A biker in cartoon style"



accelerate launch video_train_unziplora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --name=$WANDB_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --content_forward_prompt="${CONTENT_FORWARD_PROMPT}" \
  --style_forward_prompt="${STYLE_FORWARD_PROMPT}" \
  --rank="${RANK}" \
  --resolution=$RESOLUTION \
  --num_frames=$NUM_FRAMES \
  --train_batch_size=1 \
  --content_learning_rate="${CONTENT_LR}" \
  --style_learning_rate="${STYLE_LR}" \
  --weight_learning_rate="$weight_lr" \
  --similarity_lambda="$similarity_lambda" \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps="$STEPS" \
  --checkpointing_steps=500 \
  --mixed_precision="fp16" \
  --seed="0" \
  --validation_content="${VALID_CONTENT}" \
  --validation_style="${VALID_STYLE}" \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_prompt_style="${VALID_STYLE_PROMPT}" \
  --validation_prompt_content="${VALID_CONTENT_PROMPT}" \
  --sample_times=$period_sample_epoch \
  --column_ratio=$sampled_column_ratio \
  # --use_8bit_adam \