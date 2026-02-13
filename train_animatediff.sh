export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

# Motion Module Settings
export MOTION_LAYERS=2

# WandB Settings
export WANDB_NAME="animatediff_male_biker"
export WANDB_MODE="offline"

# Video Data Settings
export NUM_FRAMES=16  # Number of frames to sample per video
export RESOLUTION=512  # Lower resolution for video (memory)
export INSTANCE_DIR="./instance_videos/male_biker"
export OUTPUT_DIR="models/male_biker/male_biker"
export STEPS=10000

# Training Settings
export LEARNING_RATE=1e-4
export TRAIN_MOTION_ONLY=true

# Training prompts (male_biker specific)
export PROMPT="A male biker in cartoon style biking on the street"
export CONTENT_FORWARD_PROMPT="A male biker"
export STYLE_FORWARD_PROMPT="A biker in cartoon style"

# For validation (male_biker specific)
export VALID_CONTENT="A male biker biking on a park"
export VALID_PROMPT="A male biker in cartoon style biking on a park"
export VALID_STYLE="A biker in cartoon style biking on a park"

# for content validation (male_biker specific)
export VALID_CONTENT_PROMPT="a video of a biker in park"

# for style validation (male_biker specific)
export VALID_STYLE_PROMPT="A biker in cartoon style"

# Validation settings
export VALIDATION_STEPS=500  # How often to run validation

accelerate launch train_animatediff.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --name=$WANDB_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="${PROMPT}" \
    --content_forward_prompt="${CONTENT_FORWARD_PROMPT}" \
    --style_forward_prompt="${STYLE_FORWARD_PROMPT}" \
    --motion_module_layers=$MOTION_LAYERS \
    --resolution=$RESOLUTION \
    --num_frames=$NUM_FRAMES \
    --train_batch_size=1 \
    --learning_rate="${LEARNING_RATE}" \
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
    --validation_steps=$VALIDATION_STEPS \
    $([ "$TRAIN_MOTION_ONLY" = "true" ] && echo "--train_motion_only")
