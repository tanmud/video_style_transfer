export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

# AnimateDiff Motion Module Settings
export MOTION_LAYERS=2

# Video Data Settings
export VIDEO_DATA_DIR="./instance_videos/male_biker"
export NUM_FRAMES=16  # Number of frames per video
export RESOLUTION=512  # Video resolution (512 or 1024)

# Training Settings
export OUTPUT_DIR="./outputs/animatediff_training"
export TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION=4
export MAX_STEPS=10000
export LEARNING_RATE=1e-4
export MIXED_PRECISION="fp16"  # "no", "fp16", or "bf16"

# Training Mode
export TRAIN_MOTION_ONLY=true  # Set to false for full fine-tuning

# Scheduler Settings
export LR_SCHEDULER="constant"
export LR_WARMUP_STEPS=0

# Optimizer Settings
export ADAM_BETA1=0.9
export ADAM_BETA2=0.999
export ADAM_WEIGHT_DECAY=1e-2
export ADAM_EPSILON=1e-8
export MAX_GRAD_NORM=1.0

# Logging & Checkpointing
export LOG_EVERY=10
export SAVE_EVERY=1000
export DATALOADER_NUM_WORKERS=0

# WandB Settings
export WANDB_MODE="offline"  # "online" or "offline"

# Build accelerate command
ACCELERATE_CMD="accelerate launch train_animatediff.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --motion_module_layers=$MOTION_LAYERS \
    --video_data_dir=$VIDEO_DATA_DIR \
    --num_frames=$NUM_FRAMES \
    --resolution=$RESOLUTION \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    --max_train_steps=$MAX_STEPS \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler=$LR_SCHEDULER \
    --lr_warmup_steps=$LR_WARMUP_STEPS \
    --adam_beta1=$ADAM_BETA1 \
    --adam_beta2=$ADAM_BETA2 \
    --adam_weight_decay=$ADAM_WEIGHT_DECAY \
    --adam_epsilon=$ADAM_EPSILON \
    --max_grad_norm=$MAX_GRAD_NORM \
    --mixed_precision=$MIXED_PRECISION \
    --log_every=$LOG_EVERY \
    --save_every=$SAVE_EVERY \
    --dataloader_num_workers=$DATALOADER_NUM_WORKERS"

# Add train_motion_only flag if enabled
if [ "$TRAIN_MOTION_ONLY" = "true" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD \
    --train_motion_only"
fi

# Execute training
eval $ACCELERATE_CMD
