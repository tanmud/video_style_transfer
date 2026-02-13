export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

# AnimateDiff Motion Module Settings
export MOTION_LAYERS=2
export MOTION_MODULE_PATH="./outputs/animatediff_training/checkpoint-final/motion_modules.pth"

# Output Settings
export SAVE_DIR="./generated_videos"

# Generation Settings
export NUM_FRAMES=16
export NUM_INFERENCE_STEPS=50
export GUIDANCE_SCALE=7.5
export HEIGHT=512
export WIDTH=512
export FPS=8

# Validation Prompts (UnzipLoRA style - content + style separation)
export VALID_PROMPTS=(
    "A cat walking through a garden in cinematic style"
    "A dog running on the beach at sunset in anime style"
    "A bird flying through clouds in watercolor style"
)

VALID_PROMPT=$(IFS=,; echo "${VALID_PROMPTS[*]}")
export VALID_PROMPT

# Content prompts (for content-focused generation)
export VALID_CONTENTS=(
    "A cat walking through a garden"
    "A dog running on the beach"
    "A bird flying through clouds"
)

VALID_CONTENT=$(IFS=,; echo "${VALID_CONTENTS[*]}")
export VALID_CONTENT

# Style prompts (for style-focused generation)
export VALID_STYLES=(
    "A scene in cinematic style"
    "A scene at sunset in anime style"
    "A scene in watercolor style"
)

VALID_STYLE=$(IFS=,; echo "${VALID_STYLES[*]}")
export VALID_STYLE

# Style transfer prompts (different content with same style)
export VALID_STYLE_TRANSFER_PROMPTS=(
    "A car driving down a highway in cinematic style"
    "A person dancing on stage in anime style"
    "A flower blooming in watercolor style"
)

VALID_STYLE_TRANSFER=$(IFS=,; echo "${VALID_STYLE_TRANSFER_PROMPTS[*]}")
export VALID_STYLE_TRANSFER

# Content recontextualization prompts (same content, different context)
export VALID_CONTENT_RECON_PROMPTS=(
    "A cat sitting on a chair"
    "A dog playing with a ball"
    "A bird perched on a branch"
)

VALID_CONTENT_RECON=$(IFS=,; echo "${VALID_CONTENT_RECON_PROMPTS[*]}")
export VALID_CONTENT_RECON

accelerate launch inference_animatediff.py \
    --pretrained_model_name_or_path="$MODEL_NAME" \
    --motion_module_path="$MOTION_MODULE_PATH" \
    --motion_module_layers=$MOTION_LAYERS \
    --save_dir="$SAVE_DIR" \
    --num_frames=$NUM_FRAMES \
    --num_inference_steps=$NUM_INFERENCE_STEPS \
    --guidance_scale=$GUIDANCE_SCALE \
    --height=$HEIGHT \
    --width=$WIDTH \
    --fps=$FPS \
    --validation_prompts="$VALID_PROMPT" \
    --validation_content_prompts="$VALID_CONTENT" \
    --validation_style_prompts="$VALID_STYLE" \
    --validation_style_transfer_prompts="$VALID_STYLE_TRANSFER" \
    --validation_content_recon_prompts="$VALID_CONTENT_RECON"
