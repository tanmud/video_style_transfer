export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

# Motion Module Settings
export MOTION_LAYERS=2

# Paths
export OUTPUT_DIR="models/male_biker/male_biker"
export SAVE_DIR="output/male_biker/"

# Motion module path (trained weights)
export MOTION_MODULE_PATH="${OUTPUT_DIR}/checkpoint-final/motion_modules.pth"

# Generation Settings
export NUM_FRAMES=16
export NUM_INFERENCE_STEPS=50
export GUIDANCE_SCALE=7.5
export HEIGHT=512
export WIDTH=512
export FPS=20

# Validation Prompts (combined content + style)
export VALID_PROMPTS=(
    "a male biker in cartoon style biking on a skateboard"
    "a male biker in cartoon style biking in a snowy landscape"
)

VALID_PROMPT=$(IFS=,; echo "${VALID_PROMPTS[*]}")
export VALID_PROMPT

# Style Forward Prompts (style-focused)
export VALID_STYLES=(
    "a biker in cartoon style on a skateboard"
    "a biker in cartoon style in a snowy landscape"
)

VALID_STYLE=$(IFS=,; echo "${VALID_STYLES[*]}")
export VALID_STYLE

# Content Forward Prompts (content-focused)
export VALID_CONTENTS=(
    "a male biker on a skateboard"
    "a male biker in a snowy landscape"
)

VALID_CONTENT=$(IFS=,; echo "${VALID_CONTENTS[*]}")
export VALID_CONTENT

# Content Recontextualization Prompts
export VALID_CONTENT_RECON_PROMPTS=(
    "A photo of a male biker on a table"
    "A photo of a male biker at a beach"
)

VALID_CONTENT_RECON_PROMPT=$(IFS=,; echo "${VALID_CONTENT_RECON_PROMPTS[*]}")
export VALID_CONTENT_RECON_PROMPT

# Style Transfer Prompts (different content, same style)
export VALID_STYLE_PROMPTS=(
    "A dog running in cartoon style"
    "A car driving in cartoon style"
)

VALID_STYLE_PROMPT=$(IFS=,; echo "${VALID_STYLE_PROMPTS[*]}")
export VALID_STYLE_PROMPT

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
    --validation_prompt="${VALID_PROMPT}" \
    --validation_prompt_style_forward="${VALID_STYLE}" \
    --validation_prompt_content_forward="${VALID_CONTENT}"
