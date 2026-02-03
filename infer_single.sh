export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export RANK=64
export NUM=1

export OUTPUT_DIR="models/male_biker/male_biker"
export SAVE_DIR="output/male_biker_test/"

# Combined prompts (content + style)
export VALID_PROMPTS=(
  "a male biker on a skateboard in cartoon style"
  "a male biker in a snowy landscape in cartoon style"
)
VALID_PROMPT=$(IFS=,; echo "${VALID_PROMPTS[*]}")
export VALID_PROMPT

# Style component prompts
export VALID_STYLES=(
  "a person on a skateboard in cartoon style"
  "a person in a snowy landscape in cartoon style"
)
VALID_STYLE=$(IFS=,; echo "${VALID_STYLES[*]}")
export VALID_STYLE

# Content component prompts
export VALID_CONTENTS=(
  "a male biker on a skateboard"
  "a male biker in a snowy landscape"
)
VALID_CONTENT=$(IFS=,; echo "${VALID_CONTENTS[*]}")
export VALID_CONTENT

# Content reconstruction prompts (test content LoRA only)
export VALID_CONTENT_RECON_PROMPTS=(
  "A photo of a male biker on a table"
  "A photo of a male biker at the beach"
)
VALID_CONTENT_RECON_PROMPT=$(IFS=,; echo "${VALID_CONTENT_RECON_PROMPTS[*]}")
export VALID_CONTENT_RECON_PROMPT

# Style reconstruction prompts (test style LoRA only)
export VALID_STYLE_PROMPTS=(
  "A dog in cartoon style"
  "A chair in cartoon style"
)
VALID_STYLE_PROMPT=$(IFS=,; echo "${VALID_STYLE_PROMPTS[*]}")
export VALID_STYLE_PROMPT

accelerate launch infer.py \
  --output_dir="$OUTPUT_DIR" \
  --rank="${RANK}" \
  --num="${NUM}" \
  --with_unziplora \
  --save_dir="$SAVE_DIR" \
  --validation_prompt_content_recontext="${VALID_CONTENT_RECON_PROMPT}" \
  --validation_prompt_style="${VALID_STYLE_PROMPT}" \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_prompt_style_forward="${VALID_STYLE}" \
  --validation_prompt_content_forward="${VALID_CONTENT}"
