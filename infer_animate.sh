export RANK=64
export NUM=2

export OUTPUT_DIR="models/male_biker/male_biker"
export SAVE_DIR="output/animatediff_full_test/"

# Combined prompts
export VALID_PROMPTS=(
  "a male biker in cartoon style on Mars"
)
VALID_PROMPT=$(IFS=,; echo "${VALID_PROMPTS[*]}")

export VALID_STYLES=(
  "a person in cartoon style on Mars"
)
VALID_STYLE=$(IFS=,; echo "${VALID_STYLES[*]}")

export VALID_CONTENTS=(
  "a male biker on Mars"
)
VALID_CONTENT=$(IFS=,; echo "${VALID_CONTENTS[*]}")

# Content-only test prompts
export CONTENT_RECONTEXT_PROMPTS=(
  "A male biker at the beach"
  "A male biker in a forest"
)
CONTENT_RECONTEXT=$(IFS=,; echo "${CONTENT_RECONTEXT_PROMPTS[*]}")

# Style-only test prompts
export STYLE_PROMPTS=(
  "A dog in cartoon style"
  "A sports car in cartoon style"
)
STYLE_PROMPT=$(IFS=,; echo "${STYLE_PROMPTS[*]}")

accelerate launch infer_animate.py \
  --output_dir="$OUTPUT_DIR" \
  --rank="${RANK}" \
  --num="${NUM}" \
  --with_unziplora \
  --save_dir="$SAVE_DIR" \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_prompt_style_forward="${VALID_STYLE}" \
  --validation_prompt_content_forward="${VALID_CONTENT}" \
  --validation_prompt_content_recontext="${CONTENT_RECONTEXT}" \
  --validation_prompt_style="${STYLE_PROMPT}" \
  --num_frames=16 \
  --fps=8
