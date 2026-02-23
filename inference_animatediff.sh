export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export MOTION_LAYERS=2

# Paths
export STAGE2_DIR="models/male_biker_video"
export SAVE_DIR="output/male_biker"
export MOTION_MODULE_PATH="${STAGE2_DIR}/checkpoint-final"

# Stage-1 UnZipLoRA outputs
export UNZIPLORA_CONTENT="models/male_biker_image/content"
export UNZIPLORA_STYLE="models/male_biker_image/style"
export UNZIPLORA_CONTENT_WEIGHTS="models/male_biker_image/mergercontent.pth"
export UNZIPLORA_STYLE_WEIGHTS="models/male_biker_image/mergerstyle.pth"

# Generation settings
export NUM_FRAMES=16
export NUM_INFERENCE_STEPS=50
export GUIDANCE_SCALE=7.5
export HEIGHT=512
export WIDTH=512
export FPS=8

# Three generation modes:
# type=both  -> full subject (instance prompt)
export INSTANCE_PROMPT="A male biker in cartoon style biking on the street"
# type=content -> swap content, keep learned cartoon style
export CONTENT_PROMPT="A male biker biking in a snowy landscape"
# type=style  -> swap style onto new subject, keep learned motion
export STYLE_PROMPT="A dog running in cartoon style"

python inference_animatediff.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --motion_module_path="$MOTION_MODULE_PATH" \
  --motion_module_layers=$MOTION_LAYERS \
  --unziplora_content_path="${UNZIPLORA_CONTENT}" \
  --unziplora_style_path="${UNZIPLORA_STYLE}" \
  --unziplora_content_weight_path="${UNZIPLORA_CONTENT_WEIGHTS}" \
  --unziplora_style_weight_path="${UNZIPLORA_STYLE_WEIGHTS}" \
  --instance_prompt="${INSTANCE_PROMPT}" \
  --content_prompt="${CONTENT_PROMPT}" \
  --style_prompt="${STYLE_PROMPT}" \
  --save_dir="$SAVE_DIR" \
  --num_frames=$NUM_FRAMES \
  --num_inference_steps=$NUM_INFERENCE_STEPS \
  --guidance_scale=$GUIDANCE_SCALE \
  --height=$HEIGHT \
  --width=$WIDTH \
  --fps=$FPS
