import argparse
import torch 
import os

from diffusers import (
    AutoencoderKL,
    AnimateDiffPipeline,
    MotionAdapter,
    DDIMScheduler,
    StableDiffusionXLPipeline
)

from diffusers.utils import export_to_video
from unziplora_unet.utils import *

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-sdxl-beta"
# MODEL_ID="etri-vilab/koala-lightning-1b"
seeds = [0, 1000, 111, 1234]
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.float16

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--with_unziplora",
        action="store_true",
        help="Whether use different prompts to generate",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=4,
        help=("The number of generated figures of each seed."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help=("The directory for saved model"),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="example_output",
        help=("The directory for saved generated figures"),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=("The prompt for validation"),
    )

    parser.add_argument(
        "--validation_prompt_content_forward",
        type=str,
        default=None,
        help=("The prompt for validation"),
    )
    parser.add_argument(
        "--validation_prompt_style_forward",
        type=str,
        default=None,
        help=("The prompt for validation"),
    )

    parser.add_argument(
        "--validation_prompt_content_recontext",
        type=str,
        default=None,
        help=("The content recontext prompt for validation"),
    )

    parser.add_argument(
        "--validation_prompt_style",
        type=str,
        default=None,
        help=("The style prompt for validation"),
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Number of frames per video (must be divisible by 8)")
    parser.add_argument("--fps", type=int, default=8,
                       help="Frames per second")
    
    
    
    args = parser.parse_args()
    
    return args 

def log_validation(pipeline, prompt, prompt_content="", prompt_style="", seed=0, num=4, num_frames=16):
    generator = torch.Generator(device=device).manual_seed(seed)
    # Currently the context determination is a bit hand-wavy. We can improve it in the future if there's a better
    # way to condition it. Reference: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
    if pipeline.__class__.__name__ == 'AnimateDiffPipeline':
        pipeline_args = {"prompt": prompt, 
                        "prompt_content": prompt_content, 
                        "prompt_style": prompt_style,
                        "num_frames": num_frames,}
    else: 
        pipeline_args = {"prompt": prompt}
        
    videos = []
    for _ in range(num):
        output = pipeline(**pipeline_args, generator=generator, num_inference_steps=25)
        videos.append(output.frames[0])
    return videos

def save_img(img_dir, images, img_num):
    for _, img in enumerate(images):
        image_path = os.path.join(img_dir, f"image_{img_num}.png")
        img_num += 1
        img.save(image_path, "PNG")
    return img_num

def save_videos(video_dir, videos, video_num, fps=8):
    """
    NEW FUNCTION: Save videos instead of images
    """
    for _, video_frames in enumerate(videos):
        video_path = os.path.join(video_dir, f"video_{video_num}.mp4")
        video_num += 1
        export_to_video(video_frames, video_path, fps=fps)
        print(f"   ✅ Saved: video_{video_num-1}.mp4")
    return video_num

def generate_save_img(args, pipeline, prompt, prompt_category):
    """Single image generation (for content/style-only tests)"""
    for i in range(len(prompt)):
        img_num = 1
        prompt_dir = os.path.join(prompt_category, "_".join(prompt[i].split(" ")))
        
        if os.path.isdir(prompt_dir):
            continue
        
        os.makedirs(prompt_dir, exist_ok=True)
        
        for seed in seeds:
            print(f"      {prompt[i]} (seed={seed})")
            
            generator = torch.Generator(device=device).manual_seed(seed)
            images = [
                pipeline(prompt[i], generator=generator, num_inference_steps=50).images[0] 
                for _ in range(args.num)
            ]
            
            # Save images
            for img in images:
                image_path = os.path.join(prompt_dir, f"image_{img_num}.png")
                img.save(image_path, "PNG")
                img_num += 1

def generate_save_videos(args, pipeline, prompt, prompt_catogory, prompt_content_forward=None, prompt_style_forward=None):
    for i in range(len(prompt)):
        vid_num = 1
        prompt_dir = os.path.join(prompt_catogory, "_".join(prompt[i].split(" ")))
        if os.path.isdir(prompt_dir):
            continue
        os.makedirs(prompt_dir, exist_ok=True)
        for seed in seeds:
            print(f"    Prompt {i}: {prompt[i]}")
            if pipeline.__class__.__name__ == 'StableDiffusionXLUnZipLoRAPipeline':
                images = log_validation(
                    pipeline,
                    prompt[i],
                    prompt_content_forward[i],
                    prompt_style_forward[i],
                    seed = seed, 
                    num = args.num,
                    num_frames = args.num_frames
                )
            else:
                images = log_validation(
                    pipeline,
                    prompt[i],
                    seed = seed, 
                    num = args.num,
                    num_frames = args.num_frames
                )
            vid_num = save_videos(prompt_dir, images, vid_num, fps=args.fps)
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    with torch.no_grad():
        print("=" * 70)
        print("AnimateDiff Inference + UnZipLoRA")
        print("=" * 70)

        motion_adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_ID, torch_dtype=weight_dtype)
        
        # * Generate combined images
        vae = AutoencoderKL.from_pretrained(
            MODEL_ID,
            subfolder="vae",
            revision=None
        )
        print("model dir: ", args.output_dir)
        
        if len(args.validation_prompt) != 0 and args.validation_prompt != ['']: 
            pipeline = AnimateDiffPipeline.from_pretrained(
                MODEL_ID,
                motion_adapter=motion_adapter,
                vae=vae,
                torch_dtype=weight_dtype
            )

            # Use DDIM scheduler (better for video)
            pipeline.scheduler = DDIMScheduler.from_config(
                pipeline.scheduler.config,
                beta_schedule="linear",
                clip_sample=False
            )
            
            if args.with_unziplora:
                pipeline.unet = insert_unziplora_to_unet(pipeline.unet, 
                    f"{args.output_dir}_content", 
                    f"{args.output_dir}_style",
                    weight_content_path=f"{args.output_dir}_merger_content.pth",
                    weight_style_path=f"{args.output_dir}_merger_style.pth",
                    rank=args.rank)
            else:
                pipeline.unet = insert_unziplora_to_unet(pipeline.unet, 
                    f"{args.output_dir}_content", 
                    f"{args.output_dir}_style",
                    rank=args.rank)
            pipeline = pipeline.to(device, dtype=weight_dtype)
            pipeline.enable_vae_slicing()

            print("=" * 70)
            print("Generating videos...")
            print("=" * 70)

            prompt_category = os.path.join(args.save_dir, "combine_recontextual_outputs")
            os.makedirs(prompt_catogory, exist_ok=True)
            if args.with_unziplora:
                generate_save_videos(
                    args, pipeline, args.validation_prompt, prompt_category,
                    args.validation_prompt_content_forward, 
                    args.validation_prompt_style_forward
                )
            else:
                generate_save_videos(args, pipeline, args.validation_prompt, prompt_category)
            
            del pipeline

        print("\n" + "=" * 70)
        print("All videos generated")
        print("=" * 70)
        
        if len(args.validation_prompt_content_recontext) != 0 and args.validation_prompt_content_recontext != ['']: 
            pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            )
            prompt_catogory = os.path.join(args.save_dir, "content_recontextual_outputs")
            os.makedirs(prompt_catogory, exist_ok=True)
            
            pipeline = pipeline.to(device, dtype=weight_dtype)
            pipeline.load_lora_weights(f"{args.output_dir}_content")
            # pipeline.load_lora_weights(f"{args.output_dir}")
            print(f"generate recontext prompt {args.validation_prompt_content_recontext}")
            generate_save_img(args, pipeline, args.validation_prompt_content_recontext, prompt_catogory)
            
            del pipeline        

        if len(args.validation_prompt_style) != 0 and args.validation_prompt_style != ['']: 
            prompt_catogory = os.path.join(args.save_dir, "style_recontextual_outputs")
            os.makedirs(prompt_catogory, exist_ok=True)
            
            pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            )
            pipeline = pipeline.to(device, dtype=weight_dtype)
            pipeline.load_lora_weights(f"{args.output_dir}_style")
            # pipeline.load_lora_weights(f"{args.output_dir}")
            print(f"generate recontext prompt {args.validation_prompt_style}")
            generate_save_img(args, pipeline, args.validation_prompt_style, prompt_catogory)
        
if __name__ == "__main__":
    args = parse_args()
    args.validation_prompt=args.validation_prompt.split(",")
    args.validation_prompt_style_forward=args.validation_prompt_style_forward.split(",")
    args.validation_prompt_content_forward=args.validation_prompt_content_forward.split(",")
    args.validation_prompt_content_recontext=args.validation_prompt_content_recontext.split(",")
    args.validation_prompt_style=args.validation_prompt_style.split(",")
    
    main(args)
