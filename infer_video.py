import argparse
import torch
import os
import cv2
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from unziplora_unet.utils import *

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
seeds = [0, 1000, 111, 1234]
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.float16


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Video inference script for UnZipLoRA.")
    
    parser.add_argument(
        "--with_unziplora",
        action="store_true",
        help="Whether to use UnZipLoRA (content/style separation)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=4,
        help="The number of videos to generate per seed.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to generate per video",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second for output video",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory containing the trained model",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="example_output",
        help="The directory for saved generated videos/frames",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="The prompt for validation (can use comma-separated for multiple)",
    )
    parser.add_argument(
        "--validation_prompt_content_forward",
        type=str,
        default=None,
        help="The content forward prompt for UnZipLoRA",
    )
    parser.add_argument(
        "--validation_prompt_style_forward",
        type=str,
        default=None,
        help="The style forward prompt for UnZipLoRA",
    )
    parser.add_argument(
        "--validation_prompt_content_recontext",
        type=str,
        default=None,
        help="The content recontext prompt for validation",
    )
    parser.add_argument(
        "--validation_prompt_style",
        type=str,
        default=None,
        help="The style prompt for validation",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Whether to save individual frames",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    
    args = parser.parse_args()
    return args


def generate_video_frames(
    pipeline,
    prompt,
    prompt_content=None,
    prompt_style=None,
    seed=0,
    num_frames=16,
    inference_steps=50,
):
    """
    Generate a sequence of frames for a video.
    Each frame is generated independently with different noise but same prompt.
    """
    frames = []
    
    for frame_idx in range(num_frames):
        # Use different seed for each frame to get variation
        frame_seed = seed + frame_idx
        generator = torch.Generator(device=device).manual_seed(frame_seed)
        
        # Generate single frame
        if pipeline.__class__.__name__ == "StableDiffusionXLUnZipLoRAPipeline":
            pipeline_args = {
                "prompt": prompt,
                "prompt_content": prompt_content,
                "prompt_style": prompt_style,
            }
        else:
            pipeline_args = {"prompt": prompt}
        
        image = pipeline(
            **pipeline_args,
            generator=generator,
            num_inference_steps=inference_steps,
        ).images[0]
        
        frames.append(image)
        
        print(f"  Generated frame {frame_idx + 1}/{num_frames}", end='\r')
    
    print()  # New line after progress
    return frames


def frames_to_video(frames, output_path, fps=8):
    """
    Convert list of PIL Images to video file.
    """
    # Convert PIL images to numpy arrays
    frame_arrays = [np.array(frame.convert('RGB')) for frame in frames]
    
    # Get dimensions from first frame
    height, width, _ = frame_arrays[0].shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame_array in frame_arrays:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to: {output_path}")


def save_frames(frame_dir, frames):
    """
    Save individual frames as images.
    """
    os.makedirs(frame_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame_path = os.path.join(frame_dir, f"frame_{idx:04d}.png")
        frame.save(frame_path, "PNG")
    print(f"Frames saved to: {frame_dir}")


def generate_and_save_videos(
    args,
    pipeline,
    prompts,
    prompt_category,
    prompts_content_forward=None,
    prompts_style_forward=None,
):
    """
    Generate videos for all prompts and seeds.
    """
    for i in range(len(prompts)):
        # Create directory for this prompt
        prompt_name = "_".join(prompts[i].split()[:5])  # First 5 words
        prompt_dir = os.path.join(prompt_category, prompt_name)
        os.makedirs(prompt_dir, exist_ok=True)
        
        video_num = 1
        
        for seed in seeds:
            print(f"Generating video for prompt: {prompts[i][:50]}... (seed={seed})")
            
            # Get content and style prompts if using UnZipLoRA
            if pipeline.__class__.__name__ == "StableDiffusionXLUnZipLoRAPipeline":
                prompt_content = prompts_content_forward[i] if prompts_content_forward else None
                prompt_style = prompts_style_forward[i] if prompts_style_forward else None
            else:
                prompt_content = None
                prompt_style = None
            
            for _ in range(args.num):
                # Generate frames
                frames = generate_video_frames(
                    pipeline,
                    prompts[i],
                    prompt_content,
                    prompt_style,
                    seed=seed,
                    num_frames=args.num_frames,
                    inference_steps=args.inference_steps,
                )
                
                # Save as video
                video_path = os.path.join(prompt_dir, f"video_{video_num:03d}.mp4")
                frames_to_video(frames, video_path, fps=args.fps)
                
                # Optionally save individual frames
                if args.save_frames:
                    frame_dir = os.path.join(prompt_dir, f"video_{video_num:03d}_frames")
                    save_frames(frame_dir, frames)
                
                video_num += 1


def load_pipeline_from_sdxl(model_id, vae):
    """Load the appropriate pipeline based on whether UnZipLoRA is used."""
    from unziplora_unet.pipeline_stable_diffusion_xl import StableDiffusionXLUnZipLoRAPipeline
    
    return StableDiffusionXLUnZipLoRAPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=weight_dtype,
    )


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    with torch.no_grad():
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            MODEL_ID,
            subfolder="vae",
            revision=None,
        )
        
        print("Model dir:", args.output_dir)
        
        # Generate combined (content+style) videos
        if len(args.validation_prompt) != 0 and args.validation_prompt != "":
            pipeline = load_pipeline_from_sdxl(MODEL_ID, vae=vae)
            
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
            
            prompt_category = os.path.join(args.save_dir, "combine_recontextual_outputs")
            os.makedirs(prompt_category, exist_ok=True)
            
            if args.with_unziplora:
                generate_and_save_videos(
                    args,
                    pipeline,
                    args.validation_prompt,
                    prompt_category,
                    args.validation_prompt_content_forward,
                    args.validation_prompt_style_forward,
                )
            else:
                generate_and_save_videos(
                    args,
                    pipeline,
                    args.validation_prompt,
                    prompt_category,
                )
            
            del pipeline
        
        # Generate content-only recontextualization videos
        if len(args.validation_prompt_content_recontext) != 0 and args.validation_prompt_content_recontext != "":
            pipeline = StableDiffusionXLPipeline.from_pretrained(MODEL_ID)
            
            prompt_category = os.path.join(args.save_dir, "content_recontextual_outputs")
            os.makedirs(prompt_category, exist_ok=True)
            
            pipeline = pipeline.to(device, dtype=weight_dtype)
            pipeline.load_lora_weights(f"{args.output_dir}/content")
            
            print(f"Generate recontext prompt: {args.validation_prompt_content_recontext}")
            generate_and_save_videos(
                args,
                pipeline,
                args.validation_prompt_content_recontext,
                prompt_category,
            )
            
            del pipeline
        
        # Generate style-only recontextualization videos
        if len(args.validation_prompt_style) != 0 and args.validation_prompt_style != "":
            prompt_category = os.path.join(args.save_dir, "style_recontextual_outputs")
            os.makedirs(prompt_category, exist_ok=True)
            
            pipeline = StableDiffusionXLPipeline.from_pretrained(MODEL_ID)
            pipeline = pipeline.to(device, dtype=weight_dtype)
            pipeline.load_lora_weights(f"{args.output_dir}/style")
            
            print(f"Generate recontext prompt: {args.validation_prompt_style}")
            generate_and_save_videos(
                args,
                pipeline,
                args.validation_prompt_style,
                prompt_category,
            )


if __name__ == "__main__":
    args = parse_args()
    
    # Parse comma-separated prompts
    args.validation_prompt = args.validation_prompt.split(",") if args.validation_prompt else []
    args.validation_prompt_style_forward = args.validation_prompt_style_forward.split(",") if args.validation_prompt_style_forward else []
    args.validation_prompt_content_forward = args.validation_prompt_content_forward.split(",") if args.validation_prompt_content_forward else []
    args.validation_prompt_content_recontext = args.validation_prompt_content_recontext.split(",") if args.validation_prompt_content_recontext else []
    args.validation_prompt_style = args.validation_prompt_style.split(",") if args.validation_prompt_style else []
    
    main(args)
