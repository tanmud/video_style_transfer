#!/usr/bin/env python
import argparse
import torch
import os
import cv2
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from unziplora_unet.utils import *

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.float16

def parse_args():
    parser = argparse.ArgumentParser(description="Generate video from UnZipLoRA")
    parser.add_argument("--with_unziplora", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="output/videos")
    parser.add_argument("--validation_prompt", type=str, required=True)
    parser.add_argument("--validation_prompt_content_forward", type=str, default="")
    parser.add_argument("--validation_prompt_style_forward", type=str, default="")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    return parser.parse_args()

def generate_frame(pipeline, prompt, prompt_content, prompt_style, seed):
    """Generate a single frame"""
    generator = torch.Generator(device=device).manual_seed(seed)
    
    if pipeline.__class__.__name__ == 'StableDiffusionXLUnZipLoRAPipeline':
        pipeline_args = {
            "prompt": prompt,
            "prompt_content": prompt_content,
            "prompt_style": prompt_style
        }
    else:
        pipeline_args = {"prompt": prompt}
    
    image = pipeline(**pipeline_args, generator=generator, num_inference_steps=50).images[0]
    return image

def images_to_video(frames, output_path, fps=8):
    """Convert list of PIL images to MP4 video"""
    # Convert first image to get dimensions
    frame_array = np.array(frames[0])
    height, width = frame_array.shape[:2]
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert PIL to numpy array and BGR color space
        frame_array = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
    
    video.release()
    print(f"✅ Video saved to: {output_path}")

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    with torch.no_grad():
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            MODEL_ID,
            subfolder="vae",
            revision=None
        )
        
        print("Model dir:", args.output_dir)
        
        # Load pipeline
        pipeline = load_pipeline_from_sdxl(MODEL_ID, vae=vae)
        
        # Insert LoRAs
        if args.with_unziplora:
            pipeline.unet = insert_unziplora_to_unet(
                pipeline.unet,
                f"{args.output_dir}_content",
                f"{args.output_dir}_style",
                weight_content_path=f"{args.output_dir}_merger_content.pth",
                weight_style_path=f"{args.output_dir}_merger_style.pth",
                rank=args.rank
            )
        else:
            pipeline.unet = insert_unziplora_to_unet(
                pipeline.unet,
                f"{args.output_dir}_content",
                f"{args.output_dir}_style",
                rank=args.rank
            )
        
        pipeline = pipeline.to(device, dtype=weight_dtype)
        
        # Generate frames
        print(f"\n🎬 Generating {args.num_frames} frames...")
        frames = []
        
        for i in range(args.num_frames):
            seed = args.seed + i
            print(f"  Frame {i+1}/{args.num_frames} (seed={seed})", end='\r')
            
            frame = generate_frame(
                pipeline,
                args.validation_prompt,
                args.validation_prompt_content_forward,
                args.validation_prompt_style_forward,
                seed
            )
            frames.append(frame)
        
        print(f"\n✅ Generated {len(frames)} frames")
        
        # Save frames as images (optional)
        frames_dir = os.path.join(args.save_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
        print(f"📁 Frames saved to: {frames_dir}")
        
        # Create video
        video_name = "_".join(args.validation_prompt.split()[:5]) + ".mp4"
        video_path = os.path.join(args.save_dir, video_name)
        images_to_video(frames, video_path, fps=args.fps)

if __name__ == "__main__":
    args = parse_args()
    main(args)
