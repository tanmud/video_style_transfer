import argparse
import torch
import os
import imageio
from diffusers import DDPMScheduler
from animatediff.pipeline_animatediff_xl import AnimateDiffUnZipLoRAPipeline


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading pipeline from {args.pretrained_model_name_or_path}...")

    # Load pipeline (just like infer.py does!)
    pipeline = AnimateDiffUnZipLoRAPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
    ).to(device)

    # Load motion modules if provided
    if args.motion_module_path:
        print(f"Loading motion modules from {args.motion_module_path}...")
        motion_state = torch.load(args.motion_module_path, map_location=device)
        pipeline.unet.load_state_dict(motion_state, strict=False)
        print("Loaded trained motion modules")

    pipeline.unet.eval()

    # Parse prompts (comma-separated)
    validation_prompts = args.validation_prompt.split(",") if args.validation_prompt else []
    validation_content = args.validation_prompt_content_forward.split(",") if args.validation_prompt_content_forward else [None] * len(validation_prompts)
    validation_style = args.validation_prompt_style_forward.split(",") if args.validation_prompt_style_forward else [None] * len(validation_prompts)

    # Make sure lists are same length
    max_len = max(len(validation_prompts), len(validation_content), len(validation_style))
    validation_prompts += [validation_prompts[-1] if validation_prompts else ""] * (max_len - len(validation_prompts))
    validation_content += [None] * (max_len - len(validation_content))
    validation_style += [None] * (max_len - len(validation_style))

    # Make output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Generate videos
    for i, (prompt, content, style) in enumerate(zip(validation_prompts, validation_content, validation_style)):
        print(f"\n{'='*60}")
        print(f"Video {i+1}/{len(validation_prompts)}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt.strip()}")
        if content:
            print(f"Content: {content.strip()}")
        if style:
            print(f"Style: {style.strip()}")

        # Generate video using pipeline (CLEAN API!)
        output = pipeline(
            prompt=prompt.strip(),
            prompt_content=content.strip() if content else None,
            prompt_style=style.strip() if style else None,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            output_type="latent",  # Get latents first
        )

        # Decode latents to frames
        print("Decoding frames...")
        latents = output.images
        frames = []

        for frame_idx in range(latents.shape[1]):
            latent_frame = latents[0, frame_idx] / pipeline.vae.config.scaling_factor
            image = pipeline.vae.decode(latent_frame.unsqueeze(0), return_dict=False)[0]

            # Convert to [0, 1] range
            image = (image / 2 + 0.5).clamp(0, 1)

            # Convert to numpy uint8
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype("uint8")

            frames.append(image)

        # Save video
        safe_name = prompt.strip().replace(" ", "_")[:50]
        video_path = os.path.join(args.save_dir, f"video_{i:03d}_{safe_name}.mp4")

        print(f"Saving video to {video_path}...")
        imageio.mimsave(video_path, frames, fps=args.fps)
        print(f"Saved: {video_path}")

    print(f"\nGenerated {len(validation_prompts)} videos in {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnimateDiff inference with UnZipLoRA")

    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--motion_module_path", type=str, required=True)
    parser.add_argument("--motion_module_layers", type=int, default=2)

    # Prompts (match train.sh validation arguments)
    parser.add_argument("--validation_prompt", type=str, default="")
    parser.add_argument("--validation_prompt_content_forward", type=str, default="")
    parser.add_argument("--validation_prompt_style_forward", type=str, default="")
    parser.add_argument("--validation_prompt_content_recontext", type=str, default="")
    parser.add_argument("--validation_prompt_style", type=str, default="")

    # Generation settings
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--fps", type=int, default=8)

    # Output
    parser.add_argument("--save_dir", type=str, default="output/videos/")

    args = parser.parse_args()
    main(args)