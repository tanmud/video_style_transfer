import argparse
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from PIL import Image
import numpy as np
from pathlib import Path

# Import our utilities
from animatediff.utils import load_unet_with_motion, check_motion_module_compatibility


def encode_prompt(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt, device):
    """Encode text prompt for SDXL (dual text encoders)."""
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    prompt_embeds = text_encoder(
        text_inputs.input_ids.to(device),
        output_hidden_states=True,
    ).hidden_states[-2]

    pooled_prompt_embeds = text_encoder_2(
        text_inputs_2.input_ids.to(device),
        output_hidden_states=True,
    )[0]

    prompt_embeds_2 = text_encoder_2(
        text_inputs_2.input_ids.to(device),
        output_hidden_states=True,
    ).hidden_states[-2]

    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

    return prompt_embeds, pooled_prompt_embeds


@torch.no_grad()
def generate_video(
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    scheduler,
    prompt,
    num_frames=16,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    device="cuda",
):
    """Generate a video using the AnimateDiff UNet."""
    batch_size = 1

    # Encode prompt
    encoder_hidden_states, pooled_prompt_embeds = encode_prompt(
        text_encoder, text_encoder_2,
        tokenizer, tokenizer_2,
        [prompt], device
    )

    # Encode negative prompt
    negative_encoder_hidden_states, negative_pooled_prompt_embeds = encode_prompt(
        text_encoder, text_encoder_2,
        tokenizer, tokenizer_2,
        [""], device
    )

    # Prepare added conditions for SDXL
    add_time_ids = torch.cat([
        torch.tensor([height, width]),
        torch.tensor([0, 0]),
        torch.tensor([height, width]),
    ]).unsqueeze(0).to(device, dtype=encoder_hidden_states.dtype)

    negative_add_time_ids = add_time_ids.clone()

    # Prepare latents
    latent_channels = unet.config.in_channels
    latent_height = height // 8
    latent_width = width // 8

    latents = torch.randn(
        batch_size * num_frames,
        latent_channels,
        latent_height,
        latent_width,
        device=device,
        dtype=encoder_hidden_states.dtype,
    )

    latents = latents * scheduler.init_noise_sigma

    # Set timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Denoising loop
    print(f"Generating {num_frames} frames with prompt: '{prompt}'")
    for i, t in enumerate(timesteps):
        # Expand for CFG
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Concatenate embeddings
        encoder_hidden_states_input = torch.cat([
            negative_encoder_hidden_states.repeat(batch_size, 1, 1),
            encoder_hidden_states.repeat(batch_size, 1, 1)
        ])

        # Prepare added conditions
        added_cond_kwargs = {
            "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]),
            "time_ids": torch.cat([negative_add_time_ids, add_time_ids]),
        }

        # Predict noise with motion
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states_input,
            added_cond_kwargs=added_cond_kwargs,
            num_frames=num_frames,
        ).sample

        # Perform CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute previous noisy sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{num_inference_steps}")

    # Decode latents
    print("Decoding latents to images...")
    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents).sample

    # Post-process
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")

    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def save_video(frames, output_path, fps=8):
    """Save frames as video."""
    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in frames:
            writer.append_data(np.array(frame))
        writer.close()
        print(f"Video saved to {output_path}")
    except ImportError:
        print("imageio not installed. Saving frames as images instead.")
        output_dir = Path(output_path).parent / Path(output_path).stem
        output_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(output_dir / f"frame_{i:04d}.png")
        print(f"Frames saved to {output_dir}/")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Check motion module if provided - USING UTILS!
    if args.motion_module_path:
        print("Validating motion module file...")
        try:
            info = check_motion_module_compatibility(args.motion_module_path)
            print(f"Valid motion module with {info['total_params']:,} parameters\n")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return

    # Load models
    print("Loading models...")

    # VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    ).to(device)

    # Text encoders
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    ).to(device)

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
    ).to(device)

    # Tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
    )

    # UNet with motion modules - USING UTILS!
    print("Loading UNet with motion modules...")
    unet = load_unet_with_motion(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        motion_module_path=args.motion_module_path,
        motion_module_kwargs={"num_layers": args.motion_module_layers},
        torch_dtype=torch.float16,
        device=device,
    )
    unet.eval()

    # Scheduler
    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    # Generate video
    print()
    frames = generate_video(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler,
        prompt=args.prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        device=device,
    )

    # Save video
    output_path = Path(args.output_dir) / f"{args.prompt[:50].replace(' ', '_')}.mp4"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    save_video(frames, str(output_path), fps=args.fps)

    print(f"\nDone! Generated {len(frames)} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--motion_module_path", type=str, default=None, help="Path to trained motion_modules.pth")
    parser.add_argument("--motion_module_layers", type=int, default=2)

    # Generation
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs")

    args = parser.parse_args()
    main(args)