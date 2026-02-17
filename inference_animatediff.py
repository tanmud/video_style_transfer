import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from tqdm import tqdm

# Import the same utilities as training
from animatediff.utils import load_unet_with_motion


def encode_prompt(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt, device):
    """
    Encode text prompt for SDXL dual text encoders.
    EXACTLY the same as training!
    """
    # Tokenize with first tokenizer
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    # Tokenize with second tokenizer
    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    # Encode with first text encoder
    prompt_embeds = text_encoder(
        text_inputs.input_ids.to(device),
        output_hidden_states=True,
    ).hidden_states[-2]

    # Encode with second text encoder (for pooled embeddings)
    pooled_prompt_embeds = text_encoder_2(
        text_inputs_2.input_ids.to(device),
        output_hidden_states=True,
    )[0]

    # Get second text encoder's penultimate layer
    prompt_embeds_2 = text_encoder_2(
        text_inputs_2.input_ids.to(device),
        output_hidden_states=True,
    ).hidden_states[-2]

    # Concatenate embeddings
    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

    return prompt_embeds, pooled_prompt_embeds


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*80)
    print("Loading models (same as training)...")
    print("="*80)

    # 1. Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    vae.requires_grad_(False)
    vae.to(device)

    # 2. Load text encoders
    print("Loading text encoders...")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
    )
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    text_encoder.to(device)
    text_encoder_2.to(device)

    # 3. Load tokenizers
    print("Loading tokenizers...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
    )

    # 4. Load UNet with motion modules (SAME AS TRAINING!)
    print("Loading UNet with motion modules...")
    unet = load_unet_with_motion(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        motion_module_path=args.motion_module_path,
        motion_module_kwargs={"num_layers": args.motion_module_layers},
        torch_dtype=torch.float16,
        device=device,
    )
    unet.to(device)

    # 5. Load scheduler
    print("Loading scheduler...")
    scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    # Set timesteps for inference
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    print("All models loaded!")
    print("="*80)

    # Parse prompts
    prompts = args.validation_prompt.split(",")
    content_prompts = args.validation_prompt_content_forward.split(",") if args.validation_prompt_content_forward else [None] * len(prompts)
    style_prompts = args.validation_prompt_style_forward.split(",") if args.validation_prompt_style_forward else [None] * len(prompts)

    # Generate videos
    for idx, (prompt, content_prompt, style_prompt) in enumerate(zip(prompts, content_prompts, style_prompts)):
        print(f"\n{'='*80}")
        print(f"Video {idx+1}/{len(prompts)}")
        print(f"{'='*80}")
        print(f"Prompt: {prompt}")
        if content_prompt:
            print(f"Content: {content_prompt}")
        if style_prompt:
            print(f"Style: {style_prompt}")

        # Encode prompts
        with torch.no_grad():
            # Main prompt
            prompt_embeds, pooled_embeds = encode_prompt(
                text_encoder, text_encoder_2,
                tokenizer, tokenizer_2,
                prompt, device
            )

            # If content/style provided, encode and merge
            if content_prompt and style_prompt:
                content_embeds, content_pooled = encode_prompt(
                    text_encoder, text_encoder_2,
                    tokenizer, tokenizer_2,
                    content_prompt, device
                )
                style_embeds, style_pooled = encode_prompt(
                    text_encoder, text_encoder_2,
                    tokenizer, tokenizer_2,
                    style_prompt, device
                )
                # Average them (like training random selection)
                prompt_embeds = (content_embeds + style_embeds) / 2.0
                pooled_embeds = (content_pooled + style_pooled) / 2.0
                print("Using content/style merged embeddings")

            # Prepare SDXL time embeddings
            add_time_ids = torch.cat([
                torch.tensor([args.height, args.width]),  # original_size
                torch.tensor([0, 0]),                      # crops_coords_top_left
                torch.tensor([args.height, args.width]),  # target_size
            ]).unsqueeze(0).to(device, dtype=torch.float16)

            added_cond_kwargs = {
                "text_embeds": pooled_embeds,
                "time_ids": add_time_ids,
            }

            # Initialize random latents for all frames
            latents = torch.randn(
                (args.num_frames, 4, args.height // 8, args.width // 8),
                device=device,
                dtype=torch.float16,
            )

            # Scale by scheduler's init noise sigma
            latents = latents * scheduler.init_noise_sigma

            # Denoising loop (SAME AS TRAINING!)
            print(f"Generating {args.num_frames} frames...")
            for t in tqdm(scheduler.timesteps):
                # Expand for classifier-free guidance
                if args.guidance_scale > 1.0:
                    latent_model_input = torch.cat([latents] * 2)
                    prompt_embeds_input = torch.cat([
                        torch.zeros_like(prompt_embeds),  # Unconditional
                        prompt_embeds
                    ])
                    pooled_embeds_input = torch.cat([
                        torch.zeros_like(pooled_embeds),
                        pooled_embeds
                    ])
                    add_time_ids_input = torch.cat([add_time_ids] * 2)
                else:
                    latent_model_input = latents
                    prompt_embeds_input = prompt_embeds
                    pooled_embeds_input = pooled_embeds
                    add_time_ids_input = add_time_ids

                # Scale latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # Prepare timestep
                timestep = torch.tensor([t], device=device).expand(1)

                # Predict noise (CRITICAL: Pass num_frames!)
                noise_pred = unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds_input,
                    added_cond_kwargs={
                        "text_embeds": pooled_embeds_input,
                        "time_ids": add_time_ids_input,
                    },
                    num_frames=args.num_frames,  # ← MUST PASS THIS!
                ).sample

                # Perform guidance
                if args.guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Update latents
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Decode frames
            print("Decoding frames...")
            latents = latents / vae.config.scaling_factor

            frames = []
            for i in range(args.num_frames):
                frame_latent = latents[i:i+1].to(torch.float16)
                frame = vae.decode(frame_latent).sample
                frame = (frame / 2 + 0.5).clamp(0, 1)
                frame = frame.cpu().permute(0, 2, 3, 1).numpy()[0]
                frame = (frame * 255).astype(np.uint8)
                frames.append(Image.fromarray(frame))

            # Save video
            import os
            os.makedirs(args.save_dir, exist_ok=True)

            # Save as GIF
            output_path = os.path.join(args.save_dir, f"video_{idx:03d}.gif")
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000//args.fps,
                loop=0
            )
            print(f"Saved video to: {output_path}")

    print("\n" + "="*80)
    print("All videos generated successfully!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--motion_module_path", type=str, required=True)
    parser.add_argument("--motion_module_layers", type=int, default=2)

    # Video settings
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--fps", type=int, default=8)

    # Generation settings
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    # Prompts
    parser.add_argument("--validation_prompt", type=str, required=True)
    parser.add_argument("--validation_prompt_content_forward", type=str, default=None)
    parser.add_argument("--validation_prompt_style_forward", type=str, default=None)

    # Output
    parser.add_argument("--save_dir", type=str, default="output/")

    args = parser.parse_args()
    main(args)