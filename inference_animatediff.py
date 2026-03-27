import argparse
import os
from contextlib import nullcontext
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, EulerDiscreteScheduler # ← DDIMScheduler, NOT DDPM
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from tqdm import tqdm

from animatediff.utils import load_unet_with_motion
from animatediff.attention_processor import AnimateDiffAttnProcessor2_0
from unziplora_unet.utils import insert_unziplora_to_unet, unziplora_set_forward_type


def encode_prompt(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt, device):
    """Encode text prompt for SDXL. Identical to training."""
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


def save_video_mp4(frames, output_path, fps):
    """Save frames as MP4 video."""
    try:
        import imageio
        frames_np = [np.array(frame) for frame in frames]
        imageio.mimsave(output_path, frames_np, fps=fps,
                        codec="libx264", quality=8, pixelformat="yuv420p")
        return True
    except ImportError:
        print("imageio not found — install: pip install imageio-ffmpeg")
        return False


def generate_video(
    unet, vae,
    text_encoder, text_encoder_2,
    tokenizer, tokenizer_2,
    scheduler,
    prompt, forward_type,
    args, device, output_path,
):
    """
    Run the full denoising loop for one video.

    forward_type: "both"    → instance prompt, full content+style
                  "content" → re-contextualize content only
                  "style"   → re-contextualize style only
    """
    print(f"\n[type={forward_type}] {prompt}")
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    # Set which UnZipLoRA pathway is active for this generation
    unziplora_set_forward_type(unet, type=forward_type)

    with torch.no_grad():
        # Conditioning embeddings
        cond_embeds, cond_pooled = encode_prompt(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt, device
        )
        # Real empty-string uncond — never use zero tensors
        uncond_embeds, uncond_pooled = encode_prompt(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2, "", device
        )

        cond_embeds=cond_embeds.to(dtype=unet.dtype)
        cond_pooled=cond_pooled.to(dtype=unet.dtype)
        uncond_embeds=uncond_embeds.to(dtype=unet.dtype)
        uncond_pooled=uncond_pooled.to(dtype=unet.dtype)

    # SDXL time embeddings — (B=1, 6)
    add_time_ids = torch.cat([
        torch.tensor([args.height, args.width]),
        torch.tensor([0, 0]),
        torch.tensor([args.height, args.width]),
    ]).unsqueeze(0).to(device, dtype=unet.dtype)

    # Initial latents: (B*F, 4, H//8, W//8) with B=1
    latents = torch.randn(
        (1, 4, args.num_frames, args.height // 8, args.width // 8),
        device=device, dtype=unet.dtype,
    )
    latents = latents * scheduler.init_noise_sigma

    print(f"Denoising {args.num_frames} frames ({args.num_inference_steps} steps)...")
    for t in tqdm(scheduler.timesteps):
        # Scale latents by current sigma
        scaled = scheduler.scale_model_input(latents, t)
        # Timestep broadcast: (B,) = (1,) for B=1
        t_batch = torch.tensor([t], device=device).expand(args.num_frames)

        with torch.no_grad():
            autocast_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else \
                             torch.float16  if args.mixed_precision == "fp16" else None
            ctx = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else nullcontext()
            with ctx:
                if args.guidance_scale > 1.0:
                    # Two separate passes — avoids shape ambiguity with batched CFG
                    noise_pred_uncond = unet(
                        scaled,
                        t_batch,
                        encoder_hidden_states=uncond_embeds,    # (1, seq, dim)
                        added_cond_kwargs={
                            "text_embeds": uncond_pooled,       # (1, 1280)
                            "time_ids": add_time_ids,           # (1, 6)
                        },
                    ).sample
                    noise_pred_text = unet(
                        scaled,
                        t_batch,
                        encoder_hidden_states=cond_embeds,
                        added_cond_kwargs={
                            "text_embeds": cond_pooled,
                            "time_ids": add_time_ids,
                        },
                    ).sample
                    noise_pred = (
                        noise_pred_uncond
                        + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    )
                else:
                    noise_pred = unet(
                        scaled,
                        t_batch,
                        encoder_hidden_states=cond_embeds,
                        added_cond_kwargs={
                            "text_embeds": cond_pooled,
                            "time_ids": add_time_ids,
                        },
                    ).sample

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # AFTER — reshape (1, 4, F, H//8, W//8) → (F, 4, H//8, W//8) for per-frame decode
    print("Decoding frames...")
    latents_dec = (latents / vae.config.scaling_factor).squeeze(0).permute(1, 0, 2, 3)
    frames = []
    for i in range(args.num_frames):
        with torch.no_grad():
            frame = vae.decode(latents_dec[i:i+1].to(vae.dtype)).sample
        frame = frame.to(torch.float32)
        frame = (frame / 2 + 0.5).clamp(0, 1)
        frame = frame.cpu().permute(0, 2, 3, 1).numpy()[0]
        frames.append(Image.fromarray((frame * 255).astype(np.uint8)))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if save_video_mp4(frames, output_path, args.fps):
        print(f"Saved: {output_path}")
    else:
        gif_path = output_path.replace(".mp4", ".gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=1000 // args.fps, loop=0)
        print(f"Saved as GIF: {gif_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("AnimateDiff + UnZipLoRA Inference")
    print("=" * 70)

    print("Loading VAE...")
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else \
        torch.float16  if args.mixed_precision == "fp16" else \
        torch.float32
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.requires_grad_(False).to(device)

    # 2. Text encoders
    print("Loading text encoders...")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    text_encoder.requires_grad_(False).to(device)
    text_encoder_2.requires_grad_(False).to(device)

    # 3. Tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )

    # 4. UNet + trained motion modules
    print("Loading UNet with trained motion modules...")
    unet, _= load_unet_with_motion(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        motion_adapter_path=args.motion_adapter_path,
        torch_dtype=dtype,
        device=device,
    )

    # 5. Inject UnZipLoRA spatial weights (must follow motion load)
    print("Injecting UnZipLoRA spatial weights...")
    insert_unziplora_to_unet(
        unet,
        args.unziplora_content_path,
        args.unziplora_style_path,
        args.unziplora_content_weight_path,
        args.unziplora_style_weight_path,
    )
    unet.requires_grad_(False).to(device)

    new_processors = {}
    for name, proc in unet.attn_processors.items():
        if "motion_modules" not in name:
            new_processors[name] = AnimateDiffAttnProcessor2_0()
        else:
            new_processors[name] = proc
    unet.set_attn_processor(new_processors)

    # 6. DDIM scheduler — works correctly at 50 inference steps (DDPM does NOT)
    print("Loading DDIMScheduler...")
    scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    print("All models loaded!\n" + "=" * 70)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Three generation modes ────────────────────────────────────────────────
    # Mode 1 — type="both": full subject with learned motion
    if args.instance_prompt:
        generate_video(
            unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            scheduler,
            prompt=args.instance_prompt,
            forward_type="both",
            args=args, device=device,
            output_path=os.path.join(args.save_dir, "video_both.mp4"),
        )

    # Mode 2 — type="content": swap content, preserve learned style
    if args.content_prompt:
        generate_video(
            unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            scheduler,
            prompt=args.content_prompt,
            forward_type="content",
            args=args, device=device,
            output_path=os.path.join(args.save_dir, "video_content.mp4"),
        )

    # Mode 3 — type="style": swap style, preserve learned content
    if args.style_prompt:
        generate_video(
            unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            scheduler,
            prompt=args.style_prompt,
            forward_type="style",
            args=args, device=device,
            output_path=os.path.join(args.save_dir, "video_style.mp4"),
        )

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Base model
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)

    # UnZipLoRA paths (Stage-1 outputs)
    parser.add_argument("--unziplora_content_path", type=str, required=True)
    parser.add_argument("--unziplora_style_path", type=str, required=True)
    parser.add_argument("--unziplora_content_weight_path", type=str, required=True)
    parser.add_argument("--unziplora_style_weight_path", type=str, required=True)
    parser.add_argument("--motion_adapter_path", type=str, required=True,
                        help="Dir saved by save_checkpoint containing motion_modules.pth, OR a HF hub adapter")
    # Generation prompts — pass whichever modes you want
    parser.add_argument("--instance_prompt", type=str, default=None,
                        help="type=both  — full subject reconstruction")
    parser.add_argument("--content_prompt", type=str, default=None,
                        help="type=content — re-contextualize content")
    parser.add_argument("--style_prompt", type=str, default=None,
                        help="type=style  — re-contextualize style")

    # Video settings
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--mixed_precision", type=str, default="no")

    # Generation settings
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    # Output

    parser.add_argument("--save_dir", type=str, default="output/")

    args = parser.parse_args()
    main(args)
