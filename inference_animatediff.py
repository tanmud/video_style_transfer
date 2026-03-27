import argparse
import os
from contextlib import nullcontext
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from tqdm import tqdm

from animatediff.utils import load_unet_with_motion
from animatediff.attention_processor import AnimateDiffAttnProcessor2_0
from unziplora_unet.utils import insert_unziplora_to_unet, unziplora_set_forward_type


def encode_prompt(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt, device):
    """Encode text prompt for SDXL. Identical to training."""
    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_inputs_2 = tokenizer_2(
        prompt, padding="max_length", max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    )
    prompt_embeds = text_encoder(
        text_inputs.input_ids.to(device), output_hidden_states=True,
    ).hidden_states[-2]
    pooled_prompt_embeds = text_encoder_2(
        text_inputs_2.input_ids.to(device), output_hidden_states=True,
    )[0]
    prompt_embeds_2 = text_encoder_2(
        text_inputs_2.input_ids.to(device), output_hidden_states=True,
    ).hidden_states[-2]
    return torch.cat([prompt_embeds, prompt_embeds_2], dim=-1), pooled_prompt_embeds


def save_video(frames, output_path, fps):
    try:
        import imageio
        imageio.mimsave(
            output_path,
            [np.array(f) for f in frames],
            fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
        )
        print(f"Saved: {output_path}")
        return True
    except ImportError:
        print("imageio not found — pip install imageio-ffmpeg")
        return False


def generate_video(unet, vae, text_encoder, text_encoder_2,
                   tokenizer, tokenizer_2, scheduler,
                   prompt, forward_type, args, device, output_path):
    """
    Denoising loop for one video.
    forward_type:
      "both"    → instance prompt, full content+style reconstruction
      "content" → re-contextualise content only (new scene, keep style)
      "style"   → re-contextualise style only  (new style, keep subject)
    """
    print(f"\n[type={forward_type}] {prompt}")

    # Set which UnZipLoRA pathway is active for this generation
    unziplora_set_forward_type(unet, type=forward_type)
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    with torch.no_grad():
        cond_embeds,   cond_pooled   = encode_prompt(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt, device)
        uncond_embeds, uncond_pooled = encode_prompt(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2, "", device)

    cond_embeds   = cond_embeds.to(dtype=unet.dtype)
    cond_pooled   = cond_pooled.to(dtype=unet.dtype)
    uncond_embeds = uncond_embeds.to(dtype=unet.dtype)
    uncond_pooled = uncond_pooled.to(dtype=unet.dtype)

    # SDXL time ids — (1, 6): original_size, crop_tl, target_size
    add_time_ids = torch.cat([
        torch.tensor([args.height, args.width], dtype=torch.float32),
        torch.tensor([0, 0],                   dtype=torch.float32),
        torch.tensor([args.height, args.width], dtype=torch.float32),
    ]).unsqueeze(0).to(device, dtype=unet.dtype)

    # Initial latent noise — shape matches training: (B=1, C=4, F, H//8, W//8)
    if args.seed >= 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None
    latents = torch.randn(
        (1, 4, args.num_frames, args.height // 8, args.width // 8),
        device=device, dtype=unet.dtype, generator=generator,
    ) * scheduler.init_noise_sigma

    # ── Denoising loop ────────────────────────────────────────────────────
    autocast_dtype = (torch.bfloat16 if args.mixed_precision == "bf16" else
                      torch.float16  if args.mixed_precision == "fp16" else None)
    ctx = (torch.autocast("cuda", dtype=autocast_dtype)
           if autocast_dtype else nullcontext())

    print(f"Denoising {args.num_frames} frames × {args.num_inference_steps} steps …")
    with torch.no_grad(), ctx:
        for t in tqdm(scheduler.timesteps):
            scaled  = scheduler.scale_model_input(latents, t)
            t_batch = torch.tensor([t], device=device)  # (1,) — B=1

            if args.guidance_scale > 1.0:
                noise_uncond = unet(
                    scaled, t_batch,
                    encoder_hidden_states=uncond_embeds,
                    added_cond_kwargs={"text_embeds": uncond_pooled,
                                       "time_ids":    add_time_ids},
                ).sample
                noise_cond = unet(
                    scaled, t_batch,
                    encoder_hidden_states=cond_embeds,
                    added_cond_kwargs={"text_embeds": cond_pooled,
                                       "time_ids":    add_time_ids},
                ).sample
                noise_pred = noise_uncond + args.guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = unet(
                    scaled, t_batch,
                    encoder_hidden_states=cond_embeds,
                    added_cond_kwargs={"text_embeds": cond_pooled,
                                       "time_ids":    add_time_ids},
                ).sample

            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # ── Decode frames ─────────────────────────────────────────────────────
    # latents: (1, 4, F, H//8, W//8) → squeeze → (4, F, H//8, W//8)
    # → permute → (F, 4, H//8, W//8) → decode per frame
    print("Decoding frames …")
    latents_dec = (latents.float() / vae.config.scaling_factor).squeeze(0).permute(1, 0, 2, 3)
    frames_out = []
    with torch.no_grad():
        for i in range(args.num_frames):
            frame = vae.decode(latents_dec[i:i+1].to(vae.dtype)).sample
            frame = (frame.float() / 2 + 0.5).clamp(0, 1)
            frame = frame.cpu().permute(0, 2, 3, 1).numpy()[0]
            frames_out.append(Image.fromarray((frame * 255).astype(np.uint8)))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not save_video(frames_out, output_path, args.fps):
        gif = output_path.replace(".mp4", ".gif")
        frames_out[0].save(gif, save_all=True, append_images=frames_out[1:],
                           duration=1000 // args.fps, loop=0)
        print(f"Saved as GIF: {gif}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("AnimateDiff + UnZipLoRA Inference")
    print("=" * 70)

    dtype = (torch.bfloat16 if args.mixed_precision == "bf16" else
             torch.float16  if args.mixed_precision == "fp16" else
             torch.float32)

    # VAE always fp32 — SDXL VAE is numerically unstable at lower precision
    print("Loading VAE …")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        torch_dtype=torch.float32,
    ).requires_grad_(False).to(device)

    print("Loading text encoders …")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    ).requires_grad_(False).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    ).requires_grad_(False).to(device)

    tokenizer  = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2")

    # ── UNet: load base SDXL + Stage-2 trained motion modules ────────────
    # --motion_adapter_path should point to your Stage-2 checkpoint dir,
    # e.g. "models/male_biker_video/checkpoint-final"
    # load_unet_with_motion looks for motion_modules.pth inside that dir.
    # If you pass the raw HF adapter ID instead, Stage-2 weights are ignored.
    print("Loading UNet + Stage-2 motion modules …")
    unet, _ = load_unet_with_motion(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        motion_adapter_path=args.motion_adapter_path,
        torch_dtype=dtype, device=device,
    )

    # ── Inject Stage-1 UnZipLoRA spatial weights ──────────────────────────
    # If Option C was used during Stage-2 training, pass the Stage-2 merger
    # paths (merger_*_stage2.pth) as --unziplora_*_weight_path instead of
    # the Stage-1 files.
    print("Injecting UnZipLoRA spatial weights …")
    insert_unziplora_to_unet(
        unet,
        args.unziplora_content_path,
        args.unziplora_style_path,
        args.unziplora_content_weight_path,
        args.unziplora_style_weight_path,
    )
    unet.requires_grad_(False).to(device)

    # Spatial → AnimateDiffAttnProcessor2_0; motion layers keep their processor
    new_processors = {}
    for name, proc in unet.attn_processors.items():
        new_processors[name] = (proc if "motion_modules" in name
                                else AnimateDiffAttnProcessor2_0())
    unet.set_attn_processor(new_processors)

    scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    print("All models loaded!\n" + "=" * 70)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Three generation modes ────────────────────────────────────────────
    if args.instance_prompt:
        generate_video(
            unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            scheduler, prompt=args.instance_prompt, forward_type="both",
            args=args, device=device,
            output_path=os.path.join(args.save_dir, "video_both.mp4"),
        )

    if args.content_prompt:
        generate_video(
            unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            scheduler, prompt=args.content_prompt, forward_type="content",
            args=args, device=device,
            output_path=os.path.join(args.save_dir, "video_content.mp4"),
        )

    if args.style_prompt:
        generate_video(
            unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            scheduler, prompt=args.style_prompt, forward_type="style",
            args=args, device=device,
            output_path=os.path.join(args.save_dir, "video_style.mp4"),
        )

    print("\n" + "=" * 70 + "\nDone.\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)

    # ── UnZipLoRA paths ───────────────────────────────────────────────────
    parser.add_argument("--unziplora_content_path",        type=str, required=True)
    parser.add_argument("--unziplora_style_path",          type=str, required=True)
    parser.add_argument("--unziplora_content_weight_path", type=str, required=True,
                        help="Stage-1 merger_content.pth, or Stage-2 merger_content_stage2.pth "
                             "if --unfreeze_mergers was used during training.")
    parser.add_argument("--unziplora_style_weight_path",   type=str, required=True,
                        help="Stage-1 merger_style.pth, or Stage-2 merger_style_stage2.pth "
                             "if --unfreeze_mergers was used during training.")

    # ── Motion adapter — MUST point to Stage-2 checkpoint, not base HF adapter ──
    parser.add_argument("--motion_adapter_path", type=str, required=True,
                        help="Path to Stage-2 checkpoint dir containing motion_modules.pth, "
                             "e.g. models/male_biker_video/checkpoint-final. "
                             "DO NOT pass the raw HF adapter ID here — doing so silently "
                             "ignores all Stage-2 training.")

    # ── Prompts ───────────────────────────────────────────────────────────
    parser.add_argument("--instance_prompt", type=str, default=None,
                        help="type=both  — reconstruct trained subject with motion")
    parser.add_argument("--content_prompt", type=str, default=None,
                        help="type=content — new scene, preserve style")
    parser.add_argument("--style_prompt",   type=str, default=None,
                        help="type=style   — new style, preserve subject")

    # ── Video settings ────────────────────────────────────────────────────
    parser.add_argument("--num_frames", type=int,   default=16)
    parser.add_argument("--height",     type=int,   default=1024)
    parser.add_argument("--width",      type=int,   default=1024)
    parser.add_argument("--fps",        type=int,   default=8)

    # ── Generation settings ───────────────────────────────────────────────
    parser.add_argument("--num_inference_steps", type=int,   default=50)
    parser.add_argument("--guidance_scale",      type=float, default=7.5)
    parser.add_argument("--seed",                type=int,   default=42,
                        help="RNG seed for reproducible latent init. -1 = random.")
    parser.add_argument("--mixed_precision",     type=str,   default="bf16")

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument("--save_dir", type=str, default="output/")

    args = parser.parse_args()
    main(args)
