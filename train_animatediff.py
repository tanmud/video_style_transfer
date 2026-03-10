import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from tqdm.auto import tqdm

from animatediff.utils import (
    load_unet_with_motion,
    freeze_spatial_layers,
    print_parameter_summary,
    save_checkpoint,
    get_trainable_parameters,
)
from animatediff.video_dataset import VideoDataset, collate_fn
# UnZipLoRA — spatial LoRA injection + forward type control
from unziplora_unet.utils import insert_unziplora_to_unet, unziplora_set_forward_type


def encode_prompt(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt, device):
    """Encode text prompt for SDXL dual text encoders."""
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


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    if accelerator.is_main_process:
        print("Loading models...")

    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else \
        torch.float16  if args.mixed_precision == "fp16" else \
        torch.float32
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=dtype,
    )
    vae.requires_grad_(False)
    vae.to(accelerator.device)

    # Text encoders
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    text_encoder.to(accelerator.device)
    text_encoder_2.to(accelerator.device)

    # Tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )

    # ── Step 1: Load UNet with randomly-initialized temporal transformers ──────
    if accelerator.is_main_process:
        print("\nLoading UNet with motion modules...")
    unet = load_unet_with_motion(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        motion_adapter_path=args.motion_adapter_path,
        torch_dtype=torch.float32,
        device=accelerator.device,
    )

    # ── Step 2: Inject Stage-1 UnZipLoRA spatial weights ─────────────────────
    # Must happen BEFORE freezing so LoRA params exist when freeze runs.
    if accelerator.is_main_process:
        print("\nInjecting UnZipLoRA spatial weights (Stage 1 outputs)...")
    insert_unziplora_to_unet(
        unet,
        args.unziplora_content_path,
        args.unziplora_style_path,
        args.unziplora_content_weight_path,
        args.unziplora_style_weight_path,
    )
    # Use both content + style pathways during training
    unziplora_set_forward_type(unet, type="both")

    # ── Step 3: Freeze everything except temporal transformers ────────────────
    # freeze_spatial_layers freezes all params whose name does NOT contain
    # "temporal" — this includes the LoRA params (spatial attention layers).
    freeze_spatial_layers(unet)
    if accelerator.is_main_process:
        print_parameter_summary(unet)

    # Noise scheduler (DDPM is fine for training)
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Optimizer — only temporal params have requires_grad=True after freeze
    trainable_params = get_trainable_parameters(unet)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset
    dataset = VideoDataset(
        args.instance_data_dir,
        num_frames=args.num_frames,
        resolution=args.resolution,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )

    # LR scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # ── Pre-compute text embeddings (only 2: instance + uncond) ──────────────
    if accelerator.is_main_process:
        print("\nPre-computing text embeddings...")
    with torch.no_grad():
        instance_prompt_embeds, instance_pooled_embeds = encode_prompt(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            args.instance_prompt, accelerator.device,
        )
        # Real empty-string embeddings for 10% CFG dropout
        uncond_prompt_embeds, uncond_pooled_embeds = encode_prompt(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            "", accelerator.device,
        )
    if accelerator.is_main_process:
        print(f"  Instance prompt : {args.instance_prompt}")
        print(f"  CFG dropout rate: 10%")

    if accelerator.is_main_process:
        accelerator.init_trackers(args.name, config=vars(args))

    # ── Training loop ─────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        print(f"\nStarting training for {args.max_train_steps} steps\n")

    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(args.num_train_epochs):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                frames = batch["frames"].to(accelerator.device)
                batch_size = frames.shape[0]
                num_frames = frames.shape[1]

                # (B, F, C, H, W) → (B*F, C, H, W) for VAE
                frames_flat = frames.reshape(-1, *frames.shape[2:]).to(dtype=vae.dtype)

                with torch.no_grad():
                    latents = vae.encode(frames_flat).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                # One timestep per batch item; repeat across frames for add_noise
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch_size,), device=latents.device,
                )
                timesteps_expanded = timesteps.repeat_interleave(num_frames)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps_expanded)

                # 10% CFG dropout: condition on empty string instead of instance prompt
                use_uncond = torch.rand(1).item() < 0.1
                if use_uncond:
                    encoder_hidden_states = uncond_prompt_embeds.repeat(batch_size, 1, 1)
                    pooled_embeds = uncond_pooled_embeds.repeat(batch_size, 1)
                else:
                    encoder_hidden_states = instance_prompt_embeds.repeat(batch_size, 1, 1)
                    pooled_embeds = instance_pooled_embeds.repeat(batch_size, 1)

                # SDXL added conditions
                add_time_ids = torch.cat([
                    torch.tensor([args.resolution, args.resolution]),
                    torch.tensor([0, 0]),
                    torch.tensor([args.resolution, args.resolution]),
                ]).unsqueeze(0).repeat(batch_size, 1).to(
                    accelerator.device, dtype=latents.dtype
                )
                added_cond_kwargs = {
                    "text_embeds": pooled_embeds,
                    "time_ids": add_time_ids,
                }

                # UNet takes (B*F, 4, H//8, W//8) + (B,) timestep + (B, seq, dim) embeds
                model_pred = unet(
                    noisy_latents,
                    timesteps,          # (B,) — UNet broadcasts over F frames
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    num_frames=num_frames,
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps_expanded)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.log_every == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        save_checkpoint(
                            unwrapped_unet, args.output_dir,
                            global_step, save_full_model=False,
                        )

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    # Save final checkpoint
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        save_checkpoint(unwrapped_unet, args.output_dir, "final", save_full_model=False)
        print(f"\nTraining complete! Saved to {args.output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Base model
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")

    # UnZipLoRA Stage-1 outputs (required for Stage-2)
    parser.add_argument("--unziplora_content_path", type=str, required=True,
                        help="Path to Stage-1 content LoRA dir (content/*.safetensors)")
    parser.add_argument("--unziplora_style_path", type=str, required=True,
                        help="Path to Stage-1 style LoRA dir (style/*.safetensors)")
    parser.add_argument("--unziplora_content_weight_path", type=str, required=True,
                        help="Path to mergercontent.pth")
    parser.add_argument("--unziplora_style_weight_path", type=str, required=True,
                        help="Path to mergerstyle.pth")
    parser.add_argument("--motion_adapter_path", type=str, required=True,
                        help="Path to pretrained MotionAdapter (e.g. guoyww/animatediff-motion-adapter-sdxl-beta)")

    # Data
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512)

    # Prompts — single instance prompt only (no random prompt switching)
    parser.add_argument("--instance_prompt", type=str, required=True)

    # Training
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--seed", type=str, default="0")

    # Logging
    parser.add_argument("--name", type=str, default="animatediff-stage2")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
