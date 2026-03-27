import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import AutoencoderKL, EulerDiscreteScheduler
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
from animatediff.attention_processor import AnimateDiffAttnProcessor2_0
from animatediff.temporal_lora import (
    inject_temporal_lora,
    build_spatial_lora_index,
    compute_orth_loss,
)
from unziplora_unet.utils import insert_unziplora_to_unet, unziplora_set_forward_type


def encode_prompt(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt, device):
    """Encode text prompt for SDXL dual text encoders."""
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
    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    if accelerator.is_main_process:
        print("Loading models...")

    train_dtype = (
        torch.bfloat16 if args.mixed_precision == "bf16" else
        torch.float16  if args.mixed_precision == "fp16" else
        torch.float32
    )

    # VAE must stay fp32 — SDXL VAE is numerically unstable at lower precision
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.requires_grad_(False).to(accelerator.device)

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    text_encoder.requires_grad_(False).to(accelerator.device)
    text_encoder_2.requires_grad_(False).to(accelerator.device)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )

    # ── Step 1: Load UNet with motion modules ────────────────────────────────
    if accelerator.is_main_process:
        print("\nLoading UNet with motion modules...")
    unet, motion_max_seq = load_unet_with_motion(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        motion_adapter_path=args.motion_adapter_path,
        torch_dtype=train_dtype,
        device=accelerator.device,
    )

    if args.enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if motion_max_seq is not None and args.num_frames > motion_max_seq:
        raise ValueError(
            f"--num_frames={args.num_frames} exceeds motion adapter max "
            f"sequence length {motion_max_seq}."
        )

    # ── Step 2: Inject Stage-1 UnZipLoRA spatial weights ─────────────────────
    if accelerator.is_main_process:
        print("\nInjecting UnZipLoRA spatial weights (Stage-1 outputs)...")
    insert_unziplora_to_unet(
        unet,
        args.unziplora_content_path,
        args.unziplora_style_path,
        args.unziplora_content_weight_path,
        args.unziplora_style_weight_path,
    )

    # Spatial layers get AnimateDiffAttnProcessor2_0; motion layers keep theirs
    new_processors = {}
    for name, proc in unet.attn_processors.items():
        new_processors[name] = (proc if "motion_modules" in name
                                else AnimateDiffAttnProcessor2_0())
    unet.set_attn_processor(new_processors)
    unziplora_set_forward_type(unet, type="both")

    # ── Step 2.5: Inject temporal LoRA (Option B) ─────────────────────────────
    # Wraps each to_q/k/v/out inside motion modules with TemporalLoRALinear.
    # Base motion weights become frozen; only lora_A and lora_B are trained.
    # Checkpoint save merges delta back in → motion_modules.pth is unchanged
    # in format, inference code needs no modifications.
    if args.temporal_lora_rank > 0:
        n_wrapped = inject_temporal_lora(
            unet, rank=args.temporal_lora_rank, alpha=args.temporal_lora_alpha,
        )
        if accelerator.is_main_process:
            print(f"  Temporal LoRA: {n_wrapped} layers wrapped "
                  f"(rank={args.temporal_lora_rank}, alpha={args.temporal_lora_alpha})")

    # ── Step 3: Freeze spatial layers ─────────────────────────────────────────
    # - Spatial weights (SDXL base + Stage-1 LoRAs) → always frozen
    # - motion_modules .base.* → frozen (TemporalLoRALinear bases)
    # - motion_modules .lora_A/.lora_B → trainable (Option B deltas)
    # - motion_modules norms/pos_embed  → trainable
    # - merge_content / merge_style     → trainable iff --unfreeze_mergers (Option C)
    freeze_spatial_layers(unet, unfreeze_mergers=args.unfreeze_mergers)
    if accelerator.is_main_process:
        print_parameter_summary(unet)

    # ── Step 3.5: Build spatial LoRA index for orthogonality loss (Option B) ──
    # One-time scan; reused every step — no per-step module iteration.
    if args.lambda_orth > 0 and args.temporal_lora_rank > 0:
        spatial_index = build_spatial_lora_index(unet)
        if accelerator.is_main_process:
            print(f"  Spatial-temporal LoRA pairs: {len(spatial_index)}")
    else:
        spatial_index = {}

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    trainable_params = get_trainable_parameters(unet)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = VideoDataset(
        args.instance_data_dir, num_frames=args.num_frames, resolution=args.resolution,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.dataloader_num_workers, collate_fn=collate_fn,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # ── Pre-compute text embeddings ───────────────────────────────────────────
    if accelerator.is_main_process:
        print("\nPre-computing text embeddings...")
    with torch.no_grad():
        instance_prompt_embeds, instance_pooled_embeds = encode_prompt(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            args.instance_prompt, accelerator.device,
        )
        uncond_prompt_embeds, uncond_pooled_embeds = encode_prompt(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            "", accelerator.device,
        )
    if accelerator.is_main_process:
        print(f"  Instance prompt: {args.instance_prompt}")
    accelerator.init_trackers(args.name, config=vars(args))

    # ── Training loop ─────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        print(f"\nStarting training for {args.max_train_steps} steps\n")
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                frames     = batch["frames"].to(accelerator.device)
                batch_size = frames.shape[0]
                num_frames = frames.shape[1]

                # ── VAE encode ────────────────────────────────────────────────
                frames_flat = frames.reshape(-1, *frames.shape[2:]).to(torch.float32)
                with torch.no_grad():
                    latents_flat = vae.encode(frames_flat).latent_dist.sample()
                    latents_flat = latents_flat * vae.config.scaling_factor
                latents_flat = latents_flat.to(train_dtype)
                # latents_flat: (B*F, 4, H//8, W//8)

                # ── Noise + timesteps ─────────────────────────────────────────
                noise_flat = torch.randn_like(latents_flat)
                timesteps  = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch_size,), device=latents_flat.device,
                ).long()
                timesteps_expanded = timesteps.repeat_interleave(num_frames)  # (B*F,)
                noisy_latents_flat = noise_scheduler.add_noise(
                    latents_flat, noise_flat, timesteps_expanded
                )

                # ── Reshape to 5D for UNetMotionModel ─────────────────────────
                noisy_latents_5d = (
                    noisy_latents_flat
                    .reshape(batch_size, num_frames, *noisy_latents_flat.shape[1:])
                    .permute(0, 2, 1, 3, 4).contiguous()
                )

                # ── Text conditioning — keep at (B, ...) ─────────────────────
                # UNetMotionModel.forward calls repeat_interleave(num_frames)
                # internally. Pre-expanding to (B*F) causes double-expansion crash.
                use_uncond = torch.rand(1).item() < 0.1
                if use_uncond:
                    encoder_hidden_states = uncond_prompt_embeds.repeat(batch_size, 1, 1)
                    pooled_embeds         = uncond_pooled_embeds.repeat(batch_size, 1)
                else:
                    encoder_hidden_states = instance_prompt_embeds.repeat(batch_size, 1, 1)
                    pooled_embeds         = instance_pooled_embeds.repeat(batch_size, 1)

                add_time_ids = torch.cat([
                    torch.tensor([args.resolution, args.resolution]),
                    torch.tensor([0, 0]),
                    torch.tensor([args.resolution, args.resolution]),
                ]).unsqueeze(0).repeat(batch_size, 1).to(
                    accelerator.device, dtype=train_dtype
                )

                # ── UNet forward ──────────────────────────────────────────────
                model_pred = unet(
                    noisy_latents_5d,                        # (B, C, F, H, W)
                    timesteps,                               # (B,)
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": pooled_embeds,
                        "time_ids":    add_time_ids,
                    },
                ).sample

                # ── Diffusion target ──────────────────────────────────────────
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise_flat
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    if accelerator.is_main_process and global_step == 0:
                        print("WARNING: v_prediction active — confirm intended")
                    latents_5d = (latents_flat
                        .reshape(batch_size, num_frames, *latents_flat.shape[1:])
                        .permute(0, 2, 1, 3, 4))
                    noise_5d = (noise_flat
                        .reshape(batch_size, num_frames, *noise_flat.shape[1:])
                        .permute(0, 2, 1, 3, 4))
                    target_5d = noise_scheduler.get_velocity(latents_5d, noise_5d, timesteps)
                    target = (target_5d.permute(0, 2, 1, 3, 4)
                              .reshape(batch_size * num_frames, *latents_flat.shape[1:]))
                else:
                    raise ValueError(f"Unknown prediction_type: "
                                     f"{noise_scheduler.config.prediction_type}")

                model_pred_flat = (
                    model_pred.permute(0, 2, 1, 3, 4).contiguous()
                    .reshape(batch_size * num_frames, *noisy_latents_flat.shape[1:])
                )
                loss_mse = F.mse_loss(
                    model_pred_flat.float(), target.float(), reduction="mean"
                )

                # ── Orthogonality loss (Option B) ─────────────────────────────
                # L_orth = (1/N) sum_l [ ||(B_t A_t)^T (B_c A_c)||_F^2
                #                      + ||(B_t A_t)^T (B_s A_s)||_F^2 ]
                # Both sides are rank-r deltas. Grad flows only through
                # temporal lora_A/lora_B. Spatial deltas are detached.
                if args.lambda_orth > 0 and spatial_index:
                    loss_orth = compute_orth_loss(unet, spatial_index, args.lambda_orth)
                else:
                    loss_orth = torch.tensor(0.0, device=loss_mse.device)

                loss = loss_mse + loss_orth

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
                        "loss":      loss.detach().item(),
                        "loss_mse":  loss_mse.detach().item(),
                        "loss_orth": loss_orth.detach().item(),
                        "lr":        lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(unet)
                        save_checkpoint(
                            unwrapped, args.output_dir, global_step,
                            save_full_model=False,
                            save_mergers=args.unfreeze_mergers,
                        )

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet)
        save_checkpoint(
            unwrapped, args.output_dir, "final",
            save_full_model=False,
            save_mergers=args.unfreeze_mergers,
        )
        print(f"\nTraining complete! Saved to {args.output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ── Base model ────────────────────────────────────────────────────────────
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")

    # ── UnZipLoRA paths ───────────────────────────────────────────────────────
    parser.add_argument("--unziplora_content_path",        type=str, required=True)
    parser.add_argument("--unziplora_style_path",          type=str, required=True)
    parser.add_argument("--unziplora_content_weight_path", type=str, required=True,
                        help="merger_content.pth (Stage-1) or merger_content_stage2.pth")
    parser.add_argument("--unziplora_style_weight_path",   type=str, required=True,
                        help="merger_style.pth (Stage-1) or merger_style_stage2.pth")

    # ── Motion adapter ────────────────────────────────────────────────────────
    parser.add_argument("--motion_adapter_path", type=str, required=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--instance_prompt",   type=str, required=True)
    parser.add_argument("--num_frames",        type=int, default=16)
    parser.add_argument("--resolution",        type=int, default=512)

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir",          type=str, default="./outputs")
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--log_every",           type=int, default=10)
    parser.add_argument("--name",                type=str, default="animatediff-stage2")
    parser.add_argument("--report_to",           type=str, default="wandb")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--train_batch_size",             type=int,   default=1)
    parser.add_argument("--num_train_epochs",             type=int,   default=100)
    parser.add_argument("--max_train_steps",              type=int,   default=1000)
    parser.add_argument("--gradient_accumulation_steps",  type=int,   default=4)
    parser.add_argument("--enable_gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision",              type=str,   default="bf16")
    parser.add_argument("--seed",                         type=str,   default="0")
    parser.add_argument("--dataloader_num_workers",       type=int,   default=0)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    parser.add_argument("--learning_rate",     type=float, default=2e-5)
    parser.add_argument("--lr_scheduler",      type=str,   default="cosine")
    parser.add_argument("--lr_warmup_steps",   type=int,   default=100)
    parser.add_argument("--adam_beta1",        type=float, default=0.9)
    parser.add_argument("--adam_beta2",        type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon",      type=float, default=1e-8)
    parser.add_argument("--max_grad_norm",     type=float, default=0.5)

    # ── Option B: Temporal LoRA + orthogonality loss ──────────────────────────
    parser.add_argument("--temporal_lora_rank",  type=int,   default=32,
                        help="Rank for temporal LoRA on motion attention layers. "
                             "0 = disabled (full motion weight training, no orth loss).")
    parser.add_argument("--temporal_lora_alpha", type=float, default=1.0,
                        help="LoRA alpha. scale = alpha / rank.")
    parser.add_argument("--lambda_orth",         type=float, default=1e-4,
                        help="Weight for temporal-spatial LoRA orthogonality loss. "
                             "0 = disabled. Range: 1e-4 (conservative) to 1e-3 (standard). "
                             "Monitor loss_orth in wandb vs loss_mse.")

    # ── Option C: Unfreeze merger scalars ─────────────────────────────────────
    parser.add_argument("--unfreeze_mergers", action="store_true",
                        help="Unfreeze Stage-1 merger scalars (merge_content/style) "
                             "so they can adapt to temporal context. Saved as "
                             "merger_*_stage2.pth alongside motion_modules.pth.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
