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

# Import our utilities
from animatediff.utils import (
    load_unet_with_motion,
    freeze_spatial_layers,
    unfreeze_all_layers,
    print_parameter_summary,
    save_checkpoint,
    get_trainable_parameters,
)

# Import flexible video dataset
from animatediff.video_dataset import VideoDataset, collate_fn


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

    # Concatenate
    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

    return prompt_embeds, pooled_prompt_embeds


def main(args):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # Load models
    if accelerator.is_main_process:
        print("Loading models...")

    # VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    vae.requires_grad_(False)
    vae.to(accelerator.device)

    # Text encoders
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
    text_encoder.to(accelerator.device)
    text_encoder_2.to(accelerator.device)

    # Tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
    )

    # UNet with motion modules
    if accelerator.is_main_process:
        print("\nLoading UNet with motion modules...")
    unet = load_unet_with_motion(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        motion_module_kwargs={"num_layers": args.motion_module_layers},
        torch_dtype=torch.float32,
        device=accelerator.device,
    )

    # Freeze/unfreeze
    if args.train_motion_only:
        freeze_spatial_layers(unet)
    else:
        unfreeze_all_layers(unet)

    if accelerator.is_main_process:
        print_parameter_summary(unet)

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    # Optimizer
    trainable_params = get_trainable_parameters(unet)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset (with better error messages)
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

    # Prepare
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # PRE-COMPUTE TEXT EMBEDDINGS (like your UnzipLoRA training)
    if accelerator.is_main_process:
        print("\nPre-computing text embeddings for prompts...")

    with torch.no_grad():
        # Instance prompt (main training prompt)
        instance_prompt_embeds, instance_pooled_embeds = encode_prompt(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            args.instance_prompt, accelerator.device
        )

        # Content forward prompt
        content_prompt_embeds, content_pooled_embeds = encode_prompt(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            args.content_forward_prompt, accelerator.device
        )

        # Style forward prompt
        style_prompt_embeds, style_pooled_embeds = encode_prompt(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            args.style_forward_prompt, accelerator.device
        )

    if accelerator.is_main_process:
        print(f"  Instance prompt: {args.instance_prompt}")
        print(f"  Content forward: {args.content_forward_prompt}")
        print(f"  Style forward: {args.style_forward_prompt}")

    # Initialize trackers
    if accelerator.is_main_process:
        config = vars(args)
        accelerator.init_trackers(args.name, config=config)

    # Training loop
    if accelerator.is_main_process:
        print(f"\nStarting training for {args.max_train_steps} steps\n")

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Get frames
                frames = batch["frames"].to(accelerator.device)
                batch_size, num_frames = frames.shape[0], frames.shape[1]

                # Flatten for VAE: (B, F, C, H, W) → (B*F, C, H, W)
                frames_flat = frames.reshape(-1, *frames.shape[2:]).to(dtype=vae.dtype)

                # Encode to latents
                with torch.no_grad():
                    latents = vae.encode(frames_flat).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                )
                timesteps = timesteps.repeat_interleave(num_frames)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # USE THE PROMPTS (randomly choose which one to condition on)
                prompt_choice = torch.rand(1).item()
                if prompt_choice < 0.33:
                    # Use instance prompt (combined)
                    encoder_hidden_states = instance_prompt_embeds.repeat(batch_size, 1, 1)
                    pooled_embeds = instance_pooled_embeds.repeat(batch_size, 1)
                elif prompt_choice < 0.66:
                    # Use content forward prompt
                    encoder_hidden_states = content_prompt_embeds.repeat(batch_size, 1, 1)
                    pooled_embeds = content_pooled_embeds.repeat(batch_size, 1)
                else:
                    # Use style forward prompt
                    encoder_hidden_states = style_prompt_embeds.repeat(batch_size, 1, 1)
                    pooled_embeds = style_pooled_embeds.repeat(batch_size, 1)

                # SDXL added conditions
                add_time_ids = torch.cat([
                    torch.tensor([args.resolution, args.resolution]),
                    torch.tensor([0, 0]),
                    torch.tensor([args.resolution, args.resolution]),
                ]).unsqueeze(0).repeat(batch_size, 1).to(accelerator.device, dtype=latents.dtype)

                added_cond_kwargs = {
                    "text_embeds": pooled_embeds,
                    "time_ids": add_time_ids,
                }

                # Predict noise with motion
                model_pred = unet(
                    noisy_latents,
                    timesteps[:batch_size],
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    num_frames=num_frames,
                ).sample

                # Compute loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Backprop
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log
                if global_step % args.log_every == 0:
                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        save_checkpoint(
                            unwrapped_unet,
                            args.output_dir,
                            global_step,
                            save_full_model=False,
                        )

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Save final
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        save_checkpoint(
            unwrapped_unet,
            args.output_dir,
            "final",
            save_full_model=False,
        )
        print(f"\nTraining complete! Saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--motion_module_layers", type=int, default=2)
    parser.add_argument("--train_motion_only", action="store_true")

    # Data
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512)

    # Training prompts
    parser.add_argument("--instance_prompt", type=str, default="")
    parser.add_argument("--content_forward_prompt", type=str, default="")
    parser.add_argument("--style_forward_prompt", type=str, default="")

    # Validation prompts
    parser.add_argument("--validation_content", type=str, default="")
    parser.add_argument("--validation_style", type=str, default="")
    parser.add_argument("--validation_prompt", type=str, default="")
    parser.add_argument("--validation_prompt_content", type=str, default="")
    parser.add_argument("--validation_prompt_style", type=str, default="")
    parser.add_argument("--validation_steps", type=int, default=500)

    # Training
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
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
    parser.add_argument("--name", type=str, default="animatediff")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)