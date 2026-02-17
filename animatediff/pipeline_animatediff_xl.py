import torch
from typing import Optional, Union, List, Callable, Dict, Any
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from animatediff.utils import load_unet_with_motion


class AnimateDiffUnZipLoRAPipeline(StableDiffusionXLPipeline):
    """
    Pipeline for video generation using AnimateDiff with SDXL.

    Extends StableDiffusionXLPipeline to add temporal attention (motion modules)
    for video generation.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load pipeline and inject motion modules into UNet.

        Args:
            pretrained_model_name_or_path: Path to SDXL model
            motion_module_path: Path to trained motion module weights
            motion_module_layers: Number of temporal attention layers (default: 2)
        """
        # Extract motion module arguments
        motion_module_path = kwargs.pop("motion_module_path", None)
        motion_module_layers = kwargs.pop("motion_module_layers", 2)

        # Load base SDXL pipeline
        pipe = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Inject motion modules into UNet
        if motion_module_path is not None:
            print(f"Loading motion modules from: {motion_module_path}")
            pipe.unet = load_unet_with_motion(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                motion_module_path=motion_module_path,
                motion_module_kwargs={"num_layers": motion_module_layers},
                torch_dtype=pipe.unet.dtype,
                device=pipe.device,
            )
            print("Motion modules loaded successfully!")
        else:
            print("No motion_module_path provided - using base UNet without motion")

        return pipe

    def prepare_latents_video(
        self,
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """
        Prepare initial latents for video generation.

        Shape: (batch_size * num_frames, num_channels_latents, height // 8, width // 8)
        """
        shape = (
            batch_size * num_frames,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch "
                f"size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            latents = latents.to(device)

        # Scale by scheduler's init noise sigma
        latents = latents * self.scheduler.init_noise_sigma

        return latents

    def decode_latents_video(self, latents, num_frames):
        """
        Decode video latents frame by frame.

        Args:
            latents: (batch_size * num_frames, 4, H, W)
            num_frames: Number of frames

        Returns:
            frames: (num_frames, 3, height, width)
        """
        # Unscale latents
        latents = 1 / self.vae.config.scaling_factor * latents

        # Decode frame by frame (VAE can't handle temporal dimension)
        frames = []
        for i in range(num_frames):
            frame_latent = latents[i:i+1]  # (1, 4, H, W)
            frame = self.vae.decode(frame_latent, return_dict=False)[0]
            frames.append(frame)

        # Stack frames: (num_frames, 3, H, W)
        frames = torch.cat(frames, dim=0)

        return frames

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_content: Optional[Union[str, List[str]]] = None,
        prompt_style: Optional[Union[str, List[str]]] = None,
        num_frames: int = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[tuple] = None,
        crops_coords_top_left: tuple = (0, 0),
        target_size: Optional[tuple] = None,
        negative_original_size: Optional[tuple] = None,
        negative_crops_coords_top_left: tuple = (0, 0),
        negative_target_size: Optional[tuple] = None,
        fps: int = 8,
    ):
        """
        Generate video using AnimateDiff.

        Args:
            prompt: Main prompt for generation
            prompt_2: Secondary prompt (SDXL uses two)
            prompt_content: Content-specific prompt (optional, for control)
            prompt_style: Style-specific prompt (optional, for control)
            num_frames: Number of frames to generate
            height: Video height (default: self.unet.config.sample_size * 8)
            width: Video width (default: self.unet.config.sample_size * 8)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            fps: Frames per second for output video
            ... (other standard SDXL arguments)

        Returns:
            Video frames as numpy array or PIL images
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompts
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode main prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        # 4. Handle content/style prompts if provided
        if prompt_content is not None or prompt_style is not None:
            # Encode content prompt
            if prompt_content is not None:
                (
                    content_embeds,
                    _,
                    content_pooled_embeds,
                    _,
                ) = self.encode_prompt(
                    prompt=prompt_content,
                    prompt_2=prompt_content,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=False,
                )

            # Encode style prompt
            if prompt_style is not None:
                (
                    style_embeds,
                    _,
                    style_pooled_embeds,
                    _,
                ) = self.encode_prompt(
                    prompt=prompt_style,
                    prompt_2=prompt_style,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=False,
                )

            # Merge embeddings
            if prompt_content is not None and prompt_style is not None:
                # Average content and style
                prompt_embeds = (content_embeds + style_embeds) / 2.0
                pooled_prompt_embeds = (content_pooled_embeds + style_pooled_embeds) / 2.0
            elif prompt_content is not None:
                prompt_embeds = content_embeds
                pooled_prompt_embeds = content_pooled_embeds
            elif prompt_style is not None:
                prompt_embeds = style_embeds
                pooled_prompt_embeds = style_pooled_embeds

            print(f"Using content/style merged embeddings")

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents_video(
            batch_size * num_images_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            # Repeat for each frame
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        # Repeat embeddings for each frame
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 9. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents if doing classifier-free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict noise residual
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids
                }

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    num_frames=num_frames,  # VIDEO-SPECIFIC
                    return_dict=False,
                )[0]

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # Compute previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # Call callback
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
        else:
            # Decode video frame by frame
            image = self.decode_latents_video(latents, num_frames)

            # Convert to desired output format
            if output_type == "pil":
                # Post-processing
                image = self.image_processor.postprocess(image, output_type=output_type)
            else:
                # numpy
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg