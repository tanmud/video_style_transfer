from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from unziplora_unet.pipeline_stable_diffusion_xl import StableDiffusionXLUnZipLoRAPipeline


class AnimateDiffUnZipLoRAPipeline(StableDiffusionXLUnZipLoRAPipeline):
    """
    Pipeline for text-to-video generation using AnimateDiff with UnZipLoRA.

    Extends StableDiffusionXLUnZipLoRAPipeline to add temporal (video) generation
    while maintaining all content/style separation functionality.

    Inherits all methods from parent:
    - encode_prompt() for text encoding
    - _get_add_time_ids() for SDXL conditions
    - All the content/style separation logic

    Adds:
    - Video generation (temporal dimension handling)
    - Frame batching and decoding
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_content: Optional[Union[str, List[str]]] = None,
        prompt_content_2: Optional[Union[str, List[str]]] = None,
        prompt_style: Optional[Union[str, List[str]]] = None,
        prompt_style_2: Optional[Union[str, List[str]]] = None,
        # Video-specific parameters
        num_frames: int = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_content: Optional[Union[str, List[str]]] = None,
        negative_prompt_content_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_style: Optional[Union[str, List[str]]] = None,
        negative_prompt_style_2: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds_content: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds_content: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds_content: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds_content: Optional[torch.FloatTensor] = None,
        prompt_embeds_style: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds_style: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds_style: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds_style: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for video generation.

        Adds num_frames parameter for temporal dimension while maintaining
        all UnZipLoRA content/style separation functionality.

        Args:
            num_frames (`int`, defaults to 16):
                Number of video frames to generate.

            All other parameters same as StableDiffusionXLUnZipLoRAPipeline.

        Returns:
            List of video frames or latents with temporal dimension.
        """

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs (reuse parent's check_inputs)
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            None,  # callback_steps
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            None,  # callback_on_step_end_tensor_inputs
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt (REUSE parent's encode_prompt!)
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

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
            num_images_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # Encode content prompt
        if prompt_content is not None:
            (
                prompt_embeds_content,
                negative_prompt_embeds_content,
                pooled_prompt_embeds_content,
                negative_pooled_prompt_embeds_content,
            ) = self.encode_prompt(
                prompt=prompt_content,
                prompt_2=prompt_content_2,
                device=device,
                num_images_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt_content,
                negative_prompt_2=negative_prompt_content_2,
                prompt_embeds=prompt_embeds_content,
                negative_prompt_embeds=negative_prompt_embeds_content,
                pooled_prompt_embeds=pooled_prompt_embeds_content,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_content,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )

        # Encode style prompt
        if prompt_style is not None:
            (
                prompt_embeds_style,
                negative_prompt_embeds_style,
                pooled_prompt_embeds_style,
                negative_pooled_prompt_embeds_style,
            ) = self.encode_prompt(
                prompt=prompt_style,
                prompt_2=prompt_style_2,
                device=device,
                num_images_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt_style,
                negative_prompt_2=negative_prompt_style_2,
                prompt_embeds=prompt_embeds_style,
                negative_prompt_embeds=negative_prompt_embeds_style,
                pooled_prompt_embeds=pooled_prompt_embeds_style,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_style,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables (ADD TEMPORAL DIMENSION)
        num_channels_latents = self.unet.config.in_channels

        # Shape: (batch, num_frames, channels, height, width)
        latents = self.prepare_video_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings (REUSE parent's method!)
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

            if prompt_content is not None:
                prompt_embeds_content = torch.cat([negative_prompt_embeds_content, prompt_embeds_content], dim=0)
            if prompt_style is not None:
                prompt_embeds_style = torch.cat([negative_prompt_embeds_style, prompt_embeds_style], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        if prompt_content is not None:
            prompt_embeds_content = prompt_embeds_content.to(device)
        if prompt_style is not None:
            prompt_embeds_style = prompt_embeds_style.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_videos_per_prompt, 1)

        # 8. Denoising loop (ADAPTED FOR VIDEO)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_content=prompt_embeds_content if prompt_content is not None else None,
                    encoder_hidden_states_style=prompt_embeds_style if prompt_style is not None else None,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # Perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 9. Decode video frames
        if output_type == "latent":
            video = latents
        else:
            video = self.decode_video_latents(latents)

            if output_type == "pil":
                video = self.video_processor.postprocess_video(video, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return StableDiffusionXLPipelineOutput(images=video)

    def prepare_video_latents(
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
        Prepare latents with temporal dimension for video generation.
        Shape: (batch, num_frames, channels, height, width)
        """
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # Scale by scheduler's initial noise sigma
        latents = latents * self.scheduler.init_noise_sigma

        return latents

    def decode_video_latents(self, latents):
        """
        Decode video latents to frames.

        Args:
            latents: (batch, num_frames, channels, height, width)

        Returns:
            frames: List of decoded frames
        """
        batch_size, num_frames = latents.shape[0], latents.shape[1]

        # Decode frame by frame
        frames = []
        for i in range(num_frames):
            latent_frame = latents[:, i] / self.vae.config.scaling_factor
            image = self.vae.decode(latent_frame, return_dict=False)[0]
            frames.append(image)

        # Stack frames: (batch, num_frames, channels, height, width)
        video = torch.stack(frames, dim=1)

        return video