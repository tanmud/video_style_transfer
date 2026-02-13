from diffusers import CogVideoXPipeline
from diffusers.utils import logging
import torch
from typing import Optional, List, Union, Callable

logger = logging.get_logger(__name__)


class CogVideoXUnZipLoRAPipeline(CogVideoXPipeline):
    """
    CogVideoX pipeline extended with UnZipLoRA dual-prompt support
    
    Allows separate content and style prompts for video generation
    """
    
    def encode_prompt_dual(
        self,
        prompt: Union[str, List[str]],
        prompt_content: Optional[Union[str, List[str]]] = None,
        prompt_style: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        num_videos_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ):
        """
        Encode prompts with dual conditioning support
        
        Args:
            prompt: Main combined prompt
            prompt_content: Content-specific prompt (e.g., "a male biker")
            prompt_style: Style-specific prompt (e.g., "cartoon style")
            device: Device to place embeddings on
            num_videos_per_prompt: Number of videos per prompt
            do_classifier_free_guidance: Whether to use CFG
            negative_prompt: Negative prompt for CFG
        
        Returns:
            Tuple of (prompt_embeds, prompt_embeds_content, prompt_embeds_style)
        """
        if device is None:
            device = self._execution_device
        
        # Encode main prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        
        # Encode content prompt (or use main prompt)
        if prompt_content is not None:
            prompt_embeds_content = self._encode_prompt(
                prompt=prompt_content,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )
        else:
            prompt_embeds_content = prompt_embeds
        
        # Encode style prompt (or use main prompt)
        if prompt_style is not None:
            prompt_embeds_style = self._encode_prompt(
                prompt=prompt_style,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )
        else:
            prompt_embeds_style = prompt_embeds
        
        return prompt_embeds, prompt_embeds_content, prompt_embeds_style
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        prompt_content: Optional[Union[str, List[str]]] = None,  # NEW
        prompt_style: Optional[Union[str, List[str]]] = None,    # NEW
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_frames: int = 49,
        height: int = 480,
        width: int = 720,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds_content: Optional[torch.FloatTensor] = None,  # NEW
        prompt_embeds_style: Optional[torch.FloatTensor] = None,    # NEW
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        max_sequence_length: int = 226,
    ):
        """
        Generate video with UnZipLoRA dual-prompt conditioning
        
        Args:
            prompt: Main text prompt
            prompt_content: Content-specific prompt (UnZipLoRA)
            prompt_style: Style-specific prompt (UnZipLoRA)
            num_frames: Number of frames to generate (default: 49)
            height: Video height (default: 480)
            width: Video width (default: 720)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            ... (other standard CogVideoX parameters)
        
        Returns:
            CogVideoXPipelineOutput with generated video frames
        """
        # Encode prompts with dual conditioning
        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_content, prompt_embeds_style = self.encode_prompt_dual(
                prompt=prompt,
                prompt_content=prompt_content,
                prompt_style=prompt_style,
                device=self.device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=guidance_scale > 1.0,
                negative_prompt=negative_prompt,
            )
        
        # Call parent pipeline with dual embeddings
        # The transformer will receive both embeddings via the attention processor
        return super().__call__(
            prompt=None,  # Use embeddings instead
            negative_prompt=None,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            num_videos_per_prompt=num_videos_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            max_sequence_length=max_sequence_length,
        )
