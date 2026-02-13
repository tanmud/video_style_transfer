from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.models.transformers.transformer_temporal import TransformerTemporalModel
from diffusers.utils import logging
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

logger = logging.get_logger(__name__)


class UNetAnimateDiffConditionModel(UNet2DConditionModel):
    """
    Extended UNet2DConditionModel with temporal transformers for video generation.

    Architecture:
        Down blocks: Spatial (UnzipLoRA) → Temporal (Motion)
        Mid block:   Spatial (UnzipLoRA) → Temporal (Motion)
        Up blocks:   Spatial (UnzipLoRA) → Temporal (Motion)

    The temporal transformers model motion across frames while spatial attention
    handles content/style separation per frame.
    """

    def __init__(self, *args, **kwargs):
        # Extract motion module settings before calling super
        motion_module_kwargs = kwargs.pop("motion_module_kwargs", {})

        # Initialize base UNet2DConditionModel
        super().__init__(*args, **kwargs)

        # Get block dimensions from config
        block_out_channels = self.config.block_out_channels

        # Create temporal transformers for down blocks
        self.down_temporal_transformers = nn.ModuleList()
        for i, down_block in enumerate(self.down_blocks):
            # Only add temporal if block has cross-attention
            if hasattr(down_block, 'has_cross_attention') and down_block.has_cross_attention:
                temporal = self._make_temporal_transformer(
                    block_out_channels[i],
                    **motion_module_kwargs
                )
                self.down_temporal_transformers.append(temporal)
            else:
                self.down_temporal_transformers.append(None)

        # Create temporal transformer for mid block
        self.mid_temporal_transformer = self._make_temporal_transformer(
            block_out_channels[-1],
            **motion_module_kwargs
        )

        # Create temporal transformers for up blocks
        self.up_temporal_transformers = nn.ModuleList()
        reversed_channels = list(reversed(block_out_channels))
        for i, up_block in enumerate(self.up_blocks):
            # Only add temporal if block has cross-attention
            if hasattr(up_block, 'has_cross_attention') and up_block.has_cross_attention:
                temporal = self._make_temporal_transformer(
                    reversed_channels[i],
                    **motion_module_kwargs
                )
                self.up_temporal_transformers.append(temporal)
            else:
                self.up_temporal_transformers.append(None)

    def _make_temporal_transformer(
        self,
        in_channels: int,
        num_attention_heads: int = 8,
        attention_head_dim: int = None,
        num_layers: int = 1,
        **kwargs
    ) -> TransformerTemporalModel:
        """Create a temporal transformer (motion module)."""
        if attention_head_dim is None:
            attention_head_dim = in_channels // num_attention_heads

        return TransformerTemporalModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            num_layers=num_layers,
            norm_num_groups=32,
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_content: Optional[torch.Tensor] = None,
        encoder_hidden_states_style: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        num_frames: int = 1,
    ):
        """
        Forward pass with spatial-temporal interleaving.

        Args:
            sample: Input latents (B*F, C, H, W) - already flattened
            timestep: Diffusion timestep
            encoder_hidden_states: Text embeddings (B, seq, dim) - NOT (B*F)!
            num_frames: Number of frames F
            encoder_hidden_states_content: Content embeddings (B, seq, dim)
            encoder_hidden_states_style: Style embeddings (B, seq, dim)
            added_cond_kwargs: SDXL conditions (text_embeds, time_ids)
            ... other standard UNet args

        Returns:
            UNet2DConditionOutput with sample (B*F, C, H, W)
        """

        # Compute true batch size (B, not B*F)
        batch_size = sample.shape[0] // num_frames if num_frames > 1 else sample.shape[0]

        # Prepare cross attention kwargs with num_frames
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        cross_attention_kwargs["num_frames"] = num_frames

        # 0. Center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. Time embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # Broadcast to batch dimension (B*F)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # Handle SDXL added conditions
        aug_emb = None
        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text_time":
                if added_cond_kwargs is None:
                    raise ValueError("added_cond_kwargs must be provided for SDXL")

                text_embeds = added_cond_kwargs.get("text_embeds")
                time_ids = added_cond_kwargs.get("time_ids")

                # Expand for video batch if needed
                if text_embeds.shape[0] == batch_size and num_frames > 1:
                    text_embeds = text_embeds.unsqueeze(1).expand(-1, num_frames, -1)
                    text_embeds = text_embeds.reshape(-1, text_embeds.shape[-1])
                if time_ids.shape[0] == batch_size and num_frames > 1:
                    time_ids = time_ids.unsqueeze(1).expand(-1, num_frames, -1)
                    time_ids = time_ids.reshape(-1, time_ids.shape[-1])

                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                add_embeds = torch.cat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

            emb = emb + aug_emb if aug_emb is not None else emb

        # 2. Pre-process
        sample = self.conv_in(sample)

        # 3. Down blocks with temporal transformers
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # Spatial processing (frame-independent UnzipLoRA attention)
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_content=encoder_hidden_states_content,
                    encoder_hidden_states_style=encoder_hidden_states_style,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    num_frames=num_frames,
                )

                # Temporal processing (motion module for cross-frame attention)
                if self.down_temporal_transformers[i] is not None and num_frames > 1:
                    sample = self._apply_temporal_transformer(
                        sample,
                        self.down_temporal_transformers[i],
                        batch_size,
                        num_frames
                    )
            else:
                # No attention - just resnets
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. Mid block with temporal transformer
        if self.mid_block is not None:
            # Spatial processing
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_content=encoder_hidden_states_content,
                encoder_hidden_states_style=encoder_hidden_states_style,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=num_frames,
            )

            # Temporal processing (strongest motion modeling here)
            if self.mid_temporal_transformer is not None and num_frames > 1:
                sample = self._apply_temporal_transformer(
                    sample,
                    self.mid_temporal_transformer,
                    batch_size,
                    num_frames
                )

        # 5. Up blocks with temporal transformers
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                # Spatial processing
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_content=encoder_hidden_states_content,
                    encoder_hidden_states_style=encoder_hidden_states_style,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    num_frames=num_frames,
                )

                # Temporal processing
                if self.up_temporal_transformers[i] is not None and num_frames > 1:
                    sample = self._apply_temporal_transformer(
                        sample,
                        self.up_temporal_transformers[i],
                        batch_size,
                        num_frames
                    )
            else:
                # No attention - just resnets
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples
                )

        # 6. Post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

    def _apply_temporal_transformer(
        self,
        hidden_states: torch.Tensor,
        temporal_transformer: TransformerTemporalModel,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Apply temporal transformer to model motion across frames.

        This is the KEY function that enables motion modeling.

        Args:
            hidden_states: (B*F, C, H, W) - flattened spatial features
            temporal_transformer: Motion module (TransformerTemporalModel)
            batch_size: B (true batch size, NOT B*F)
            num_frames: F (number of frames)

        Returns:
            hidden_states: (B*F, C, H, W) with temporal consistency

        Shape transformations:
            Input:  (B*F, C, H, W) - spatial features from all frames flattened
            Step 1: (B, F, C, H, W) - group frames by video
            Step 2: (B, C, F, H, W) - move frames to temporal dimension
            Step 3: Apply temporal attention across F dimension
            Step 4: (B, F, C, H, W) - move frames back
            Output: (B*F, C, H, W) - flatten for next spatial layer
        """
        bf, c, h, w = hidden_states.shape

        # Safety check
        if bf != batch_size * num_frames:
            raise ValueError(
                f"Expected hidden_states batch size {batch_size * num_frames}, "
                f"got {bf} (batch_size={batch_size}, num_frames={num_frames})"
            )

        # Step 1: Reshape from flattened to grouped
        # (B*F, C, H, W) → (B, F, C, H, W)
        hidden_states = hidden_states.view(batch_size, num_frames, c, h, w)

        # Step 2: Move frames to temporal dimension
        # (B, F, C, H, W) → (B, C, F, H, W)
        # Now F is in the "temporal" position where attention will be applied
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()

        # Step 3: Apply temporal transformer
        # TransformerTemporalModel expects (B, C, F, H, W)
        # It will attend across the F dimension, modeling motion
        hidden_states = temporal_transformer(hidden_states, num_frames=num_frames)

        # Step 4: Move frames back from temporal dimension
        # (B, C, F, H, W) → (B, F, C, H, W)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()

        # Step 5: Flatten back to spatial processing format
        # (B, F, C, H, W) → (B*F, C, H, W)
        hidden_states = hidden_states.view(bf, c, h, w)

        return hidden_states

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load a pretrained UNet and add temporal transformers.

        Example:
            unet = UNetAnimateDiffConditionModel.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="unet",
                motion_module_kwargs={"num_layers": 2}
            )
        """
        # Extract motion module kwargs before loading
        motion_module_kwargs = kwargs.pop("motion_module_kwargs", {})

        # Load base UNet2DConditionModel
        model = super(UNet2DConditionModel, cls).from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        # Now add temporal transformers by re-initializing with motion modules
        # This preserves all the loaded spatial weights
        config = model.config
        state_dict = model.state_dict()

        # Create new instance with motion modules
        new_model = cls(**config, motion_module_kwargs=motion_module_kwargs)

        # Load spatial weights (temporal transformers stay randomly initialized)
        new_model.load_state_dict(state_dict, strict=False)

        return new_model

    def save_motion_modules(self, output_dir: str):
        """
        Save only the motion module weights (temporal transformers).

        This creates a lightweight checkpoint (~100MB) containing only
        the learned motion patterns, which can be loaded onto any SDXL UNet.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        motion_state = {
            'down_temporal': self.down_temporal_transformers.state_dict(),
            'mid_temporal': self.mid_temporal_transformer.state_dict(),
            'up_temporal': self.up_temporal_transformers.state_dict(),
        }

        save_path = os.path.join(output_dir, "motion_modules.pth")
        torch.save(motion_state, save_path)
        logger.info(f"Motion modules saved to {save_path}")

    def load_motion_modules(self, motion_module_path: str):
        """
        Load motion module weights from a file.

        This allows you to:
        1. Train motion on one SDXL checkpoint
        2. Use those motion modules with different SDXL checkpoints
        3. Share motion modules without sharing full model weights
        """
        motion_state = torch.load(motion_module_path, map_location="cpu")

        self.down_temporal_transformers.load_state_dict(motion_state['down_temporal'])
        self.mid_temporal_transformer.load_state_dict(motion_state['mid_temporal'])
        self.up_temporal_transformers.load_state_dict(motion_state['up_temporal'])

        logger.info(f"Motion modules loaded from {motion_module_path}")