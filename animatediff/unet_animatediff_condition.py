import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D

# Use the correct temporal transformer
from animatediff.temporal_transformer import TemporalTransformer


class UNetAnimateDiffConditionModel(UNet2DConditionModel):
    """
    UNet2DConditionModel extended with temporal transformers for video generation.

    Architecture:
    - Spatial layers: Standard SDXL UNet (pretrained, frozen during motion-only training)
    - Temporal layers: Added after spatial blocks (learnable motion modules)

    Shape flow:
    - Input: (B*F, C, H, W) where B=batch, F=frames
    - After spatial layers: (B*F, C, H, W)
    - Reshape to (B, C, F, H, W) for temporal layers
    - Temporal attention: Each spatial position attends across F frames
    - Reshape back to (B*F, C, H, W) for next spatial layer

    The temporal transformer uses:
    - Positional encoding to track frame order
    - Self-attention across frames at each spatial location
    - Residual connections to preserve spatial features
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load pretrained SDXL UNet and add motion modules.

        Args:
            pretrained_model_name_or_path: Path to SDXL checkpoint
            motion_module_kwargs: Dict with motion module config (e.g., {"num_layers": 2})
        """
        # Extract motion module config
        motion_module_kwargs = kwargs.pop("motion_module_kwargs", {})

        # Load base SDXL UNet
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Add temporal transformers
        model._add_temporal_transformers(motion_module_kwargs)

        return model

    def _add_temporal_transformers(self, motion_module_kwargs: Dict[str, Any]):
        """Add temporal transformer modules after each spatial block."""
        num_layers = motion_module_kwargs.get("num_layers", 2)
        num_heads = motion_module_kwargs.get("num_heads", 8)

        # Store temporal transformers
        self.down_temporal = nn.ModuleList()
        self.mid_temporal = None
        self.up_temporal = nn.ModuleList()

        print("Adding temporal transformers to UNet...")

        # Add temporal transformers to down blocks
        for i, block in enumerate(self.down_blocks):
            if hasattr(block, "attentions") and block.attentions is not None:
                # Get channels from the block
                if hasattr(block.resnets[0], 'out_channels'):
                    channels = block.resnets[0].out_channels
                else:
                    channels = block.resnets[0].conv1.out_channels

                temporal = TemporalTransformer(
                    in_channels=channels,
                    num_layers=num_layers,
                    num_heads=num_heads,
                )
                self.down_temporal.append(temporal)
                print(f"  Down block {i}: Added temporal transformer ({channels} channels)")
            else:
                self.down_temporal.append(None)

        # Add temporal transformer to mid block
        if hasattr(self.mid_block, "attentions") and self.mid_block.attentions is not None:
            if hasattr(self.mid_block.resnets[0], 'out_channels'):
                channels = self.mid_block.resnets[0].out_channels
            else:
                channels = self.mid_block.resnets[0].conv1.out_channels

            self.mid_temporal = TemporalTransformer(
                in_channels=channels,
                num_layers=num_layers,
                num_heads=num_heads,
            )
            print(f"  Mid block: Added temporal transformer ({channels} channels)")

        # Add temporal transformers to up blocks
        for i, block in enumerate(self.up_blocks):
            if hasattr(block, "attentions") and block.attentions is not None:
                if hasattr(block.resnets[0], 'out_channels'):
                    channels = block.resnets[0].out_channels
                else:
                    channels = block.resnets[0].conv1.out_channels

                temporal = TemporalTransformer(
                    in_channels=channels,
                    num_layers=num_layers,
                    num_heads=num_heads,
                )
                self.up_temporal.append(temporal)
                print(f"  Up block {i}: Added temporal transformer ({channels} channels)")
            else:
                self.up_temporal.append(None)

        print("Temporal transformers added successfully")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        num_frames: int = 1,
        **kwargs
    ):
        """
        Forward pass with temporal attention.

        Args:
            sample: (B*F, C, H, W) noisy latents
            timestep: (B,) timesteps
            encoder_hidden_states: (B, seq_len, dim) text embeddings
            num_frames: number of frames F
        """
        batch_frames = sample.shape[0]
        batch_size = batch_frames // num_frames

        # Expand text embeddings for all frames
        if encoder_hidden_states.shape[0] == batch_size:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # Call parent forward with temporal injection
        return self._forward_with_temporal(
            sample, timestep, encoder_hidden_states, num_frames, **kwargs
        )

    def _forward_with_temporal(self, sample, timestep, encoder_hidden_states, num_frames, **kwargs):
        """Modified forward pass that inserts temporal transformers."""
        batch_frames = sample.shape[0]
        batch_size = batch_frames // num_frames

        # Time embedding
        t_emb = self.time_proj(timestep)
        emb = self.time_embedding(t_emb)

        # Added cond kwargs for SDXL
        added_cond_kwargs = kwargs.get("added_cond_kwargs", {})
        if "text_embeds" in added_cond_kwargs and added_cond_kwargs["text_embeds"].shape[0] == batch_size:
            added_cond_kwargs["text_embeds"] = added_cond_kwargs["text_embeds"].repeat_interleave(num_frames, dim=0)
        if "time_ids" in added_cond_kwargs and added_cond_kwargs["time_ids"].shape[0] == batch_size:
            added_cond_kwargs["time_ids"] = added_cond_kwargs["time_ids"].repeat_interleave(num_frames, dim=0)

        # Process time embeds for SDXL
        aug_emb = None
        if self.config.addition_embed_type == "text_time":
            if added_cond_kwargs is None:
                raise ValueError("added_cond_kwargs must be provided for text_time embedding")
            text_embeds = added_cond_kwargs.get("text_embeds")
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((batch_frames, -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            aug_emb = self.add_embedding(add_embeds)

        print(emb.shape)
        print(aug_emb.shape if aug_emb is not None else "No aug_emb")
        emb = emb.repeat_interleave(num_frames, dim=0)  # (B*F, dim)
        emb = emb + aug_emb if aug_emb is not None else emb

        # Initial conv
        sample = self.conv_in(sample)

        # Down blocks with temporal
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            # Apply temporal transformer
            if self.down_temporal[i] is not None:
                sample = self._apply_temporal(sample, self.down_temporal[i], batch_size, num_frames)

            down_block_res_samples += res_samples

        # Mid block with temporal
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
        if self.mid_temporal is not None:
            sample = self._apply_temporal(sample, self.mid_temporal, batch_size, num_frames)

        # Up blocks with temporal
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                )

            # Apply temporal transformer
            if self.up_temporal[i] is not None:
                sample = self._apply_temporal(sample, self.up_temporal[i], batch_size, num_frames)

        # Output
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        from diffusers.models.unet_2d_condition import UNet2DConditionOutput
        return UNet2DConditionOutput(sample=sample)

    def _apply_temporal(self, hidden_states, temporal_module, batch_size, num_frames):
        """
        Apply temporal transformer to hidden states.

        Converts: (B*F, C, H, W) -> (B, C, F, H, W) -> temporal attention -> (B*F, C, H, W)

        This is where the magic happens:
        1. Reshape to expose frame dimension
        2. Apply temporal attention (frames attend to each other)
        3. Reshape back for spatial processing
        """
        bf, c, h, w = hidden_states.shape

        # Safety check
        if bf != batch_size * num_frames:
            raise ValueError(f"Shape mismatch: expected {batch_size * num_frames}, got {bf}")

        # Reshape: (B*F, C, H, W) -> (B, F, C, H, W) -> (B, C, F, H, W)
        hidden_states = hidden_states.view(batch_size, num_frames, c, h, w)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, F, H, W)

        # Apply temporal attention across F dimension
        hidden_states = temporal_module(hidden_states, num_frames=num_frames)

        # Reshape back: (B, C, F, H, W) -> (B, F, C, H, W) -> (B*F, C, H, W)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()  # (B, F, C, H, W)
        hidden_states = hidden_states.view(bf, c, h, w)

        return hidden_states

    def save_motion_modules(self, save_directory):
        """Save only the temporal transformer weights."""
        import os
        os.makedirs(save_directory, exist_ok=True)

        state_dict = {
            "down_temporal": {k: v.cpu() for k, v in self.down_temporal.state_dict().items()},
            "mid_temporal": {k: v.cpu() for k, v in self.mid_temporal.state_dict().items()} if self.mid_temporal else {},
            "up_temporal": {k: v.cpu() for k, v in self.up_temporal.state_dict().items()},
        }

        save_path = os.path.join(save_directory, "motion_modules.pth")
        torch.save(state_dict, save_path)
        print(f"Motion modules saved to {save_path}")

    def load_motion_modules(self, load_path):
        """Load temporal transformer weights."""
        state_dict = torch.load(load_path, map_location="cpu")

        self.down_temporal.load_state_dict(state_dict["down_temporal"])
        if self.mid_temporal and state_dict.get("mid_temporal"):
            self.mid_temporal.load_state_dict(state_dict["mid_temporal"])
        self.up_temporal.load_state_dict(state_dict["up_temporal"])

        print(f"Motion modules loaded from {load_path}")