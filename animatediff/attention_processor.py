import torch
import torch.nn.functional as F
from typing import Optional


class AnimateDiffAttnProcessor2_0:
    """
    AttnProcessor2_0 adapted for UNetMotionModel (AnimateDiff).
    Handles the mismatch between spatial hidden_states (B*F, seq, dim)
    and text encoder_hidden_states (B, seq, dim).
    Calls LoRACompatibleLinear layers without UnZipLoRA-specific kwargs.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AnimateDiffAttnProcessor2_0 requires PyTorch 2.0+.")

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        **kwargs,  # absorb unused kwargs (e.g. encoder_hidden_states_content/style)
    ) -> torch.FloatTensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # Always derive batch_size from hidden_states (B*F after motion model reshape)
        batch_size = hidden_states.shape[0]
        sequence_length = (
            hidden_states.shape[1]
            if encoder_hidden_states is None
            else encoder_hidden_states.shape[1]
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # LoRACompatibleLinear only takes (hidden_states, scale) — no extra kwargs
        args = () if not hasattr(attn.to_q, 'lora_layer') else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # Repeat encoder_hidden_states to match B*F if not already done
        if encoder_hidden_states.shape[0] != batch_size:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                batch_size // encoder_hidden_states.shape[0], dim=0
            )

        key   = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
