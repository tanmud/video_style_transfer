from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention as DiffusersAttention


class Attention(DiffusersAttention):
    """
    Extended Attention module that supports UnzipLoRA content/style separation
    and video batch processing (expanding encoder states for num_frames).
    """

    def set_processor(self, processor):
        """Set the attention processor."""
        self.processor = processor


class AttnProcessor:
    """
    Default attention processor with video batch support.
    Expands encoder_hidden_states from (B, seq, dim) to (B*F, seq, dim) when needed.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_states_content: Optional[torch.FloatTensor] = None,
        encoder_hidden_states_style: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        num_frames: Optional[int] = None,  # NEW: for video
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # Expand encoder states for video if needed
        if encoder_hidden_states is not None and num_frames is not None and num_frames > 1:
            true_batch_size = batch_size // num_frames
            if encoder_hidden_states.shape[0] == true_batch_size:
                # (B, seq, dim) -> (B, F, seq, dim) -> (B*F, seq, dim)
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(
                    -1, num_frames, -1, -1
                ).reshape(batch_size, -1, encoder_hidden_states.shape[-1])

        if encoder_hidden_states_content is not None and num_frames is not None and num_frames > 1:
            true_batch_size = batch_size // num_frames
            if encoder_hidden_states_content.shape[0] == true_batch_size:
                encoder_hidden_states_content = encoder_hidden_states_content.unsqueeze(1).expand(
                    -1, num_frames, -1, -1
                ).reshape(batch_size, -1, encoder_hidden_states_content.shape[-1])

        if encoder_hidden_states_style is not None and num_frames is not None and num_frames > 1:
            true_batch_size = batch_size // num_frames
            if encoder_hidden_states_style.shape[0] == true_batch_size:
                encoder_hidden_states_style = encoder_hidden_states_style.unsqueeze(1).expand(
                    -1, num_frames, -1, -1
                ).reshape(batch_size, -1, encoder_hidden_states_style.shape[-1])

        # Query projection
        query = attn.to_q(hidden_states, scale=scale)

        # Key/Value projection (use content/style if provided)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        # Reshape for multi-head attention
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Attention computation
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class AttnProcessor2_0:
    """
    Scaled dot-product attention processor (PyTorch 2.0+) with video batch support.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_states_content: Optional[torch.FloatTensor] = None,
        encoder_hidden_states_style: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        num_frames: Optional[int] = None,  # NEW: for video
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape

        # Expand encoder states for video if needed
        if encoder_hidden_states is not None and num_frames is not None and num_frames > 1:
            true_batch_size = batch_size // num_frames
            if encoder_hidden_states.shape[0] == true_batch_size:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(
                    -1, num_frames, -1, -1
                ).reshape(batch_size, -1, encoder_hidden_states.shape[-1])

        if encoder_hidden_states_content is not None and num_frames is not None and num_frames > 1:
            true_batch_size = batch_size // num_frames
            if encoder_hidden_states_content.shape[0] == true_batch_size:
                encoder_hidden_states_content = encoder_hidden_states_content.unsqueeze(1).expand(
                    -1, num_frames, -1, -1
                ).reshape(batch_size, -1, encoder_hidden_states_content.shape[-1])

        if encoder_hidden_states_style is not None and num_frames is not None and num_frames > 1:
            true_batch_size = batch_size // num_frames
            if encoder_hidden_states_style.shape[0] == true_batch_size:
                encoder_hidden_states_style = encoder_hidden_states_style.unsqueeze(1).expand(
                    -1, num_frames, -1, -1
                ).reshape(batch_size, -1, encoder_hidden_states_style.shape[-1])

        # Query projection
        query = attn.to_q(hidden_states, scale=scale)

        # Key/Value projection
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        # Reshape for multi-head
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention (PyTorch 2.0)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states