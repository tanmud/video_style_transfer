import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    Allows the model to know which frame is which.
    """
    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (B, F, C) tensor
        Returns:
            (B, F, C) tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class TemporalTransformerBlock(nn.Module):
    """
    Single temporal transformer block with self-attention across frames.

    This is the correct way to do temporal attention:
    1. LayerNorm
    2. Self-attention across F (frame) dimension  
    3. Residual connection
    4. LayerNorm
    5. Feed-forward network
    6. Residual connection
    """
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # Input: (B, F, C)
        )

        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (B, F, C) tensor where F is number of frames
        Returns:
            (B, F, C) tensor
        """
        # Self-attention across frames
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Feed-forward
        x = x + self.ffn(self.norm2(x))

        return x


class TemporalTransformer(nn.Module):
    """
    Complete temporal transformer module for AnimateDiff.

    Processes (B, C, F, H, W) by:
    1. Treating each spatial position (h, w) as a separate sequence
    2. Applying self-attention across F frames
    3. Learning temporal dependencies at each spatial location
    """

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers

        # Positional encoding for frame positions
        self.pos_encoding = PositionalEncoding(in_channels, max_len=32)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(in_channels, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Optional: projection layers
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, hidden_states: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
        """
        Apply temporal attention across frames.

        Args:
            hidden_states: (B, C, F, H, W) tensor
            num_frames: number of frames F

        Returns:
            (B, C, F, H, W) tensor with temporal attention applied
        """
        batch_size, channels, frames, height, width = hidden_states.shape

        # Reshape: (B, C, F, H, W) -> (B*H*W, F, C)
        # This treats each spatial position as a separate batch element
        # and creates F-length sequences for each position
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1)  # (B, H, W, F, C)
        hidden_states = hidden_states.reshape(-1, frames, channels)  # (B*H*W, F, C)

        # Add positional encoding so model knows frame order
        hidden_states = self.pos_encoding(hidden_states)

        # Apply transformer blocks (self-attention across F)
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Reshape back: (B*H*W, F, C) -> (B, C, F, H, W)
        hidden_states = hidden_states.view(batch_size, height, width, frames, channels)
        hidden_states = hidden_states.permute(0, 4, 3, 1, 2).contiguous()  # (B, C, F, H, W)

        return hidden_states


# Quick test
if __name__ == "__main__":
    print("Testing Temporal Transformer...")

    # Create module
    temporal = TemporalTransformer(
        in_channels=320,
        num_layers=2,
        num_heads=8,
    )

    # Test input: (B=2, C=320, F=16, H=32, W=32)
    x = torch.randn(2, 320, 16, 32, 32)

    print(f"Input shape: {x.shape}")

    # Forward pass
    out = temporal(x, num_frames=16)

    print(f"Output shape: {out.shape}")
    print(f"Shape preserved: {out.shape == x.shape}")

    # Check gradients
    loss = out.sum()
    loss.backward()

    print(f"Gradients computed successfully")
    print(f"Parameters: {sum(p.numel() for p in temporal.parameters()):,}")