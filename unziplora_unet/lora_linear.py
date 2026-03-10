# Copied verbatim from diffusers 0.27.x diffusers/models/lora.py
# Preserved here because diffusers 0.30+ removed this module.

from typing import Optional, Union
import torch
import torch.nn.functional as F
from torch import nn


class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank
        return up_hidden_states.to(orig_dtype)


class LoRACompatibleLinear(nn.Linear):
    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return
        dtype, device = self.weight.data.dtype, self.weight.data.device
        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()
        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank
        fused_weight = w_orig + lora_scale * (w_up @ w_down)
        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError("NaN detected in fused LoRA weight.")
        self.weight.data = fused_weight.to(device=device, dtype=dtype)
        self.lora_layer = None

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return
        fused_weight = self.weight.data
        dtype, device = fused_weight.dtype, fused_weight.device
        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device=device).float()
        self.weight.data = (fused_weight.float() - self.lora_scale * (w_up @ w_down)).to(dtype=dtype)
        self.w_up = self.w_down = None

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        dtype = self.weight.dtype
        if self.lora_layer is None:
            out = super().forward(hidden_states)
            return out
        else:
            out = super().forward(hidden_states)
            return out + (scale * self.lora_layer(hidden_states))


class LoRACompatibleConv(nn.Conv2d):
    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return
        dtype, device = self.weight.data.dtype, self.weight.data.device
        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()
        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank
        fused_weight = w_orig + lora_scale * (w_up @ w_down).reshape(w_orig.shape)
        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError("NaN detected in fused LoRA conv weight.")
        self.weight.data = fused_weight.to(device=device, dtype=dtype)
        self.lora_layer = None

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return
        fused_weight = self.weight.data
        dtype, device = fused_weight.dtype, fused_weight.device
        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device=device).float()
        self.weight.data = (fused_weight.float() - self.lora_scale * (w_up @ w_down).reshape(fused_weight.shape)).to(dtype=dtype)
        self.w_up = self.w_down = None

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.lora_layer is None:
            return super().forward(hidden_states)
        else:
            return super().forward(hidden_states) + (
                scale * self.lora_layer(hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(-2, -1))
                .transpose(-2, -1).reshape(hidden_states.shape[0], -1, *hidden_states.shape[2:])
            )
