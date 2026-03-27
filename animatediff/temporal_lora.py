"""
animatediff/temporal_lora.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TemporalLoRALinear(nn.Module):
    """
    Low-rank adapter wrapping a frozen nn.Linear.
    Effective weight: W_base + scale * (lora_B @ lora_A)   scale = alpha / rank
    Init: lora_A ~ N(0,0.01), lora_B = 0  →  delta is zero at step 0.
    """
    def __init__(self, base: nn.Linear, rank: int = 32, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.in_features  = base.in_features
        self.out_features = base.out_features
        self.rank  = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base.weight, self.base.bias)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out

    def get_delta(self) -> torch.Tensor:
        """(out, in) low-rank delta — has grad through lora_A/B."""
        return (self.lora_B @ self.lora_A) * self.scale

    def merged_weight(self) -> torch.Tensor:
        """W_base + delta as a single detached tensor (for checkpointing)."""
        with torch.no_grad():
            return (self.base.weight + self.get_delta()).detach()


def inject_temporal_lora(unet, rank: int = 32, alpha: float = 1.0) -> int:
    """
    Replace to_q/k/v/out[0] inside every motion_modules Attention block
    with TemporalLoRALinear. Call AFTER insert_unziplora_to_unet and
    set_attn_processor, BEFORE freeze_spatial_layers.
    Returns number of linear layers wrapped.
    """
    count = 0
    for name, module in unet.named_modules():
        if "motion_modules" not in name:
            continue
        if not (hasattr(module, "to_q") and hasattr(module, "to_k")
                and hasattr(module, "to_v") and hasattr(module, "to_out")):
            continue
        if isinstance(getattr(module, "to_q", None), TemporalLoRALinear):
            continue  # idempotent
        for proj in ("to_q", "to_k", "to_v"):
            lin = getattr(module, proj, None)
            if isinstance(lin, nn.Linear):
                setattr(module, proj, TemporalLoRALinear(lin, rank, alpha))
                count += 1
        if (hasattr(module, "to_out") and len(module.to_out) > 0
                and isinstance(module.to_out[0], nn.Linear)):
            module.to_out[0] = TemporalLoRALinear(module.to_out[0], rank, alpha)
            count += 1
    return count


def build_spatial_lora_index(unet) -> Dict[str, object]:
    """
    One-time scan. Returns {temporal_module_name -> spatial_lora_layer}.

    Pairing: strips 'motion_modules.{i}.temporal_transformer' from the
    temporal path to get the equivalent spatial path, then matches by
    shape (same inner_dim).

    Example:
      Temporal: down_blocks.1.attentions.0.motion_modules.0
                    .temporal_transformer.transformer_blocks.0.attn1.to_q
      Spatial:  down_blocks.1.attentions.0
                    .transformer_blocks.0.attn1.to_q
    """
    spatial_map: Dict[str, object] = {}
    for name, module in unet.named_modules():
        if "motion_modules" in name:
            continue
        lora = getattr(module, "lora_layer", None)
        if lora is not None:
            spatial_map[name] = lora

    index: Dict[str, object] = {}
    for name, module in unet.named_modules():
        if "motion_modules" not in name:
            continue
        if not isinstance(module, TemporalLoRALinear):
            continue
        parts = name.split(".")
        try:
            mm_idx = parts.index("motion_modules")
        except ValueError:
            continue
        prefix   = ".".join(parts[:mm_idx])
        after_mm = parts[mm_idx + 2:]          # skip "motion_modules", "{i}"
        if "temporal_transformer" in after_mm:
            tt_idx   = after_mm.index("temporal_transformer")
            after_mm = after_mm[tt_idx + 1:]   # skip "temporal_transformer"
        spatial_path = prefix + "." + ".".join(after_mm)
        if spatial_path not in spatial_map:
            continue
        lora = spatial_map[spatial_path]
        try:
            B_c = lora.lora_matrix_dic.content_up.weight
            A_c = lora.lora_matrix_dic.content_down.weight
            if (B_c.shape[0] == module.out_features
                    and A_c.shape[1] == module.in_features):
                index[name] = lora
        except AttributeError:
            continue
    return index


def compute_orth_loss(unet, spatial_index: Dict, lambda_orth: float) -> torch.Tensor:
    """
    Corrected symmetric Option B loss:

      L_orth = (1/N) * sum_l [ ||(B_t A_t)^T (B_c A_c)||_F^2
                               + ||(B_t A_t)^T (B_s A_s)||_F^2 ]

    delta_temp = B_t @ A_t  (out,in) — has grad  (temporal LoRA params)
    delta_c/s  = B_c @ A_c  (out,in) — detached  (frozen Stage-1 spatial LoRA)

    Both sides are rank-r deltas in the same weight space — the constraint is
    symmetric and well-posed (unlike the earlier full-weight vs delta version).
    """
    if lambda_orth == 0.0 or not spatial_index:
        return torch.tensor(0.0)

    total: Optional[torch.Tensor] = None
    count = 0
    for name, module in unet.named_modules():
        if name not in spatial_index:
            continue
        if not isinstance(module, TemporalLoRALinear):
            continue
        lora       = spatial_index[name]
        delta_temp = module.get_delta()       # (out,in), has grad
        with torch.no_grad():
            B_c     = lora.lora_matrix_dic.content_up.weight.float()
            A_c     = lora.lora_matrix_dic.content_down.weight.float()
            B_s     = lora.lora_matrix_dic.style_up.weight.float()
            A_s     = lora.lora_matrix_dic.style_down.weight.float()
            delta_c = (B_c @ A_c).detach()
            delta_s = (B_s @ A_s).detach()
        W = delta_temp.float()
        contrib = (torch.sum((W.T @ delta_c) ** 2)
                 + torch.sum((W.T @ delta_s) ** 2))
        total = contrib if total is None else total + contrib
        count += 1

    if total is None or count == 0:
        return torch.tensor(0.0)
    return lambda_orth * total / count


def get_merged_motion_state_dict(unet) -> Dict[str, torch.Tensor]:
    """
    Build a motion_modules state dict with temporal LoRA deltas folded into
    base weights. Output uses original key names (no .base. prefix), so it
    loads into a fresh UNetMotionModel without TemporalLoRALinear wrappers.
    """
    merged: Dict[str, torch.Tensor] = {}
    wrapped_paths = set()
    for name, module in unet.named_modules():
        if not isinstance(module, TemporalLoRALinear):
            continue
        if "motion_modules" not in name:
            continue
        merged[name + ".weight"] = module.merged_weight().cpu()
        if module.base.bias is not None:
            merged[name + ".bias"] = module.base.bias.detach().cpu()
        wrapped_paths.add(name)
    for k, v in unet.state_dict().items():
        if "motion_modules" not in k:
            continue
        if any(k.startswith(wp + ".") for wp in wrapped_paths):
            continue
        merged[k] = v.detach().cpu()
    return merged
