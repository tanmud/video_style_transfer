"""
animatediff/utils.py  —  replaces utils-5.py
"""
import os
from typing import List, Optional
import torch
from diffusers import MotionAdapter, UNet2DConditionModel, UNetMotionModel
from diffusers.utils import logging

logger = logging.get_logger(__name__)


def load_unet_with_motion(
    pretrained_model_name_or_path: str,
    motion_adapter_path: str,
    torch_dtype: torch.dtype = torch.float32,
    device: str = "cuda",
):
    logger.info(f"Loading base UNet from {pretrained_model_name_or_path}")
    base_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", torch_dtype=torch_dtype,
    )
    pth_file = _find_pth(motion_adapter_path)
    if pth_file is not None:
        motion_adapter = None
        logger.info(f"Will load trained motion weights from {pth_file}")
    else:
        logger.info(f"Loading MotionAdapter from {motion_adapter_path}")
        motion_adapter = MotionAdapter.from_pretrained(motion_adapter_path, torch_dtype=torch_dtype)
    try:
        unet = UNetMotionModel.from_unet2d(base_unet, motion_adapter)
    except AttributeError as e:
        logger.warning(f"from_unet2d failed ({e}) — using state_dict fallback")
        unet = _state_dict_build_motion_unet(base_unet, motion_adapter, torch_dtype)
    if pth_file is not None:
        sd = torch.load(pth_file, map_location="cpu")
        missing, unexpected = unet.load_state_dict(sd, strict=False)
        n_loaded = len([k for k in sd if "motion_modules" in k])
        logger.info(f"Loaded {n_loaded} motion module tensors")
        if missing:    logger.warning(f"Missing keys ({len(missing)}): {missing[:3]} ...")
        if unexpected: logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
    motion_max_seq = (motion_adapter.config.motion_max_seq_length
                      if motion_adapter is not None else None)
    unet.to(device)
    return unet, motion_max_seq


def _state_dict_build_motion_unet(base_unet, motion_adapter, torch_dtype):
    unet = UNetMotionModel.from_config(base_unet.config).to(torch_dtype)
    unet.load_state_dict(base_unet.state_dict(), strict=False)
    if motion_adapter is not None:
        unet.load_state_dict(motion_adapter.state_dict(), strict=False)
    return unet


def _find_pth(path: str) -> Optional[str]:
    if path.endswith(".pth") and os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidate = os.path.join(path, "motion_modules.pth")
        if os.path.isfile(candidate):
            return candidate
    return None


def freeze_spatial_layers(unet, unfreeze_mergers: bool = False):
    """
    Freeze all non-temporal parameters.

    motion_modules params:
      - .base.weight / .base.bias  →  frozen  (TemporalLoRALinear base, Option B)
      - .lora_A / .lora_B          →  trainable (temporal delta, Option B)
      - everything else            →  trainable (norms, pos_embed, etc.)
    merge_content / merge_style    →  trainable iff unfreeze_mergers=True (Option C)
    all spatial weights            →  frozen
    """
    trainable = frozen = merger_count = temporal_lora_count = 0
    for name, param in unet.named_parameters():
        if "motion_modules" in name:
            if ".base.weight" in name or ".base.bias" in name:
                param.requires_grad_(False); frozen += param.numel()
            else:
                param.requires_grad_(True);  trainable += param.numel()
                if ".lora_A" in name or ".lora_B" in name:
                    temporal_lora_count += param.numel()
        elif unfreeze_mergers and ("merge_content" in name or "merge_style" in name):
            param.requires_grad_(True);  trainable += param.numel()
            merger_count += param.numel()
        else:
            param.requires_grad_(False); frozen += param.numel()

    logger.info(f"Frozen:    {frozen:,}")
    logger.info(f"Trainable: {trainable:,}  ({trainable/(trainable+frozen)*100:.2f}%)")
    if temporal_lora_count: logger.info(f"  Temporal LoRA (A+B): {temporal_lora_count:,}")
    if merger_count:        logger.info(f"  Merger scalars:      {merger_count:,}")


def get_trainable_parameters(unet) -> List[torch.nn.Parameter]:
    return [p for p in unet.parameters() if p.requires_grad]


def save_checkpoint(unet, output_dir: str, step,
                    save_full_model: bool = False, save_mergers: bool = False):
    """
    Save motion weights.
    - If temporal LoRA active (Option B): saves merged weights (W_base + B@A)
      under original key names — loadable by fresh UNetMotionModel unchanged.
    - If save_mergers=True (Option C): saves merger_*_stage2.pth alongside,
      same format as Stage-1 merger files for direct use at inference.
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_path, exist_ok=True)

    if save_full_model:
        unet.save_pretrained(checkpoint_path)
        logger.info(f"Saved full model to {checkpoint_path}")
        return

    try:
        from animatediff.temporal_lora import TemporalLoRALinear, get_merged_motion_state_dict
        has_temporal_lora = any(isinstance(m, TemporalLoRALinear) for m in unet.modules())
    except ImportError:
        has_temporal_lora = False

    if has_temporal_lora:
        motion_state = get_merged_motion_state_dict(unet)
        logger.info("Saving merged motion weights (temporal LoRA folded in)")
    else:
        motion_state = {k: v.cpu() for k, v in unet.state_dict().items()
                        if "motion_modules" in k}

    motion_path = os.path.join(checkpoint_path, "motion_modules.pth")
    torch.save(motion_state, motion_path)
    logger.info(f"Saved motion modules → {motion_path}")

    if save_mergers:
        merger_c, merger_s = _extract_merger_state_dicts(unet)
        if merger_c:
            torch.save(merger_c, os.path.join(checkpoint_path, "merger_content_stage2.pth"))
            torch.save(merger_s, os.path.join(checkpoint_path, "merger_style_stage2.pth"))
            logger.info(f"Saved Stage-2 mergers → {checkpoint_path}/merger_*_stage2.pth")
        else:
            logger.warning("save_mergers=True but no merger scalars found in UNet")


def _extract_merger_state_dicts(unet):
    """
    Extract merger scalars in the format use_lora_mergers_for_inference() expects:
      key: "unet.{attn_name}.{proj}.lora.merge_content"
    These files can directly replace Stage-1 merger .pth files at inference.
    """
    merger_content, merger_style = {}, {}
    for name, module in unet.named_modules():
        if "motion_modules" in name: continue
        lora = getattr(module, "lora_layer", None)
        if lora is None: continue
        mc = getattr(lora, "merge_content", None)
        ms = getattr(lora, "merge_style",   None)
        if mc is None or ms is None: continue
        merger_content[f"unet.{name}.lora.merge_content"] = mc.detach().cpu()
        merger_style  [f"unet.{name}.lora.merge_style"]   = ms.detach().cpu()
    return merger_content, merger_style


def print_parameter_summary(unet):
    try:
        from animatediff.temporal_lora import TemporalLoRALinear
        temporal_lora = sum(p.numel() for n, p in unet.named_parameters()
                            if "motion_modules" in n and (".lora_A" in n or ".lora_B" in n))
    except ImportError:
        temporal_lora = 0
    mergers   = sum(p.numel() for n, p in unet.named_parameters()
                    if "merge_content" in n or "merge_style" in n)
    total     = sum(p.numel() for p in unet.parameters())
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print("=" * 62)
    print(f"  Total:          {total:>15,}")
    print(f"  Trainable:      {trainable:>15,}  ({trainable/total*100:.2f}%)")
    if temporal_lora: print(f"    Temporal LoRA: {temporal_lora:>15,}")
    if mergers and any(p.requires_grad for n, p in unet.named_parameters()
                       if "merge_content" in n or "merge_style" in n):
        print(f"    Mergers:       {mergers:>15,}  [Option C active]")
    print("=" * 62)
