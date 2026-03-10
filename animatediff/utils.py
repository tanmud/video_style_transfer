import os
import torch
from typing import List, Optional

from diffusers import MotionAdapter, UNet2DConditionModel
from diffusers import UNetMotionModel
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
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch_dtype,
    )

    pth_file = _find_pth(motion_adapter_path)

    if pth_file is not None:
        motion_adapter = None
        logger.info(f"Will load trained motion weights from {pth_file}")
    else:
        logger.info(f"Loading MotionAdapter from {motion_adapter_path}")
        motion_adapter = MotionAdapter.from_pretrained(
            motion_adapter_path, torch_dtype=torch_dtype
        )

    # from_unet2d has a bug in older diffusers: iterates ALL down blocks and calls
    # .attentions on each — but SDXL's first DownBlock2D has no .attentions.
    # Try it first; fall back to manual state_dict transfer if it hits that bug.
    try:
        unet = UNetMotionModel.from_unet2d(base_unet, motion_adapter)
        logger.info("from_unet2d succeeded")
    except AttributeError as e:
        logger.warning(f"from_unet2d failed ({e}) — using state_dict fallback")
        unet = _state_dict_build_motion_unet(base_unet, motion_adapter, torch_dtype)

    # Load trained motion weights if this is a saved checkpoint
    if pth_file is not None:
        logger.info(f"Loading saved motion weights from {pth_file}")
        sd = torch.load(pth_file, map_location="cpu")
        missing, unexpected = unet.load_state_dict(sd, strict=False)
        n_loaded = len([k for k in sd if "motion_modules" in k])
        logger.info(f"Loaded {n_loaded} motion module tensors")
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:3]} ...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")

    motion_max_seq = motion_adapter.config.motion_max_seq_length if motion_adapter is not None else None 
    unet.to(device)
    return unet, motion_max_seq


def _state_dict_build_motion_unet(base_unet, motion_adapter, torch_dtype):
    """
    Fallback builder: initialise UNetMotionModel from config then copy weights
    via load_state_dict — avoids from_unet2d's DownBlock2D attentions bug.
    """
    unet = UNetMotionModel.from_config(base_unet.config)
    unet = unet.to(torch_dtype)

    # Copy all matching spatial weights
    missing_spatial, _ = unet.load_state_dict(base_unet.state_dict(), strict=False)
    logger.info(f"Spatial weights copied (unmatched: {len(missing_spatial)})")

    # Overlay motion adapter weights
    if motion_adapter is not None:
        missing_motion, _ = unet.load_state_dict(
            motion_adapter.state_dict(), strict=False
        )
        logger.info(f"Motion weights copied (unmatched: {len(missing_motion)})")

    return unet


def _find_pth(path: str) -> Optional[str]:
    if path.endswith(".pth") and os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidate = os.path.join(path, "motion_modules.pth")
        if os.path.isfile(candidate):
            return candidate
    return None


def freeze_spatial_layers(unet):
    trainable, frozen = 0, 0
    for name, param in unet.named_parameters():
        if "motion_modules" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    logger.info(f"Frozen:    {frozen:,} spatial parameters")
    logger.info(f"Trainable: {trainable:,} motion module parameters")
    logger.info(f"Ratio:     {trainable / (trainable + frozen) * 100:.2f}%")


def get_trainable_parameters(unet) -> List[torch.nn.Parameter]:
    return [p for p in unet.parameters() if p.requires_grad]


def save_checkpoint(unet, output_dir: str, step, save_full_model: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_path, exist_ok=True)
    if save_full_model:
        unet.save_pretrained(checkpoint_path)
        logger.info(f"Saved full model to {checkpoint_path}")
    else:
        motion_state = {
            k: v.cpu()
            for k, v in unet.state_dict().items()
            if "motion_modules" in k
        }
        save_path = os.path.join(checkpoint_path, "motion_modules.pth")
        torch.save(motion_state, save_path)
        logger.info(f"Saved motion modules to {save_path}")


def print_parameter_summary(unet):
    total = sum(p.numel() for p in unet.parameters())
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    motion = sum(
        p.numel() for n, p in unet.named_parameters() if "motion_modules" in n
    )
    print("=" * 60)
    print(f"Total:     {total:>15,}")
    print(f"Trainable: {trainable:>15,}  ({trainable/total*100:.2f}%)")
    print(f"Motion:    {motion:>15,}")
    print("=" * 60)
