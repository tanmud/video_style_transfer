import os
import torch
from typing import List, Optional

from diffusers import MotionAdapter
from diffusers.models.unet_motion_model import UNetMotionModel
from diffusers.utils import logging
from diffusers import UNet2DConditionModel


logger = logging.get_logger(__name__)


def load_unet_with_motion(
    pretrained_model_name_or_path: str,
    motion_adapter_path: str,
    torch_dtype: torch.dtype = torch.float32,
    device: str = "cuda",
):
    """
    Load UNetMotionModel from a base SDXL checkpoint + motion adapter.

    motion_adapter_path accepts three forms:
      1. HF hub id            e.g. "guoyww/animatediff-motion-adapter-sdxl-beta"
      2. Local adapter dir    a directory containing adapter_config.json + safetensors
      3. Local checkpoint dir a directory containing motion_modules.pth (saved by save_checkpoint)
    """
    logger.info(f"Loading base UNet from {pretrained_model_name_or_path}")
    base_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch_dtype,
    )

    # ── Detect which form motion_adapter_path is ─────────────────────────
    pth_file = _find_pth(motion_adapter_path)

    if pth_file is not None:
        # Form 3 — local checkpoint dir with motion_modules.pth
        logger.info(f"Detected local motion checkpoint: {pth_file}")
        logger.info("Initialising UNetMotionModel with default motion modules...")
        unet = UNetMotionModel.from_unet2d(base_unet)  # default (random) motion weights

        logger.info("Loading saved motion weights via load_state_dict...")
        state_dict = torch.load(pth_file, map_location="cpu")
        missing, unexpected = unet.load_state_dict(state_dict, strict=False)

        motion_keys_loaded = [k for k in state_dict if "motion_modules" in k]
        logger.info(f"Loaded {len(motion_keys_loaded)} motion module tensors")
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:3]} ...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
    else:
        # Form 1 or 2 — HF hub id or local adapter dir with config.json
        logger.info(f"Loading MotionAdapter from {motion_adapter_path}")
        motion_adapter = MotionAdapter.from_pretrained(
            motion_adapter_path, torch_dtype=torch_dtype
        )
        unet = UNetMotionModel.from_unet2d(base_unet, motion_adapter)
        logger.info("UNetMotionModel assembled from base UNet + MotionAdapter")

    unet.to(device)
    return unet


def _find_pth(path: str) -> Optional[str]:
    """
    Returns the path to motion_modules.pth if this looks like a local checkpoint dir,
    None otherwise (= treat as HF hub id or MotionAdapter dir).
    """
    # Direct .pth file
    if path.endswith(".pth") and os.path.isfile(path):
        return path
    # Dir containing motion_modules.pth
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
    motion = sum(p.numel() for n, p in unet.named_parameters() if "motion_modules" in n)
    print("=" * 60)
    print(f"Total:     {total:>15,}")
    print(f"Trainable: {trainable:>15,}  ({trainable/total*100:.2f}%)")
    print(f"Motion:    {motion:>15,}")
    print("=" * 60)
import os
import torch
from typing import List, Optional

from diffusers import MotionAdapter
from diffusers.models.unet_motion_model import UNetMotionModel
from diffusers.utils import logging

from unziplora_unet.unet_2d_condition import UNet2DConditionModel

logger = logging.get_logger(__name__)


def load_unet_with_motion(
    pretrained_model_name_or_path: str,
    motion_adapter_path: str,
    torch_dtype: torch.dtype = torch.float32,
    device: str = "cuda",
):
    """
    Load UNetMotionModel from a base SDXL checkpoint + motion adapter.

    motion_adapter_path accepts three forms:
      1. HF hub id            e.g. "guoyww/animatediff-motion-adapter-sdxl-beta"
      2. Local adapter dir    a directory containing adapter_config.json + safetensors
      3. Local checkpoint dir a directory containing motion_modules.pth (saved by save_checkpoint)
    """
    logger.info(f"Loading base UNet from {pretrained_model_name_or_path}")
    base_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch_dtype,
    )

    # ── Detect which form motion_adapter_path is ─────────────────────────
    pth_file = _find_pth(motion_adapter_path)

    if pth_file is not None:
        # Form 3 — local checkpoint dir with motion_modules.pth
        logger.info(f"Detected local motion checkpoint: {pth_file}")
        logger.info("Initialising UNetMotionModel with default motion modules...")
        unet = UNetMotionModel.from_unet2d(base_unet)  # default (random) motion weights

        logger.info("Loading saved motion weights via load_state_dict...")
        state_dict = torch.load(pth_file, map_location="cpu")
        missing, unexpected = unet.load_state_dict(state_dict, strict=False)

        motion_keys_loaded = [k for k in state_dict if "motion_modules" in k]
        logger.info(f"Loaded {len(motion_keys_loaded)} motion module tensors")
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:3]} ...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
    else:
        # Form 1 or 2 — HF hub id or local adapter dir with config.json
        logger.info(f"Loading MotionAdapter from {motion_adapter_path}")
        motion_adapter = MotionAdapter.from_pretrained(
            motion_adapter_path, torch_dtype=torch_dtype
        )
        unet = UNetMotionModel.from_unet2d(base_unet, motion_adapter)
        logger.info("UNetMotionModel assembled from base UNet + MotionAdapter")

    unet.to(device)
    return unet


def _find_pth(path: str) -> Optional[str]:
    """
    Returns the path to motion_modules.pth if this looks like a local checkpoint dir,
    None otherwise (= treat as HF hub id or MotionAdapter dir).
    """
    # Direct .pth file
    if path.endswith(".pth") and os.path.isfile(path):
        return path
    # Dir containing motion_modules.pth
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
    motion = sum(p.numel() for n, p in unet.named_parameters() if "motion_modules" in n)
    print("=" * 60)
    print(f"Total:     {total:>15,}")
    print(f"Trainable: {trainable:>15,}  ({trainable/total*100:.2f}%)")
    print(f"Motion:    {motion:>15,}")
    print("=" * 60)
