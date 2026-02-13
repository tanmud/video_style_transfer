import torch
import os
from typing import Optional, Dict, List
from diffusers import UNet2DConditionModel
from diffusers.utils import logging

logger = logging.get_logger(__name__)

from animatediff.unet_animatediff_condition import UNetAnimateDiffConditionModel

def load_unet_with_motion(
    pretrained_model_name_or_path: str,
    motion_module_path: Optional[str] = None,
    motion_module_kwargs: Optional[Dict] = None,
    torch_dtype: torch.dtype = torch.float32,
    device: str = "cuda",
):
    """
    Load UNet with motion modules, optionally loading trained motion weights.

    Args:
        pretrained_model_name_or_path: Path to SDXL checkpoint
        motion_module_path: Path to trained motion_modules.pth (optional)
        motion_module_kwargs: Config for motion modules (num_layers, etc.)
        torch_dtype: Data type for model
        device: Device to load model on

    Returns:
        UNet with motion modules loaded

    Example:
        # Load with randomly initialized motion modules
        unet = load_unet_with_motion(
            "stabilityai/stable-diffusion-xl-base-1.0",
            motion_module_kwargs={"num_layers": 2}
        )

        # Load with trained motion modules
        unet = load_unet_with_motion(
            "stabilityai/stable-diffusion-xl-base-1.0",
            motion_module_path="./outputs/motion_modules.pth",
            motion_module_kwargs={"num_layers": 2}
        )
    """

    if motion_module_kwargs is None:
        motion_module_kwargs = {}

    logger.info(f"Loading UNet from {pretrained_model_name_or_path}")

    # Load UNet with motion modules
    unet = UNetAnimateDiffConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch_dtype,
        motion_module_kwargs=motion_module_kwargs,
    )

    # Load trained motion modules if provided
    if motion_module_path is not None:
        logger.info(f"Loading motion modules from {motion_module_path}")
        unet.load_motion_modules(motion_module_path)
    else:
        logger.info("Using randomly initialized motion modules (for training)")

    unet.to(device)
    return unet


def freeze_spatial_layers(unet):
    """
    Freeze all spatial layers, only train temporal transformers.

    This is the recommended training strategy:
    - Preserves pretrained SDXL quality
    - Faster training
    - Lower memory usage
    - Only learns motion patterns

    Args:
        unet: UNetAnimateDiffConditionModel
    """
    trainable_count = 0
    frozen_count = 0

    for name, param in unet.named_parameters():
        if "temporal" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    logger.info(f"Froze {frozen_count:,} spatial parameters")
    logger.info(f"Training {trainable_count:,} temporal parameters")
    logger.info(f"Trainable ratio: {trainable_count / (trainable_count + frozen_count) * 100:.2f}%")


def unfreeze_all_layers(unet):
    """
    Unfreeze all layers for full fine-tuning.

    Use this if you want to adapt both spatial and temporal layers
    to your specific video dataset.

    Args:
        unet: UNetAnimateDiffConditionModel
    """
    trainable_count = 0

    for name, param in unet.named_parameters():
        param.requires_grad = True
        trainable_count += param.numel()

    logger.info(f"Training all {trainable_count:,} parameters")


def get_trainable_parameters(unet) -> List[torch.nn.Parameter]:
    """
    Get list of trainable parameters.

    Args:
        unet: UNetAnimateDiffConditionModel

    Returns:
        List of parameters with requires_grad=True
    """
    return [p for p in unet.parameters() if p.requires_grad]


def count_parameters(unet) -> Dict[str, int]:
    """
    Count parameters in different parts of the model.

    Args:
        unet: UNetAnimateDiffConditionModel

    Returns:
        Dict with parameter counts for different components
    """
    counts = {
        "total": 0,
        "trainable": 0,
        "frozen": 0,
        "spatial": 0,
        "temporal": 0,
        "down_temporal": 0,
        "mid_temporal": 0,
        "up_temporal": 0,
    }

    for name, param in unet.named_parameters():
        num_params = param.numel()
        counts["total"] += num_params

        if param.requires_grad:
            counts["trainable"] += num_params
        else:
            counts["frozen"] += num_params

        if "temporal" in name:
            counts["temporal"] += num_params
            if "down_temporal" in name:
                counts["down_temporal"] += num_params
            elif "mid_temporal" in name:
                counts["mid_temporal"] += num_params
            elif "up_temporal" in name:
                counts["up_temporal"] += num_params
        else:
            counts["spatial"] += num_params

    return counts


def print_parameter_summary(unet):
    """
    Print a summary of model parameters.

    Args:
        unet: UNetAnimateDiffConditionModel
    """
    counts = count_parameters(unet)

    print("=" * 60)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 60)
    print(f"Total parameters:        {counts['total']:>15,}")
    print(f"Trainable parameters:    {counts['trainable']:>15,}")
    print(f"Frozen parameters:       {counts['frozen']:>15,}")
    print("-" * 60)
    print(f"Spatial parameters:      {counts['spatial']:>15,}")
    print(f"Temporal parameters:     {counts['temporal']:>15,}")
    print(f"  - Down temporal:       {counts['down_temporal']:>15,}")
    print(f"  - Mid temporal:        {counts['mid_temporal']:>15,}")
    print(f"  - Up temporal:         {counts['up_temporal']:>15,}")
    print("=" * 60)

    if counts['trainable'] > 0:
        ratio = counts['trainable'] / counts['total'] * 100
        print(f"Training {ratio:.2f}% of parameters")
    print()


def save_checkpoint(
    unet,
    output_dir: str,
    step: int,
    save_full_model: bool = False,
):
    """
    Save training checkpoint.

    Args:
        unet: UNetAnimateDiffConditionModel
        output_dir: Directory to save checkpoint
        step: Training step number
        save_full_model: If True, save full UNet. If False, only save motion modules.

    By default, only saves motion modules (~100MB) for efficiency.
    Set save_full_model=True to save complete model (~5GB).
    """
    os.makedirs(output_dir, exist_ok=True)

    if save_full_model:
        # Save complete UNet
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
        unet.save_pretrained(checkpoint_path)
        logger.info(f"Saved full model checkpoint to {checkpoint_path}")
    else:
        # Save only motion modules (recommended)
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        unet.save_motion_modules(checkpoint_path)
        logger.info(f"Saved motion modules checkpoint to {checkpoint_path}")


def merge_motion_modules(
    base_unet_path: str,
    motion_module_path: str,
    output_path: str,
):
    """
    Merge motion modules with a base SDXL UNet and save complete model.

    This allows you to:
    1. Train motion on one SDXL checkpoint
    2. Merge those motions with a different SDXL checkpoint
    3. Save as a complete AnimateDiff UNet

    Args:
        base_unet_path: Path to base SDXL checkpoint
        motion_module_path: Path to trained motion_modules.pth
        output_path: Where to save merged model

    Example:
        merge_motion_modules(
            base_unet_path="my-custom-sdxl/unet",
            motion_module_path="./trained_motion/motion_modules.pth",
            output_path="./merged_animatediff_unet"
        )
    """
    

    logger.info(f"Loading base UNet from {base_unet_path}")
    unet = UNetAnimateDiffConditionModel.from_pretrained(
        base_unet_path,
        subfolder="unet" if not base_unet_path.endswith("unet") else None,
    )

    logger.info(f"Loading motion modules from {motion_module_path}")
    unet.load_motion_modules(motion_module_path)

    logger.info(f"Saving merged model to {output_path}")
    unet.save_pretrained(output_path)

    logger.info("Merge complete!")


def convert_standard_unet_to_animatediff(
    unet_path: str,
    output_path: str,
    motion_module_kwargs: Optional[Dict] = None,
):
    """
    Convert a standard UNet2DConditionModel to AnimateDiff version.

    Adds randomly initialized motion modules to an existing SDXL UNet.
    The result needs to be trained on video data.

    Args:
        unet_path: Path to standard SDXL UNet
        output_path: Where to save AnimateDiff version
        motion_module_kwargs: Config for motion modules

    Example:
        convert_standard_unet_to_animatediff(
            unet_path="my-finetuned-sdxl/unet",
            output_path="my-finetuned-sdxl-animatediff",
            motion_module_kwargs={"num_layers": 2}
        )
    """

    if motion_module_kwargs is None:
        motion_module_kwargs = {}

    logger.info(f"Loading standard UNet from {unet_path}")
    unet = UNetAnimateDiffConditionModel.from_pretrained(
        unet_path,
        motion_module_kwargs=motion_module_kwargs,
    )

    logger.info(f"Saving AnimateDiff UNet to {output_path}")
    unet.save_pretrained(output_path)

    logger.info("Conversion complete! Motion modules are randomly initialized.")
    logger.info("Train on video data before using for inference.")


def check_motion_module_compatibility(motion_module_path: str) -> Dict:
    """
    Check if motion module file is valid and get info.

    Args:
        motion_module_path: Path to motion_modules.pth

    Returns:
        Dict with info about the motion modules
    """
    if not os.path.exists(motion_module_path):
        raise FileNotFoundError(f"Motion module file not found: {motion_module_path}")

    state_dict = torch.load(motion_module_path, map_location="cpu")

    required_keys = ["down_temporal", "mid_temporal", "up_temporal"]
    for key in required_keys:
        if key not in state_dict:
            raise ValueError(f"Invalid motion module file: missing '{key}' key")

    # Count parameters
    info = {
        "valid": True,
        "down_temporal_params": sum(p.numel() for p in state_dict["down_temporal"].values()),
        "mid_temporal_params": sum(p.numel() for p in state_dict["mid_temporal"].values()),
        "up_temporal_params": sum(p.numel() for p in state_dict["up_temporal"].values()),
    }
    info["total_params"] = sum([
        info["down_temporal_params"],
        info["mid_temporal_params"],
        info["up_temporal_params"],
    ])

    logger.info(f"Motion module file is valid")
    logger.info(f"Total parameters: {info['total_params']:,}")

    return info


def get_optimizer_param_groups(
    unet,
    spatial_lr: float = 1e-5,
    temporal_lr: float = 1e-4,
):
    """
    Create parameter groups with different learning rates.

    Useful for full fine-tuning where you want:
    - Lower LR for pretrained spatial layers
    - Higher LR for randomly initialized temporal layers

    Args:
        unet: UNetAnimateDiffConditionModel
        spatial_lr: Learning rate for spatial parameters
        temporal_lr: Learning rate for temporal parameters

    Returns:
        List of parameter groups for optimizer

    Example:
        param_groups = get_optimizer_param_groups(unet, spatial_lr=1e-5, temporal_lr=1e-4)
        optimizer = torch.optim.AdamW(param_groups)
    """
    spatial_params = []
    temporal_params = []

    for name, param in unet.named_parameters():
        if not param.requires_grad:
            continue

        if "temporal" in name:
            temporal_params.append(param)
        else:
            spatial_params.append(param)

    param_groups = []
    if spatial_params:
        param_groups.append({
            "params": spatial_params,
            "lr": spatial_lr,
            "name": "spatial",
        })
    if temporal_params:
        param_groups.append({
            "params": temporal_params,
            "lr": temporal_lr,
            "name": "temporal",
        })

    logger.info(f"Created {len(param_groups)} parameter groups")
    logger.info(f"  Spatial: {len(spatial_params)} params @ LR {spatial_lr}")
    logger.info(f"  Temporal: {len(temporal_params)} params @ LR {temporal_lr}")

    return param_groups