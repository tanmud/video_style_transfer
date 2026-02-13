import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image


class VideoDataset(Dataset):
    """
    Video dataset that loads frames from folders.

    Supports two structures:

    1. Multiple videos (subdirectories):
       instance_data_dir/
           video_001/
               frame_000.png
               frame_001.png
           video_002/
               frame_000.png
               ...

    2. Single video (flat directory):
       instance_data_dir/
           frame_000.png
           frame_001.png
           frame_002.png
           ...
    """

    def __init__(self, instance_data_dir, num_frames=16, resolution=512):
        self.instance_data_dir = Path(instance_data_dir)
        self.num_frames = num_frames
        self.resolution = resolution

        if not self.instance_data_dir.exists():
            raise ValueError(f"Directory does not exist: {instance_data_dir}")

        # Check what's in the directory
        all_items = list(self.instance_data_dir.iterdir())
        subdirs = [d for d in all_items if d.is_dir()]
        image_files = [f for f in all_items if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]

        print(f"\nChecking {instance_data_dir}...")
        print(f"  Found {len(subdirs)} subdirectories")
        print(f"  Found {len(image_files)} image files")

        # Determine structure
        if len(subdirs) > 0:
            # Structure 1: Multiple video folders
            self.video_paths = sorted(subdirs)
            self.is_flat = False
            print(f"  → Using multi-video structure: {len(self.video_paths)} videos")
            for i, vp in enumerate(self.video_paths[:3]):
                frames_in_video = len(list(vp.glob("*.png"))) + len(list(vp.glob("*.jpg")))
                print(f"     Video {i}: {vp.name} ({frames_in_video} frames)")
            if len(self.video_paths) > 3:
                print(f"     ... and {len(self.video_paths) - 3} more videos")

        elif len(image_files) > 0:
            # Structure 2: Single video (flat)
            self.video_paths = [self.instance_data_dir]
            self.is_flat = True
            print(f"  → Using flat structure: 1 video with {len(image_files)} frames")

        else:
            raise ValueError(
                f"No video data found in {instance_data_dir}!\n"
                f"Expected either:\n"
                f"  1. Subdirectories containing frames (video_001/, video_002/, ...)\n"
                f"  2. Image files directly in the directory (.png, .jpg)\n"
                f"Found: {len(all_items)} items but none were videos or images\n"
                f"Items: {[str(item) for item in all_items[:10]]}"
            )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        # Load frames (support both .png and .jpg)
        frame_paths = (
            sorted(video_path.glob("*.png")) + 
            sorted(video_path.glob("*.jpg")) +
            sorted(video_path.glob("*.jpeg"))
        )
        frame_paths = sorted(frame_paths)[:self.num_frames]

        if len(frame_paths) == 0:
            raise ValueError(f"No image files found in {video_path}")

        if len(frame_paths) < self.num_frames:
            # Repeat last frame if not enough frames
            frame_paths = frame_paths + [frame_paths[-1]] * (self.num_frames - len(frame_paths))

        frames = []
        for frame_path in frame_paths:
            try:
                img = Image.open(frame_path).convert("RGB")
                img = img.resize((self.resolution, self.resolution))
                img = np.array(img).astype(np.float32) / 255.0
                img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
                frames.append(img)
            except Exception as e:
                raise ValueError(f"Error loading {frame_path}: {e}")

        # Stack: (F, H, W, C) → (F, C, H, W)
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

        return {"frames": frames}


def collate_fn(examples):
    """Collate batch of videos."""
    frames = torch.stack([ex["frames"] for ex in examples])
    return {"frames": frames}