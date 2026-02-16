import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import cv2


class VideoDataset(Dataset):
    """
    Video dataset that loads MP4 files directly.

    Expected structure:
        instance_data_dir/
            video1.mp4
            video2.mp4
            ...

    Or just:
        instance_data_dir/
            video.mp4
    """

    def __init__(self, instance_data_dir, num_frames=16, resolution=512):
        self.instance_data_dir = Path(instance_data_dir)
        self.num_frames = num_frames
        self.resolution = resolution

        if not self.instance_data_dir.exists():
            raise ValueError(f"Directory does not exist: {instance_data_dir}")

        # Find all MP4 files
        self.video_paths = []

        # Check current directory
        video_files = list(self.instance_data_dir.glob("*.mp4"))
        self.video_paths.extend(video_files)

        # Also check subdirectories (one level deep)
        for subdir in self.instance_data_dir.iterdir():
            if subdir.is_dir():
                video_files = list(subdir.glob("*.mp4"))
                self.video_paths.extend(video_files)

        self.video_paths = sorted(self.video_paths)

        if len(self.video_paths) == 0:
            raise ValueError(
                f"No MP4 videos found in {instance_data_dir}!\n"
                f"Expected: .mp4 files in the directory or subdirectories\n"
                f"Found: {list(self.instance_data_dir.iterdir())[:10]}"
            )

        print(f"\nFound {len(self.video_paths)} video(s):")
        for i, vp in enumerate(self.video_paths[:5]):
            cap = cv2.VideoCapture(str(vp))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            print(f"  {i+1}. {vp.name} ({total_frames} frames, {fps:.1f} fps)")
        if len(self.video_paths) > 5:
            print(f"  ... and {len(self.video_paths) - 5} more videos")

    def __len__(self):
        return len(self.video_paths)

    def load_video_frames(self, video_path, num_frames):
        """Load frames from MP4 video."""
        cap = cv2.VideoCapture(str(video_path))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_frames:
            # If video is shorter, we'll repeat last frame
            print(f"Warning: {video_path.name} has only {total_frames} frames, need {num_frames}")
            frame_indices = list(range(total_frames))
        else:
            # Sample evenly spaced frames
            max_start = total_frames - num_frames
            start_frame = np.random.randint(0, max_start + 1) if max_start > 0 else 0

        frames = []
        for idx in range(start_frame, start_frame + num_frames):
            frame_idx = min(frame_idx, total_frames - 1)  # Ensure we don't go out of bounds
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                print(f"Warning: Could not read frame {idx} from {video_path}")
                # Use last successfully read frame
                if len(frames) > 0:
                    frames.append(frames[-1])
                continue

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize
            frame = cv2.resize(frame, (self.resolution, self.resolution))

            # Normalize to [-1, 1]
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - 0.5) / 0.5

            frames.append(frame)

        cap.release()

        # If we don't have enough frames, repeat last frame
        while len(frames) < num_frames:
            frames.append(frames[-1])

        # Only take num_frames
        frames = frames[:num_frames]

        # Stack: (F, H, W, C) → (F, C, H, W)
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

        return frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = self.load_video_frames(video_path, self.num_frames)
        return {"frames": frames}


def collate_fn(examples):
    """Collate batch of videos."""
    frames = torch.stack([ex["frames"] for ex in examples])
    return {"frames": frames}