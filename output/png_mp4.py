#!/usr/bin/env python
import cv2
import os
import argparse
from pathlib import Path

def images_to_video(input_folder, output_path, fps=8):
    """Convert frame_0000.png to frame_NNNN.png to MP4 video"""
    
    image_folder = Path(input_folder)
    
    # Find all frame_####.png files
    images = sorted(image_folder.glob("frame_*.png"))
    
    if not images:
        print(f"❌ No images found matching frame_*.png in {input_folder}")
        return
    
    print(f"📁 Found {len(images)} images")
    print(f"   First: {images[0].name}")
    print(f"   Last: {images[-1].name}")
    
    # Read first image to get dimensions
    first_frame = cv2.imread(str(images[0]))
    if first_frame is None:
        print(f"❌ Could not read {images[0]}")
        return
    
    height, width, _ = first_frame.shape
    print(f"📐 Resolution: {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    print(f"🎬 Creating video at {fps} FPS...")
    for i, image_path in enumerate(images):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"⚠️  Skipping {image_path.name} (could not read)")
            continue
        
        video.write(img)
        print(f"  Processing: {i+1}/{len(images)} - {image_path.name}", end='\r')
    
    video.release()
    print(f"\n✅ Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert frame_NNNN.png to MP4")
    parser.add_argument("input_folder", type=str, help="Folder containing frame_*.png files")
    parser.add_argument("-o", "--output", type=str, default="output.mp4", help="Output video file")
    parser.add_argument("-f", "--fps", type=int, default=8, help="Frames per second (default: 8)")
    
    args = parser.parse_args()
    
    images_to_video(args.input_folder, args.output, args.fps)

if __name__ == "__main__":
    main()
