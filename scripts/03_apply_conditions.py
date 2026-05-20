"""
Script: 03_apply_conditions.py
Purpose: Create degraded versions of LFW images at 5 brightness levels and 4 pose angles.

Output folder structure:
  degraded/
  ├── brightness_0.25/    (very dark)
  ├── brightness_0.50/    (dark)
  ├── brightness_1.00/    (baseline - clean copy)
  ├── brightness_1.50/    (bright / overlit)
  ├── brightness_2.00/    (overexposed)
  ├── pose_0/             (frontal, no rotation)
  ├── pose_15/            (slight turn)
  ├── pose_30/            (moderate turn)
  └── pose_45/            (extreme turn)
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────
LFW_DIR       = r"e:\Environment\CP\face_recognition_thesis\data\lfw\lfw-deepfunneled\lfw-deepfunneled"
DEMOGRAPHICS  = r"e:\Environment\CP\face_recognition_thesis\data\lfw_demographics.csv"
OUTPUT_DIR    = r"e:\Environment\CP\face_recognition_thesis\data\degraded"
IMAGE_SIZE    = 112   # resize all faces to 112×112 before degradation

# Brightness multiplier levels (factor applied to pixel values, clipped to 0-255)
BRIGHTNESS_LEVELS = {
    "brightness_0.25": 0.25,   # very dark  – nighttime surveillance
    "brightness_0.50": 0.50,   # dark       – dimly lit indoor
    "brightness_1.00": 1.00,   # normal     – baseline / clean
    "brightness_1.50": 1.50,   # bright     – harsh lighting
    "brightness_2.00": 2.00,   # overexposed– direct sunlight / backlit
}

# Pose rotation angles (in-plane rotation to simulate head turns)
POSE_LEVELS = {
    "pose_0":  0,    # frontal (baseline)
    "pose_15": 15,   # slight turn
    "pose_30": 30,   # moderate turn
    "pose_45": 45,   # extreme turn
}
# ──────────────────────────────────────────────────────────────────────────────


def apply_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Multiply pixel values by factor, clip to valid range [0, 255]."""
    img_float = img.astype(np.float32) * factor
    return np.clip(img_float, 0, 255).astype(np.uint8)


def apply_pose_rotation(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image in-plane by `angle` degrees around the image centre.
    
    Rotates the face to simulate a non-frontal pose (head-turn / tilt).
    The image is kept the same size – corners may be black after rotation.
    """
    if angle == 0:
        return img
    h, w = img.shape[:2]
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated


def get_image_paths_from_demographics() -> list[str]:
    """Get all image paths that belong to identities in our demographics CSV."""
    if not os.path.exists(DEMOGRAPHICS):
        # Fall back: process ALL images in the dataset
        print(f"WARNING: {DEMOGRAPHICS} not found. Processing all images.")
        all_imgs = glob.glob(os.path.join(LFW_DIR, "*", "*.jpg"))
        return all_imgs

    df = pd.read_csv(DEMOGRAPHICS)
    print(f"Loaded {len(df)} labeled identities.")

    image_paths = []
    for name in df['name']:
        person_dir = os.path.join(LFW_DIR, name)
        imgs = glob.glob(os.path.join(person_dir, "*.jpg"))
        image_paths.extend(imgs)

    print(f"Total images to process: {len(image_paths)}")
    return image_paths


def process_images():
    # Create all output subdirectories
    all_conditions = {**BRIGHTNESS_LEVELS, **{k: v for k, v in POSE_LEVELS.items()}}
    for condition_name in list(BRIGHTNESS_LEVELS.keys()) + list(POSE_LEVELS.keys()):
        os.makedirs(os.path.join(OUTPUT_DIR, condition_name), exist_ok=True)

    image_paths = get_image_paths_from_demographics()

    print(f"\nApplying degradations to {len(image_paths)} images...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    errors = 0
    for img_path in tqdm(image_paths, desc="Processing"):
        img = cv2.imread(img_path)
        if img is None:
            errors += 1
            continue

        # Resize to standard face size
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # Extract relative path: "PersonName/PersonName_0001.jpg"
        person_name = os.path.basename(os.path.dirname(img_path))
        img_filename = os.path.basename(img_path)
        rel_path = os.path.join(person_name, img_filename)

        # ── Apply brightness conditions ────────────────────────────────────
        for condition_name, factor in BRIGHTNESS_LEVELS.items():
            out_dir = os.path.join(OUTPUT_DIR, condition_name, person_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, img_filename)

            degraded = apply_brightness(img, factor)
            cv2.imwrite(out_path, degraded)

        # ── Apply pose conditions ──────────────────────────────────────────
        for condition_name, angle in POSE_LEVELS.items():
            out_dir = os.path.join(OUTPUT_DIR, condition_name, person_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, img_filename)

            degraded = apply_pose_rotation(img, angle)
            cv2.imwrite(out_path, degraded)

    print(f"\n✅ Done! Degraded images saved to: {OUTPUT_DIR}")
    print(f"   Errors (unreadable images): {errors}")
    print(f"\nGenerated conditions:")
    for name in list(BRIGHTNESS_LEVELS.keys()) + list(POSE_LEVELS.keys()):
        count = len(glob.glob(os.path.join(OUTPUT_DIR, name, "*", "*.jpg")))
        print(f"   {name:25s}: {count} images")


if __name__ == "__main__":
    process_images()
