"""
Script: 01_auto_label_lfw.py  (v2 - with resume support and incremental saving)

Key improvements over v1:
- Suppresses TensorFlow log spam entirely
- Saves progress every 10 identities so interrupts don't lose work
- On restart, skips already-processed identities (resume from where we stopped)
- Falls back gracefully on DeepFace detection errors
"""

import os
import sys

# Silence ALL TensorFlow and DeepFace logs before any import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import glob
import warnings
import logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
import time

# ─── Configuration ────────────────────────────────────────────────────────────
LFW_DIR      = r"e:\Environment\CP\face_recognition_thesis\data\lfw\lfw-deepfunneled\lfw-deepfunneled"
OUTPUT_CSV   = r"e:\Environment\CP\face_recognition_thesis\data\lfw_demographics.csv"
MIN_IMAGES   = 5      # Minimum images per identity to be included
SAVE_EVERY   = 10     # Save progress to disk every N identities
# ──────────────────────────────────────────────────────────────────────────────


def get_all_identities(min_images=MIN_IMAGES):
    """Return list of identity dicts with name, num_images, first_image."""
    identities = []
    for person in sorted(os.listdir(LFW_DIR)):
        person_dir = os.path.join(LFW_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        images = glob.glob(os.path.join(person_dir, "*.jpg"))
        if len(images) >= min_images:
            identities.append({
                "name": person,
                "num_images": len(images),
                "first_image": sorted(images)[0],
            })
    return identities


def load_existing_results():
    """Load already-processed identities so we can skip them on resume."""
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"[RESUME] Found existing CSV with {len(df)} labeled identities. Skipping these.")
        return df
    return pd.DataFrame(columns=["name", "num_images", "gender", "race", "subgroup", "sample_image"])


def label_one(img_path):
    """Run DeepFace on a single image and return (gender, race) or (None, None)."""
    try:
        result = DeepFace.analyze(
            img_path=img_path,
            actions=["gender", "race"],
            enforce_detection=False,
            silent=True
        )
        obj = result[0] if isinstance(result, list) else result

        # DeepFace 0.0.93+ returns nested dicts
        gender_raw = obj.get("dominant_gender", None)
        if gender_raw is None:
            g = obj.get("gender", {})
            gender_raw = "Man" if isinstance(g, dict) and g.get("Man", 0) >= g.get("Woman", 0) else "Woman"

        race_raw = obj.get("dominant_race", "unknown")
        return str(gender_raw).capitalize(), str(race_raw).capitalize()

    except Exception:
        return None, None


def run():
    all_identities = get_all_identities()
    print(f"Total identities with {MIN_IMAGES}+ images: {len(all_identities)}")

    existing_df = load_existing_results()
    already_done = set(existing_df["name"].tolist())

    to_process = [i for i in all_identities if i["name"] not in already_done]
    print(f"Remaining to process: {len(to_process)}")

    if not to_process:
        print("All identities already labeled! CSV is complete.")
        return

    results = existing_df.to_dict("records")
    errors = 0
    start = time.time()

    for idx, identity in enumerate(tqdm(to_process, desc="Labeling")):
        gender, race = label_one(identity["first_image"])

        if gender is None or race is None:
            errors += 1
            gender = "Unknown"
            race   = "Unknown"

        results.append({
            "name":         identity["name"],
            "num_images":   identity["num_images"],
            "gender":       gender,
            "race":         race,
            "subgroup":     f"{race}_{gender}",
            "sample_image": identity["first_image"],
        })

        # Incremental save every SAVE_EVERY records
        if (idx + 1) % SAVE_EVERY == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - start
    print(f"\n✅ Done in {elapsed:.0f}s! Saved {len(df)} identities to {OUTPUT_CSV}")
    print(f"   Errors (face not detected): {errors}/{len(to_process)}")
    print("\n--- Subgroup Distribution ---")
    print(df["subgroup"].value_counts().to_string())


if __name__ == "__main__":
    run()
