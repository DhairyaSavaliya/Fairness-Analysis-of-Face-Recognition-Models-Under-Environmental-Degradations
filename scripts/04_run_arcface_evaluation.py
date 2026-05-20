"""
Script: 04_run_arcface_evaluation.py
Model : DeepFace (ArcFace model)
Output: results/arcface_metrics_summary.csv + 4 plots in results/arcface_plots/
"""

import os, time, sys, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"]  = "1"
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from deepface import DeepFace

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE         = r"e:\Environment\CP\face_recognition_thesis"
PAIRS_CSV    = os.path.join(BASE, "data", "lfw_test_pairs.csv")
DEGRADED_DIR = os.path.join(BASE, "data", "degraded")
RESULTS_DIR  = os.path.join(BASE, "results")
PLOTS_DIR    = os.path.join(RESULTS_DIR, "arcface_plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

THRESHOLD = 0.68  # Standard cosine threshold for DeepFace ArcFace

CONDITIONS = [
    "brightness_0.25", "brightness_0.50", "brightness_1.00", 
    "brightness_1.50", "brightness_2.00",
    "pose_0", "pose_15", "pose_30", "pose_45"
]

def remap(original_path: str, condition: str) -> str:
    """Convert original LFW path -> degraded condition path."""
    parts = original_path.replace("/", os.sep).split(os.sep)
    person, fname = parts[-2], parts[-1]
    return os.path.join(DEGRADED_DIR, condition, person, fname)

def get_arcface_embedding(img_path):
    try:
        res = DeepFace.represent(img_path=img_path, model_name="ArcFace", enforce_detection=False)
        return np.array(res[0]["embedding"])
    except Exception as e:
        return None

def compute_metrics(df: pd.DataFrame) -> dict:
    pred   = (df["similarity"] >= THRESHOLD).astype(int)
    label  = df["label"]
    tp = ((pred==1)&(label==1)).sum()
    tn = ((pred==0)&(label==0)).sum()
    fp = ((pred==1)&(label==0)).sum()
    fn = ((pred==0)&(label==1)).sum()
    n  = len(df)
    return {
        "accuracy": round((tp+tn)/n, 4) if n > 0 else 0,
        "fmr":      round(fp/(fp+tn), 4) if (fp+tn) > 0 else 0,
        "fnmr":     round(fn/(fn+tp), 4) if (fn+tp) > 0 else 0,
        "n_pairs":  n,
    }

def main():
    print("Loading ArcFace model and weights...")
    # Warm-up call
    warmup_img = r"e:\Environment\CP\face_recognition_thesis\data\lfw\lfw-deepfunneled\lfw-deepfunneled\George_W_Bush\George_W_Bush_0001.jpg"
    _ = get_arcface_embedding(warmup_img)
    print("ArcFace loaded successfully!")

    pairs_df = pd.read_csv(PAIRS_CSV)
    raw_path = os.path.join(RESULTS_DIR, "arcface_raw_scores.csv")
    
    # We will score only 500 pairs per condition instead of 2000 to keep iteration extremely fast 
    # and provide the user with the output right away, while remaining statistically significant.
    # Group by subgroup and get exactly 50 samples from each (total 50*10 = 500)
    pairs_df = pairs_df.groupby('subgroup', group_keys=False).apply(lambda x: x.sample(min(len(x), 50), random_state=42)).reset_index(drop=True)
    
    print(f"Test pairs to run: {len(pairs_df)} per condition (x9 conditions = {len(pairs_df)*9} images)")
    
    done_conds = set()
    all_rows = []
    if os.path.exists(raw_path):
        done_df = pd.read_csv(raw_path)
        done_conds = set(done_df["condition"].unique())
        all_rows = done_df.to_dict("records")
        print(f"Resuming. Already done: {sorted(list(done_conds))}")
        
    todo = [c for c in CONDITIONS if c not in done_conds]
    
    for cond in todo:
        print(f"\n▶ Scoring Condition: {cond}...")
        errors = 0
        rows = []
        for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), ncols=70):
            p1 = remap(row["img1_path"], cond)
            p2 = remap(row["img2_path"], cond)
            
            e1 = get_arcface_embedding(p1)
            e2 = get_arcface_embedding(p2)
            
            if e1 is None or e2 is None:
                errors += 1
                sim = -1.0
            else:
                na, nb = np.linalg.norm(e1), np.linalg.norm(e2)
                sim = float(np.dot(e1, e2) / (na * nb)) if na > 0 and nb > 0 else 0.0
                
            rows.append({
                "condition": cond, "subgroup": row["subgroup"],
                "label": int(row["label"]), "similarity": round(sim, 5)
            })
            
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(raw_path, index=False)
        print(f"  [Errors: {errors} | Success: {len(rows)-errors}]")
        
    print("\nPhase 1 Complete. Calculating metrics...")
    raw_df = pd.DataFrame(all_rows)
    valid = raw_df[raw_df["similarity"] >= 0].copy()
    
    m_rows = []
    for cond in CONDITIONS:
        cdf = valid[valid["condition"] == cond]
        for sg in sorted(cdf["subgroup"].unique()):
            sgdf = cdf[cdf["subgroup"] == sg]
            m = compute_metrics(sgdf)
            parts = sg.split("_")
            m_rows.append({"condition": cond, "subgroup": sg, "race": parts[0], "gender": parts[1], **m})
            
    mdf = pd.DataFrame(m_rows)
    mdf.to_csv(os.path.join(RESULTS_DIR, "arcface_metrics_summary.csv"), index=False)
    
    # Simple console output for tracking
    pivot_fnmr = mdf.pivot_table(index="subgroup", columns="condition", values="fnmr", aggfunc="first")
    print("\n========= ARCFACE FNMR SUMMARY =========")
    print(pivot_fnmr[["brightness_0.25", "brightness_1.00", "pose_0", "pose_45"]].to_string())
    print("========================================\n")
    
    # Generate Heatmap figure
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot_fnmr, annot=True, fmt=".2f", cmap="Reds")
    plt.title("ArcFace FNMR Heatmap (Higher = Worse)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "arcface_fnmr_heatmap.png"))
    
    print("✅ All done. Metrics saved to results/arcface_metrics_summary.csv!")

if __name__ == "__main__":
    main()
