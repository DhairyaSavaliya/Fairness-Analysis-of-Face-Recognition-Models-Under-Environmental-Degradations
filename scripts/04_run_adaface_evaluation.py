"""
Script: 04_run_adaface_evaluation.py
Model : AdaFace IR-50 (CVPR 2022 - State of the Art)
Device: CPU (PyTorch)
Output: results/adaface_raw_scores.csv
        results/adaface_metrics_summary.csv
        results/adaface_plots/ (4 plots)
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE         = r"e:\Environment\CP\face_recognition_thesis"
ADAFACE_DIR  = os.path.join(BASE, "scripts", "AdaFace")
CKPT_PATH    = os.path.join(ADAFACE_DIR, "pretrained", "adaface_ir50_ms1mv2.ckpt")
PAIRS_CSV    = os.path.join(BASE, "data", "lfw_test_pairs.csv")
DEGRADED_DIR = os.path.join(BASE, "data", "degraded")
RESULTS_DIR  = os.path.join(BASE, "results")
PLOTS_DIR    = os.path.join(RESULTS_DIR, "adaface_plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

# Add AdaFace to path so we can import net.py
sys.path.insert(0, ADAFACE_DIR)

THRESHOLD = 0.3   # AdaFace cosine threshold for same person

CONDITIONS = [
    "brightness_0.25", "brightness_0.50", "brightness_1.00",
    "brightness_1.50", "brightness_2.00",
    "pose_0", "pose_15", "pose_30", "pose_45",
]


# ─── Model Loading ────────────────────────────────────────────────────────────
def load_adaface_model():
    """Load AdaFace IR-50 model with pretrained weights."""
    import net
    print("Building AdaFace IR-50 architecture...", flush=True)
    model = net.build_model("ir_50")

    print(f"Loading checkpoint from {CKPT_PATH}...", flush=True)
    statedict = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)["state_dict"]
    # Strip the 'model.' prefix from keys
    model_statedict = {k[6:]: v for k, v in statedict.items() if k.startswith("model.")}
    model.load_state_dict(model_statedict)
    model.eval()
    print("AdaFace model loaded successfully on CPU!", flush=True)
    return model


# ─── Image Preprocessing ─────────────────────────────────────────────────────
def preprocess_image(img_path):
    """
    Load image with OpenCV (BGR), resize to 112x112,
    normalize to [-1, 1] range as AdaFace expects.
    Returns a (1, 3, 112, 112) torch tensor or None on failure.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (112, 112))
    # BGR format, normalize: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
    img = ((img / 255.0) - 0.5) / 0.5
    # HWC -> CHW, add batch dim
    tensor = torch.tensor(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


# ─── Embedding Extraction ────────────────────────────────────────────────────
@torch.no_grad()
def get_embedding(model, img_path):
    """Get 512-dim normalized embedding from AdaFace."""
    tensor = preprocess_image(img_path)
    if tensor is None:
        return None
    feature, _ = model(tensor)
    return feature[0].numpy()  # Already L2-normalized by the model


def cosine_sim(a, b):
    return float(np.dot(a, b))  # Already unit-norm from AdaFace


# ─── Path Remapping ──────────────────────────────────────────────────────────
def remap(original_path, condition):
    parts = original_path.replace("/", os.sep).split(os.sep)
    person, fname = parts[-2], parts[-1]
    return os.path.join(DEGRADED_DIR, condition, person, fname)


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(df):
    pred  = (df["similarity"] >= THRESHOLD).astype(int)
    label = df["label"]
    tp = int(((pred == 1) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    n  = len(df)
    return {
        "accuracy": round((tp + tn) / n, 4) if n > 0 else 0,
        "fmr":      round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0,
        "fnmr":     round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0,
        "n_pairs":  n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Score all pairs across all conditions
# ═══════════════════════════════════════════════════════════════════════════════
def phase1_score(model):
    pairs_df = pd.read_csv(PAIRS_CSV)
    # Use 50 pairs per subgroup for speed (500 total per condition)
    pairs_df = pairs_df.groupby("subgroup", group_keys=False).apply(
        lambda x: x.sample(min(len(x), 50), random_state=42)
    ).reset_index(drop=True)
    
    print(f"Test pairs: {len(pairs_df)} per condition x {len(CONDITIONS)} conditions = {len(pairs_df)*len(CONDITIONS)} total", flush=True)
    print(f"Subgroups: {sorted(pairs_df['subgroup'].unique())}", flush=True)

    raw_path = os.path.join(RESULTS_DIR, "adaface_raw_scores.csv")

    # Resume support
    done_conds = set()
    all_rows = []
    if os.path.exists(raw_path):
        done_df = pd.read_csv(raw_path)
        done_conds = set(done_df["condition"].unique())
        all_rows = done_df.to_dict("records")
        print(f"[RESUME] Already scored: {sorted(done_conds)}", flush=True)

    todo = [c for c in CONDITIONS if c not in done_conds]
    if not todo:
        print("All conditions already scored!", flush=True)
        return pd.read_csv(raw_path)

    for cond in todo:
        t0 = time.time()
        print(f"\n▶ Scoring: {cond}", flush=True)
        errors = 0
        rows = []

        for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), ncols=70, desc=cond):
            p1 = remap(row["img1_path"], cond)
            p2 = remap(row["img2_path"], cond)

            e1 = get_embedding(model, p1)
            e2 = get_embedding(model, p2)

            if e1 is None or e2 is None:
                errors += 1
                sim = -1.0
            else:
                sim = cosine_sim(e1, e2)

            rows.append({
                "condition": cond,
                "subgroup":  row["subgroup"],
                "label":     int(row["label"]),
                "similarity": round(sim, 5),
            })

        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(raw_path, index=False)
        elapsed = time.time() - t0
        valid = sum(1 for r in rows if r["similarity"] >= 0)
        print(f"  Done in {elapsed:.0f}s | Errors: {errors} | Valid: {valid}/{len(pairs_df)}", flush=True)

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(raw_path, index=False)
    print(f"\nAll raw scores saved to {raw_path}", flush=True)
    return raw_df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Compute metrics per subgroup × condition
# ═══════════════════════════════════════════════════════════════════════════════
def phase2_metrics(raw_df):
    valid = raw_df[raw_df["similarity"] >= 0].copy()
    rows = []

    for cond in CONDITIONS:
        cdf = valid[valid["condition"] == cond]
        for sg in sorted(cdf["subgroup"].unique()):
            sgdf = cdf[cdf["subgroup"] == sg]
            m = compute_metrics(sgdf)
            parts = sg.rsplit("_", 1)
            rows.append({
                "condition": cond, "subgroup": sg,
                "race": parts[0] if len(parts) == 2 else sg,
                "gender": parts[1] if len(parts) == 2 else "?",
                **m,
            })

    mdf = pd.DataFrame(rows)
    metrics_path = os.path.join(RESULTS_DIR, "adaface_metrics_summary.csv")
    mdf.to_csv(metrics_path, index=False)

    # Print readable tables
    pivot_acc = mdf.pivot_table(index="subgroup", columns="condition", values="accuracy", aggfunc="first")
    ordered = [c for c in CONDITIONS if c in pivot_acc.columns]
    print("\n" + "=" * 90, flush=True)
    print("ADAFACE ACCURACY TABLE (rows=subgroup, cols=condition)", flush=True)
    print("=" * 90, flush=True)
    print(pivot_acc[ordered].to_string(), flush=True)

    pivot_fnmr = mdf.pivot_table(index="subgroup", columns="condition", values="fnmr", aggfunc="first")
    print("\n" + "=" * 90, flush=True)
    print("ADAFACE FNMR TABLE (False Non-Match Rate — higher = worse)", flush=True)
    print("=" * 90, flush=True)
    print(pivot_fnmr[ordered].to_string(), flush=True)

    print(f"\nMetrics saved to {metrics_path}", flush=True)
    return mdf


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Generate plots
# ═══════════════════════════════════════════════════════════════════════════════
def phase3_plots(mdf):
    b_conds = [c for c in CONDITIONS if "brightness" in c]
    p_conds = [c for c in CONDITIONS if "pose" in c]

    # ── Plot 1: Accuracy by Brightness ────────────────────────────────────────
    bdf = mdf[mdf["condition"].isin(b_conds)]
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=bdf, x="condition", y="accuracy", hue="subgroup", ax=ax)
    ax.set_title("AdaFace Accuracy by Demographic Subgroup\nAcross Brightness Conditions",
                 fontweight="bold", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Brightness Level")
    ax.set_ylabel("Accuracy")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "1_accuracy_brightness.png"), dpi=150)
    plt.close()
    print("  Saved: 1_accuracy_brightness.png", flush=True)

    # ── Plot 2: Accuracy by Pose ──────────────────────────────────────────────
    pdf = mdf[mdf["condition"].isin(p_conds)]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=pdf, x="condition", y="accuracy", hue="subgroup", ax=ax)
    ax.set_title("AdaFace Accuracy by Demographic Subgroup\nAcross Pose Conditions",
                 fontweight="bold", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Pose Angle")
    ax.set_ylabel("Accuracy")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "2_accuracy_pose.png"), dpi=150)
    plt.close()
    print("  Saved: 2_accuracy_pose.png", flush=True)

    # ── Plot 3: FNMR Heatmap ─────────────────────────────────────────────────
    pivot = mdf.pivot_table(index="subgroup", columns="condition", values="fnmr", aggfunc="first")
    ordered = [c for c in CONDITIONS if c in pivot.columns]
    pivot = pivot[ordered]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, vmin=0, vmax=1, ax=ax)
    ax.set_title("AdaFace FNMR Heatmap\nDemographic Subgroup × Environmental Condition (↑ = worse bias)",
                 fontweight="bold", fontsize=13)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "3_fnmr_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved: 3_fnmr_heatmap.png", flush=True)

    # ── Plot 4: Bias Amplification ────────────────────────────────────────────
    base = mdf[mdf["condition"] == "brightness_1.00"][["subgroup", "accuracy"]]\
              .rename(columns={"accuracy": "base_acc"})
    aug = mdf.merge(base, on="subgroup")
    aug["acc_drop"] = aug["base_acc"] - aug["accuracy"]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=aug, x="condition", y="acc_drop", hue="subgroup", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("AdaFace Bias Amplification: Accuracy Drop vs. Clean Baseline\nper Demographic Subgroup (↑ = more bias introduced)",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Accuracy Drop")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "4_bias_amplification.png"), dpi=150)
    plt.close()
    print("  Saved: 4_bias_amplification.png", flush=True)

    print(f"\nAll plots saved to: {PLOTS_DIR}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()

    print("=" * 60, flush=True)
    print("AdaFace IR-50 Evaluation Pipeline", flush=True)
    print("=" * 60, flush=True)

    model = load_adaface_model()

    print("\n" + "=" * 60, flush=True)
    print("PHASE 1 — Scoring all pairs across 9 conditions", flush=True)
    print("=" * 60, flush=True)
    raw_df = phase1_score(model)

    print("\n" + "=" * 60, flush=True)
    print("PHASE 2 — Computing fairness metrics", flush=True)
    print("=" * 60, flush=True)
    mdf = phase2_metrics(raw_df)

    print("\n" + "=" * 60, flush=True)
    print("PHASE 3 — Generating plots", flush=True)
    print("=" * 60, flush=True)
    phase3_plots(mdf)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}", flush=True)
    print(f"✅ ALL DONE in {elapsed / 60:.1f} minutes", flush=True)
    print(f"   Raw scores : {RESULTS_DIR}\\adaface_raw_scores.csv", flush=True)
    print(f"   Metrics    : {RESULTS_DIR}\\adaface_metrics_summary.csv", flush=True)
    print(f"   Plots      : {PLOTS_DIR}", flush=True)
    print(f"{'=' * 60}", flush=True)
