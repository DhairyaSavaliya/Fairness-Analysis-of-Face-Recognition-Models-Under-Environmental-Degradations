"""
Generate 4 comparison plots: ArcFace vs AdaFace
1) Accuracy across Brightness conditions
2) Accuracy across Pose conditions
3) Accuracy by Gender (normal condition)
4) Accuracy by Race/Skin Color (normal condition)
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"E:\Environment\CP\face_recognition_thesis\results"

ada = pd.read_csv(f"{OUT}\\adaface_metrics_summary.csv")
arc = pd.read_csv(f"{OUT}\\arcface_metrics_summary.csv")

ada["model"] = "AdaFace"
arc["model"] = "ArcFace"
df = pd.concat([ada, arc], ignore_index=True)

colors = {"AdaFace": "#2196F3", "ArcFace": "#FF5722"}

# ============================================================
# PLOT 1: Accuracy across Brightness (both models averaged)
# ============================================================
b_conds = ["brightness_0.25", "brightness_0.50", "brightness_1.00", "brightness_1.50", "brightness_2.00"]
b_labels = ["0.25\n(Dark)", "0.50\n(Dim)", "1.00\n(Normal)", "1.50\n(Bright)", "2.00\n(Over)"]

fig, ax = plt.subplots(figsize=(10, 6))
bdf = df[df["condition"].isin(b_conds)]
means = bdf.groupby(["condition", "model"])["accuracy"].mean().unstack()
means = means.reindex(b_conds)

x = np.arange(len(b_conds))
w = 0.35
ax.bar(x - w/2, means["AdaFace"], w, label="AdaFace", color=colors["AdaFace"], edgecolor="white", linewidth=0.5)
ax.bar(x + w/2, means["ArcFace"], w, label="ArcFace", color=colors["ArcFace"], edgecolor="white", linewidth=0.5)

for i, v in enumerate(means["AdaFace"]):
    ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
for i, v in enumerate(means["ArcFace"]):
    ax.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(b_labels, fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Average Accuracy", fontsize=12, fontweight="bold")
ax.set_xlabel("Brightness Level", fontsize=12, fontweight="bold")
ax.set_title("ArcFace vs AdaFace: Accuracy Across Brightness Conditions", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}\\compare_brightness.png", dpi=200)
plt.close()
print("Saved: compare_brightness.png")

# ============================================================
# PLOT 2: Accuracy across Pose (both models averaged)
# ============================================================
p_conds = ["pose_0", "pose_15", "pose_30", "pose_45"]
p_labels = ["0\n(Frontal)", "15\n(Slight)", "30\n(Moderate)", "45\n(Extreme)"]

fig, ax = plt.subplots(figsize=(9, 6))
pdf = df[df["condition"].isin(p_conds)]
means = pdf.groupby(["condition", "model"])["accuracy"].mean().unstack()
means = means.reindex(p_conds)

x = np.arange(len(p_conds))
ax.bar(x - w/2, means["AdaFace"], w, label="AdaFace", color=colors["AdaFace"], edgecolor="white", linewidth=0.5)
ax.bar(x + w/2, means["ArcFace"], w, label="ArcFace", color=colors["ArcFace"], edgecolor="white", linewidth=0.5)

for i, v in enumerate(means["AdaFace"]):
    ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
for i, v in enumerate(means["ArcFace"]):
    ax.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(p_labels, fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Average Accuracy", fontsize=12, fontweight="bold")
ax.set_xlabel("Pose Angle (degrees)", fontsize=12, fontweight="bold")
ax.set_title("ArcFace vs AdaFace: Accuracy Across Pose Conditions", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}\\compare_pose.png", dpi=200)
plt.close()
print("Saved: compare_pose.png")

# ============================================================
# PLOT 3: Accuracy by Gender (Normal condition only)
# ============================================================
normal = df[df["condition"] == "brightness_1.00"].copy()
normal["gender_clean"] = normal["gender"].str.strip()

fig, ax = plt.subplots(figsize=(7, 6))
gmeans = normal.groupby(["gender_clean", "model"])["accuracy"].mean().unstack()

x = np.arange(len(gmeans))
ax.bar(x - w/2, gmeans["AdaFace"], w, label="AdaFace", color=colors["AdaFace"], edgecolor="white", linewidth=0.5)
ax.bar(x + w/2, gmeans["ArcFace"], w, label="ArcFace", color=colors["ArcFace"], edgecolor="white", linewidth=0.5)

for i, v in enumerate(gmeans["AdaFace"]):
    ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
for i, v in enumerate(gmeans["ArcFace"]):
    ax.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(gmeans.index, fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Average Accuracy", fontsize=12, fontweight="bold")
ax.set_xlabel("Gender", fontsize=12, fontweight="bold")
ax.set_title("ArcFace vs AdaFace: Accuracy by Gender\n(Normal Condition)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}\\compare_gender.png", dpi=200)
plt.close()
print("Saved: compare_gender.png")

# ============================================================
# PLOT 4: Accuracy by Race/Skin Color (Normal condition only)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
rmeans = normal.groupby(["race", "model"])["accuracy"].mean().unstack()
rmeans = rmeans.sort_values("AdaFace", ascending=False)

x = np.arange(len(rmeans))
ax.bar(x - w/2, rmeans["AdaFace"], w, label="AdaFace", color=colors["AdaFace"], edgecolor="white", linewidth=0.5)
ax.bar(x + w/2, rmeans["ArcFace"], w, label="ArcFace", color=colors["ArcFace"], edgecolor="white", linewidth=0.5)

for i, v in enumerate(rmeans["AdaFace"]):
    ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
for i, v in enumerate(rmeans["ArcFace"]):
    ax.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(rmeans.index, fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Average Accuracy", fontsize=12, fontweight="bold")
ax.set_xlabel("Race / Skin Color", fontsize=12, fontweight="bold")
ax.set_title("ArcFace vs AdaFace: Accuracy by Race\n(Normal Condition)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}\\compare_race.png", dpi=200)
plt.close()
print("Saved: compare_race.png")

print("\nAll 4 comparison plots generated!")
