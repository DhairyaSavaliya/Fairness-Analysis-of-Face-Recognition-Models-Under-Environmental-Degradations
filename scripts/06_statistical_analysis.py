"""
Script: 06_statistical_analysis.py
Performs 3 mathematical analyses on AdaFace fairness metrics:
1. Correlation: brightness/pose vs accuracy per subgroup
2. Regression: polynomial fit of accuracy vs degradation
3. Bias Amplification Factor (BAF): fairness gap widening under stress
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

OUT  = r"E:\Environment\CP\face_recognition_thesis\results"
ada  = pd.read_csv(f"{OUT}\\adaface_metrics_summary.csv")

print("=" * 65)
print("   ADAFACE STATISTICAL FAIRNESS ANALYSIS")
print("=" * 65)


# ================================================================
# 1. CORRELATION ANALYSIS
# ================================================================
print("\n--- 1. CORRELATION: Brightness Level vs Accuracy ---")

b_map = {
    "brightness_0.25": 0.25, "brightness_0.50": 0.50,
    "brightness_1.00": 1.00, "brightness_1.50": 1.50,
    "brightness_2.00": 2.00
}
b_df = ada[ada["condition"].isin(b_map.keys())].copy()
b_df["b_val"] = b_df["condition"].map(b_map)

print(f"\n{'Subgroup':<28} {'Pearson r':>10} {'p-value':>10} {'Interpretation':>20}")
print("-" * 72)
corr_results = []
for sg in sorted(b_df["subgroup"].unique()):
    sub = b_df[b_df["subgroup"] == sg]
    r, p = stats.pearsonr(sub["b_val"], sub["accuracy"])
    interp = "Strong" if abs(r) > 0.7 else ("Moderate" if abs(r) > 0.4 else "Weak")
    direction = "neg" if r < 0 else "pos"
    print(f"{sg:<28} {r:>10.3f} {p:>10.4f} {interp+' '+direction:>20}")
    corr_results.append({"subgroup": sg, "pearson_r": r, "p_value": p})

corr_df = pd.DataFrame(corr_results)

# Plot: Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#e74c3c" if r < -0.4 else ("#3498db" if r > 0.4 else "#95a5a6") for r in corr_df["pearson_r"]]
bars = ax.barh(corr_df["subgroup"], corr_df["pearson_r"], color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=1.2)
ax.axvline(-0.4, color="red", linewidth=1, linestyle="--", alpha=0.5, label="Moderate threshold")
ax.axvline(0.4, color="blue", linewidth=1, linestyle="--", alpha=0.5)
for b, v in zip(bars, corr_df["pearson_r"]):
    ax.text(v + (0.02 if v >= 0 else -0.02), b.get_y() + b.get_height()/2,
            f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.set_xlabel("Pearson Correlation (r)", fontsize=12, fontweight="bold")
ax.set_title("Correlation: Brightness Level vs Accuracy per Subgroup\n(AdaFace)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}\\stat_1_brightness_correlation.png", dpi=200)
plt.close()
print("\n[DONE] stat_1_brightness_correlation.png")

# Pose correlation
print("\n--- CORRELATION: Pose Angle vs Accuracy ---")
p_map = {"pose_0": 0, "pose_15": 15, "pose_30": 30, "pose_45": 45}
p_df = ada[ada["condition"].isin(p_map.keys())].copy()
p_df["p_val"] = p_df["condition"].map(p_map)

pose_corr = []
print(f"\n{'Subgroup':<28} {'Pearson r':>10} {'p-value':>10}")
print("-" * 52)
for sg in sorted(p_df["subgroup"].unique()):
    sub = p_df[p_df["subgroup"] == sg]
    r, p = stats.pearsonr(sub["p_val"], sub["accuracy"])
    print(f"{sg:<28} {r:>10.3f} {p:>10.4f}")
    pose_corr.append({"subgroup": sg, "pearson_r": r, "p_value": p})

pose_corr_df = pd.DataFrame(pose_corr)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#e74c3c" if r < -0.4 else "#95a5a6" for r in pose_corr_df["pearson_r"]]
bars = ax.barh(pose_corr_df["subgroup"], pose_corr_df["pearson_r"], color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=1.2)
ax.axvline(-0.4, color="red", linewidth=1, linestyle="--", alpha=0.5, label="Moderate threshold (-0.4)")
for b, v in zip(bars, pose_corr_df["pearson_r"]):
    ax.text(v - 0.02, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", ha="right", fontsize=9)
ax.set_xlabel("Pearson Correlation (r)", fontsize=12, fontweight="bold")
ax.set_title("Correlation: Pose Angle vs Accuracy per Subgroup\n(AdaFace)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}\\stat_2_pose_correlation.png", dpi=200)
plt.close()
print("[DONE] stat_2_pose_correlation.png")


# ================================================================
# 2. REGRESSION ANALYSIS
# ================================================================
print("\n--- 2. REGRESSION: Accuracy vs Degradation ---")

# Brightness regression (all subgroups combined)
b_avg = b_df.groupby("b_val")["accuracy"].mean().reset_index()
X_b = b_avg["b_val"].values.reshape(-1, 1)
y_b = b_avg["accuracy"].values

# Polynomial degree 2
poly = PolynomialFeatures(degree=2)
X_b_poly = poly.fit_transform(X_b)
from sklearn.linear_model import LinearRegression
reg_b = LinearRegression().fit(X_b_poly, y_b)
y_pred_b = reg_b.predict(X_b_poly)
r2_b = r2_score(y_b, y_pred_b)

c = reg_b.coef_
i = reg_b.intercept_
eq_b = f"Accuracy = {c[2]:.4f}*B² + {c[1]:.4f}*B + {i:.4f}"
print(f"\nBrightness Regression (polynomial degree 2):")
print(f"  Equation: {eq_b}")
print(f"  R² = {r2_b:.4f}")

# Plot brightness regression
x_range = np.linspace(0.25, 2.0, 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_range = reg_b.predict(x_range_poly)

fig, ax = plt.subplots(figsize=(9, 5))
for sg, c_col in zip(sorted(b_df["subgroup"].unique()), plt.cm.tab10(np.linspace(0, 1, 10))):
    sub = b_df[b_df["subgroup"] == sg]
    ax.scatter(sub["b_val"], sub["accuracy"], color=c_col, alpha=0.7, s=40)
ax.plot(x_range.flatten(), y_range, color="black", linewidth=2.5, linestyle="-",
        label=f"Poly Fit: {eq_b}\nR² = {r2_b:.4f}")
ax.set_xlabel("Brightness Level", fontsize=12, fontweight="bold")
ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
ax.set_title("Regression: Accuracy vs Brightness Level (AdaFace)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower center")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}\\stat_3_brightness_regression.png", dpi=200)
plt.close()
print("[DONE] stat_3_brightness_regression.png")

# Pose regression (linear)
p_avg = p_df.groupby("p_val")["accuracy"].mean().reset_index()
X_p = p_avg["p_val"].values.reshape(-1, 1)
y_p = p_avg["accuracy"].values
reg_p = LinearRegression().fit(X_p, y_p)
y_pred_p = reg_p.predict(X_p)
r2_p = r2_score(y_p, y_pred_p)
m = reg_p.coef_[0]
b_int = reg_p.intercept_
eq_p = f"Accuracy = {m:.5f} * Pose + {b_int:.4f}"
print(f"\nPose Regression (linear):")
print(f"  Equation: {eq_p}")
print(f"  R² = {r2_p:.4f}")

x_p_range = np.linspace(0, 45, 100).reshape(-1, 1)
y_p_range = reg_p.predict(x_p_range)

fig, ax = plt.subplots(figsize=(8, 5))
for sg, c_col in zip(sorted(p_df["subgroup"].unique()), plt.cm.tab10(np.linspace(0, 1, 10))):
    sub = p_df[p_df["subgroup"] == sg]
    ax.scatter(sub["p_val"], sub["accuracy"], color=c_col, alpha=0.7, s=40)
ax.plot(x_p_range.flatten(), y_p_range, color="black", linewidth=2.5,
        label=f"Linear Fit: {eq_p}\nR² = {r2_p:.4f}")
ax.set_xlabel("Pose Angle (degrees)", fontsize=12, fontweight="bold")
ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
ax.set_title("Regression: Accuracy vs Pose Angle (AdaFace)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}\\stat_4_pose_regression.png", dpi=200)
plt.close()
print("[DONE] stat_4_pose_regression.png")


# ================================================================
# 3. BIAS AMPLIFICATION FACTOR (BAF)
# ================================================================
print("\n--- 3. BIAS AMPLIFICATION FACTOR (BAF) ---")

# Baseline: brightness_1.00 (normal)
baseline = ada[ada["condition"] == "brightness_1.00"][["subgroup", "accuracy"]].set_index("subgroup")

conditions_to_check = {
    "brightness_0.25": "Dark",
    "brightness_2.00": "Overexposed",
    "pose_30":         "Pose 30",
    "pose_45":         "Pose 45"
}

baf_rows = []
print(f"\n{'Condition':<18} {'Max Acc':>8} {'Min Acc':>8} {'Baseline Gap':>14} {'Degraded Gap':>14} {'BAF':>8}")
print("-" * 75)

for cond, label in conditions_to_check.items():
    cond_df = ada[ada["condition"] == cond][["subgroup", "accuracy"]].set_index("subgroup")
    baseline_gap = baseline["accuracy"].max() - baseline["accuracy"].min()
    degraded_gap = cond_df["accuracy"].max() - cond_df["accuracy"].min()
    baf = (degraded_gap - baseline_gap) / baseline_gap * 100
    worst_sg = cond_df["accuracy"].idxmin()
    best_sg  = cond_df["accuracy"].idxmax()
    print(f"{label:<18} {cond_df['accuracy'].max():>8.3f} {cond_df['accuracy'].min():>8.3f} "
          f"{baseline_gap:>14.3f} {degraded_gap:>14.3f} {baf:>7.1f}%")
    baf_rows.append({"label": label, "condition": cond,
                     "baseline_gap": baseline_gap, "degraded_gap": degraded_gap,
                     "baf_pct": baf, "worst_subgroup": worst_sg, "best_subgroup": best_sg})

baf_df = pd.DataFrame(baf_rows)

# Plot BAF
fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = ["#e74c3c" if b > 0 else "#2ecc71" for b in baf_df["baf_pct"]]
bars = ax.bar(baf_df["label"], baf_df["baf_pct"], color=bar_colors, edgecolor="white", width=0.5)
ax.axhline(0, color="black", linewidth=1.2)
for b, v in zip(bars, baf_df["baf_pct"]):
    ax.text(b.get_x() + b.get_width()/2,
            v + (1 if v >= 0 else -3),
            f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Bias Amplification Factor (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("Degradation Condition", fontsize=12, fontweight="bold")
ax.set_title("Bias Amplification Factor per Condition\n(AdaFace — % increase in fairness gap vs baseline)",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}\\stat_5_bias_amplification_factor.png", dpi=200)
plt.close()
print("\n[DONE] stat_5_bias_amplification_factor.png")

# Worst subgroup summary
print("\n--- WORST SUBGROUP PER CONDITION ---")
for _, r in baf_df.iterrows():
    print(f"  {r['label']:<18}: Most disadvantaged = {r['worst_subgroup']}")

print("\n[ALL DONE] 5 statistical plots saved to results/")
