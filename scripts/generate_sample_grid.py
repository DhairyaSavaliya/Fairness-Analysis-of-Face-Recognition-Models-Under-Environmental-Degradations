"""
Generate a poster-ready sample grid showing one face under all conditions:
Row 1: Brightness levels (0.25, 0.50, 1.00, 1.50, 2.00)
Row 2: Pose rotations (0, 15, 30, 45)
"""
import os, cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from PIL import Image

LFW = r"E:\Environment\CP\face_recognition_thesis\data\lfw\lfw-deepfunneled\lfw-deepfunneled"
OUT = r"E:\Environment\CP\face_recognition_thesis\results"

mtcnn = MTCNN(image_size=112, margin=20, keep_all=False, post_process=False, device="cpu")

def crop_face(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = mtcnn(Image.fromarray(rgb))
    if t is not None:
        arr = t.permute(1, 2, 0).numpy().astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    s = min(h, w)
    cx, cy = w // 2, h // 2
    crop = bgr[cy - s//2:cy + s//2, cx - s//2:cx + s//2]
    return cv2.resize(crop, (112, 112))

# Pick a clear frontal face
identity = "Colin_Powell"
img_path = os.path.join(LFW, identity, os.listdir(os.path.join(LFW, identity))[0])
raw = cv2.imread(img_path)
face = crop_face(raw)

# --- Row 1: Brightness ---
brightness_vals = [0.25, 0.50, 1.00, 1.50, 2.00]
brightness_labels = ["0.25 (Dark)", "0.50 (Dim)", "1.00 (Normal)", "1.50 (Bright)", "2.00 (Overexposed)"]
bright_imgs = []
for b in brightness_vals:
    img = np.clip(face.astype(np.float32) * b, 0, 255).astype(np.uint8)
    bright_imgs.append(img)

# --- Row 2: Pose ---
pose_vals = [0, 15, 30, 45]
pose_labels = ["0 (Frontal)", "15 (Slight)", "30 (Moderate)", "45 (Extreme)"]
pose_imgs = []
h, w = face.shape[:2]
for p in pose_vals:
    M = cv2.getRotationMatrix2D((w//2, h//2), p, 1.0)
    img = cv2.warpAffine(face, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    pose_imgs.append(img)

# --- Plot ---
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle("Sample Face Under Environmental Degradations", fontsize=18, fontweight="bold", y=0.98)

# Row 1
for i, (img, lab) in enumerate(zip(bright_imgs, brightness_labels)):
    axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"Brightness {lab}", fontsize=9, fontweight="bold")
    axes[0, i].axis("off")

# Row 2 (4 images, hide 5th cell)
for i, (img, lab) in enumerate(zip(pose_imgs, pose_labels)):
    axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f"Pose {lab}", fontsize=9, fontweight="bold")
    axes[1, i].axis("off")
axes[1, 4].axis("off")  # hide empty cell

# Row labels
fig.text(0.02, 0.72, "Brightness\nVariation", ha="center", va="center", fontsize=12, fontweight="bold", rotation=90)
fig.text(0.02, 0.28, "Pose\nRotation", ha="center", va="center", fontsize=12, fontweight="bold", rotation=90)

plt.tight_layout(rect=[0.04, 0, 1, 0.95])
out_path = os.path.join(OUT, "sample_degradation_grid.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
