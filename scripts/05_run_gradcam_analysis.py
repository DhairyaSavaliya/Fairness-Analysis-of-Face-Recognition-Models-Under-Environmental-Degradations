"""
Script: 05_run_gradcam_analysis.py  (v5 — Activation-based Attention Maps)

Instead of gradient-based GradCAM (which is unreliable for face embeddings
because the backward target choice critically affects the result), this
script visualises ACTIVATION MAGNITUDE at the last convolutional layer.

This directly answers: "Where does the network fire most strongly?"
  - Clean face → activations concentrate on eyes/nose/mouth
  - Degraded face → activations scatter or shift to edges/noise

No gradients, no backward pass, no target selection problem.
"""

import os, sys, glob, cv2, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from facenet_pytorch import MTCNN
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "AdaFace"))
import net

CKPT = os.path.join(os.path.dirname(__file__),
                     "AdaFace", "pretrained", "adaface_ir50_ms1mv2.ckpt")
LFW  = r"E:\Environment\CP\face_recognition_thesis\data\lfw\lfw-deepfunneled\lfw-deepfunneled"
OUT  = r"E:\Environment\CP\face_recognition_thesis\results\gradcam_plots"
os.makedirs(OUT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MTCNN
# ═══════════════════════════════════════════════════════════════════════════════
_mtcnn = MTCNN(image_size=112, margin=20, keep_all=False,
               post_process=False, device="cpu")

def crop_face(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t = _mtcnn(pil)
    if t is not None:
        arr = t.permute(1, 2, 0).numpy().astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    s = min(h, w)
    cx, cy = w // 2, h // 2
    crop = bgr[cy - s//2:cy + s//2, cx - s//2:cx + s//2]
    return cv2.resize(crop, (112, 112))


# ═══════════════════════════════════════════════════════════════════════════════
#  Activation-based Spatial Attention Map
# ═══════════════════════════════════════════════════════════════════════════════
class ActivationAttention:
    """
    Hooks a convolutional layer and computes a spatial attention map
    based on the L2 norm of activations across channels at each position.

    This is gradient-free — it simply shows WHERE the network fires.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        target_layer.register_forward_hook(self._hook)

    def _hook(self, m, inp, out):
        self.acts = out.detach()

    @torch.no_grad()
    def __call__(self, face_tensor):
        """
        face_tensor: (1, 3, 112, 112) normalised to [-1, 1]
        Returns: (112, 112) numpy heatmap in [0, 1]
        """
        self.model(face_tensor)
        acts = self.acts                                # (1, C, H, W)

        # L2 norm across channels at each spatial position
        # This gives a (H, W) map of "how strongly the network fires here"
        spatial = torch.norm(acts, p=2, dim=1).squeeze()  # (H, W)
        cam = spatial.cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Upsample + light Gaussian smooth (low sigma to preserve scatter)
        cam = cv2.resize(cam, (112, 112), interpolation=cv2.INTER_CUBIC)
        cam = gaussian_filter(cam, sigma=1.5)

        # Re-normalise
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def to_tensor(bgr112):
    t = bgr112.astype(np.float32) / 255.0
    t = (t - 0.5) / 0.5
    return torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0)


def make_overlay(bgr, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1.0 - alpha, heatmap, alpha, 0)


def save_grid(title, c_bgr, c_ov, d_bgr, d_ov, path):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    for (lab, img), a in zip([
        ("Baseline (Normal)", c_bgr),
        ("Attention Map — Baseline", c_ov),
        ("Degraded Condition", d_bgr),
        ("Attention Map — Degraded", d_ov),
    ], ax.flat):
        a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        a.set_title(lab, fontsize=13)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [DONE] {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = net.build_model("ir_50")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"AdaFace on {device} [DONE]")

    # Hook body[20] — last block of the 14×14 stage (256ch).
    # 14×14 = 196 spatial cells → fine enough to show scatter clearly.
    attn = ActivationAttention(model, model.body[20])
    print("Activation hook: body[20]  256ch x 14x14  (sigma=1.5)\n")

    cases = [
        dict(label    = "Indian Man — Overexposure (Brightness 2.0)",
             identity = "Binyamin_Ben-Eliezer",
             deg_type = "brightness", deg_val = 2.0,
             out      = "indian_man_overexposure.png"),

        dict(label    = "White Man — Underexposure (Brightness 0.25)",
             identity = "Colin_Powell",
             deg_type = "brightness", deg_val = 0.25,
             out      = "white_man_underexposure.png"),

        dict(label    = "Asian Woman — Pose Collapse (45° Rotation)",
             identity = "Jennifer_Lopez",
             deg_type = "pose", deg_val = 45,
             out      = "asian_woman_pose_collapse.png"),

        dict(label    = "Black Woman — Underexposure (Brightness 0.25)",
             identity = "Serena_Williams",
             deg_type = "brightness", deg_val = 0.25,
             out      = "black_woman_underexposure.png"),

        dict(label    = "Asian Man — Pose Collapse (45° Rotation)",
             identity = "Junichiro_Koizumi",
             deg_type = "pose", deg_val = 45,
             out      = "asian_man_pose_collapse.png"),
    ]

    for c in cases:
        print(f"-> {c['label']}")
        orig_dir = os.path.join(LFW, c["identity"])
        imgs = sorted(glob.glob(os.path.join(orig_dir, "*.jpg")))
        if not imgs:
            print("   ⚠ missing"); continue

        raw = cv2.imread(imgs[0])
        face_clean = crop_face(raw)

        if c["deg_type"] == "brightness":
            face_deg = np.clip(
                face_clean.astype(np.float32) * c["deg_val"], 0, 255
            ).astype(np.uint8)
        else:
            h, w = face_clean.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), c["deg_val"], 1.0)
            face_deg = cv2.warpAffine(
                face_clean, M, (w, h),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        tc = to_tensor(face_clean).to(device)
        td = to_tensor(face_deg).to(device)

        cam_c = attn(tc)
        cam_d = attn(td)

        print(f"   Clean — mean={cam_c.mean():.3f}  max={cam_c.max():.3f}")
        print(f"   Deg   — mean={cam_d.mean():.3f}  max={cam_d.max():.3f}")

        ov_c = make_overlay(face_clean, cam_c)
        ov_d = make_overlay(face_deg,   cam_d)

        save_grid(c["label"], face_clean, ov_c, face_deg, ov_d,
                  os.path.join(OUT, c["out"]))

    print("\n[DONE]")


if __name__ == "__main__":
    main()
