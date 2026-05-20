import os, sys, glob, cv2, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from facenet_pytorch import MTCNN
from PIL import Image
from deepface import DeepFace

LFW  = r"E:\Environment\CP\face_recognition_thesis\data\lfw\lfw-deepfunneled\lfw-deepfunneled"
OUT  = r"E:\Environment\CP\face_recognition_thesis\results\gradcam_plots\arcface"
os.makedirs(OUT, exist_ok=True)

_mtcnn = MTCNN(image_size=112, margin=20, keep_all=False, post_process=False, device="cpu")

def crop_face(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = _mtcnn(Image.fromarray(rgb))
    if t is not None:
        arr = t.permute(1, 2, 0).numpy().astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    s = min(h, w)
    cx, cy = w // 2, h // 2
    crop = bgr[cy - s//2:cy + s//2, cx - s//2:cx + s//2]
    return cv2.resize(crop, (112, 112))

class ArcFaceAttention:
    def __init__(self):
        print("Loading ArcFace...", flush=True)
        cw = DeepFace.build_model("ArcFace")
        self.base_model = cw.model
        target = self.base_model.get_layer("conv5_block3_add")
        self.extractor = tf.keras.Model(inputs=self.base_model.inputs, outputs=target.output)

    def __call__(self, face_bgr):
        img = face_bgr.astype(np.float32) / 255.0
        img_exp = np.expand_dims(img, axis=0) # (1, 112, 112, 3)
        acts = self.extractor.predict(img_exp, verbose=0)[0]
        spatial = np.linalg.norm(acts, axis=-1)
        
        if spatial.max() > spatial.min():
            spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min())
            
        spatial = cv2.resize(spatial, (112, 112), interpolation=cv2.INTER_CUBIC)
        spatial = gaussian_filter(spatial, sigma=1.5)
        
        if spatial.max() > spatial.min():
            spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min())
            
        return spatial

def make_overlay(bgr, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1.0 - alpha, heatmap, alpha, 0)

def save_grid(title, c_bgr, c_ov, d_bgr, d_ov, path):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    for (lab, img), a in zip([
        ("Baseline (Normal)", c_bgr), ("Attention Map \u2014 Baseline", c_ov),
        ("Degraded Condition", d_bgr), ("Attention Map \u2014 Degraded", d_ov),
    ], ax.flat):
        a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        a.set_title(lab, fontsize=13)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [DONE] {os.path.basename(path)}", flush=True)

def main():
    attn = ArcFaceAttention()
    cases = [
        dict(label="ArcFace: Indian Man \u2014 Overexposure (2.0)", identity="Binyamin_Ben-Eliezer", type="brightness", val=2.0, out="1_arcface_indian_man.png"),
        dict(label="ArcFace: White Man \u2014 Underexposure (0.25)", identity="Colin_Powell", type="brightness", val=0.25, out="2_arcface_white_man.png"),
        dict(label="ArcFace: Asian Woman \u2014 Pose Collapse (45\u00b0)", identity="Jennifer_Lopez", type="pose", val=45, out="3_arcface_asian_woman.png"),
        dict(label="ArcFace: Black Woman \u2014 Underexposure (0.25)", identity="Serena_Williams", type="brightness", val=0.25, out="4_arcface_black_woman.png"),
        dict(label="ArcFace: Asian Man \u2014 Pose Collapse (45\u00b0)", identity="Junichiro_Koizumi", type="pose", val=45, out="5_arcface_asian_man.png"),
    ]

    for c in cases:
        print(f"-> {c['label']}", flush=True)
        orig_dir = os.path.join(LFW, c["identity"])
        imgs = sorted(glob.glob(os.path.join(orig_dir, "*.jpg")))
        if not imgs: continue
        raw = cv2.imread(imgs[0])
        f_c = crop_face(raw)
        
        if c["type"] == "brightness":
            f_d = np.clip(f_c.astype(np.float32) * c["val"], 0, 255).astype(np.uint8)
        else:
            h, w = f_c.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), c["val"], 1.0)
            f_d = cv2.warpAffine(f_c, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            
        cam_c = attn(f_c)
        cam_d = attn(f_d)
        
        ov_c = make_overlay(f_c, cam_c)
        ov_d = make_overlay(f_d, cam_d)
        
        save_grid(c["label"], f_c, ov_c, f_d, ov_d, os.path.join(OUT, c["out"]))

if __name__ == "__main__":
    main()
