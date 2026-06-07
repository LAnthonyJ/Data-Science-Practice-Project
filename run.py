"""
Inference script — generates submission_blend_top86.csv using pre-trained checkpoints.

Usage:
    python run.py --data_path ./data [--output submission.csv]

Requirements:
    - Pre-trained checkpoints in ./checkpoints/
    - Test images in {data_path}/test_images/test_images/
    - sample_submission.csv in {data_path}/
"""
import argparse, os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageOps
import cv2

# ── Texture Feature Extractor (21-dim) ──────────────────────────
def compute_texture_features(image):
    """21 handcrafted texture descriptors from a PIL RGB image."""
    image = image.resize((256, 256), Image.LANCZOS)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY).astype(np.float64)

    feats = []
    for ws in [7, 15, 31]:
        m = cv2.blur(gray, (ws, ws))
        ms = cv2.blur(gray ** 2, (ws, ws))
        std = np.sqrt(np.maximum(ms - m ** 2, 0))
        feats.extend([
            float(np.mean(std)), float(np.std(std)),
            float(np.percentile(std, 90)), float(np.percentile(std, 50)),
            float(np.max(std)),
        ])

    feats.append(float(np.std(gray)))
    feats.append(float(np.max(gray) - np.min(gray)))

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.sqrt(gx ** 2 + gy ** 2)
    feats.extend([float(np.mean(gm)), float(np.std(gm))])

    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = mag.shape; ch, cw = h // 2, w // 2; r = min(h, w) // 4
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - ch) ** 2 + (xx - cw) ** 2 > r ** 2
    feats.append(float((mag[mask] ** 2).sum() / ((mag ** 2).sum() + 1e-8)))

    hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
    hist = hist[hist > 0]
    feats.append(float(-np.sum(hist * np.log2(hist))))

    return np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# ── Dataset ─────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, df, img_dir, tex_feats, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tex = tex_feats.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{self.img_dir}/{row['id']}").convert("RGB")
        tex = torch.from_numpy(self.tex[idx])
        if self.transform:
            img = self.transform(img)
        return img, tex, row["id"]


# ── Model ───────────────────────────────────────────────────────
class MeteoriteClassifier(nn.Module):
    """DINOv3-L + Texture hybrid classifier with MLP head."""
    def __init__(self, dino_dim=1024, tex_dim=21, dino_name="vit_large_patch16_dinov3.lvd1689m",
                 dino_weights=None):
        super().__init__()
        kwargs = dict(pretrained=True, num_classes=0, img_size=512)
        # Use local weights if provided (required for offline environments)
        if dino_weights and os.path.exists(dino_weights):
            kwargs["pretrained_cfg_overlay"] = {"file": dino_weights}
        self.backbone = timm.create_model(dino_name, **kwargs)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        total_dim = dino_dim + tex_dim
        self.head = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x, tex):
        with torch.no_grad():
            feat = self.backbone(x)
        return self.head(torch.cat([feat, tex], dim=1))


# ── TTA Transforms ──────────────────────────────────────────────
def build_tta_transforms():
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _make(extra):
        return transforms.Compose([
            transforms.Resize(512), transforms.CenterCrop(512),
            *extra, transforms.ToTensor(), norm,
        ])

    return {
        "orig": _make([]),
        "hflip": _make([transforms.RandomHorizontalFlip(p=1.0)]),
        "vflip": _make([transforms.RandomVerticalFlip(p=1.0)]),
        "hvflip": _make([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
        ]),
    }


# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Meteorite Classification Inference")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Path to pre-trained checkpoints")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Output submission file path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k", type=int, default=86,
                        help="Number of top predictions as positive class")
    parser.add_argument("--dino_weights", type=str, default=None,
                        help="Path to local DINOv3 weights (model.safetensors)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load test data ──
    data_path = args.data_path
    test_img_dir = f"{data_path}/test_images/test_images"
    sample_csv = f"{data_path}/sample_submission.csv"

    test_df = pd.read_csv(sample_csv)
    print(f"Test images: {len(test_df)}")

    # ── Load or compute texture features ──
    tex_cache_path = f"{args.checkpoint_dir}/texture_cache.npz"
    if os.path.exists(tex_cache_path):
        print("Loading cached texture features...")
        test_tex = np.load(tex_cache_path)["test_tex"]
    else:
        print("Computing texture features for test set...")
        feats = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Texture"):
            try:
                p = f"{test_img_dir}/{row['id']}"
                img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                feats.append(compute_texture_features(img))
            except Exception:
                feats.append(np.zeros(21, dtype=np.float32))
        test_tex = np.stack(feats)
        test_tex = np.nan_to_num(test_tex, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Load model ──
    print("Loading DINOv3+Tex classifier...")
    model = MeteoriteClassifier(dino_weights=args.dino_weights).to(device)

    head_path = f"{args.checkpoint_dir}/dinov3_tex_mlp_head.pth"
    if os.path.exists(head_path):
        ckpt = torch.load(head_path, map_location=device)
        model.head.load_state_dict(ckpt["head"])
        print(f"Loaded trained head (epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"Warning: head checkpoint not found at {head_path}, using untrained model.")

    model.eval()

    # ── 4-view TTA inference ──
    print("Running 4-view TTA inference...")
    tta_transforms = build_tta_transforms()
    all_probs = []

    for view_name, tfm in tta_transforms.items():
        ds = TestDataset(test_df, test_img_dir, test_tex, tfm)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        probs = []
        with torch.no_grad():
            for x, tex, _ in tqdm(loader, desc=f"TTA {view_name}", leave=False):
                x, tex = x.to(device), tex.to(device)
                logits = model(x, tex)
                prob = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                probs.append(prob)

        all_probs.append(np.concatenate(probs))

    dino_tex_prob = np.mean(all_probs, axis=0)

    # ── Blend with ConvNeXtV2-Base ──
    conv_path = f"{args.checkpoint_dir}/convnextv2_base_test_prob.csv"
    if os.path.exists(conv_path):
        print("Blending with ConvNeXtV2-Base...")
        conv = pd.read_csv(conv_path)
        df = pd.DataFrame({"id": test_df["id"].tolist(), "prob_dino_tex": dino_tex_prob})
        df = df.merge(conv[["id", "prob"]].rename(columns={"prob": "prob_conv"}), on="id")
        df["prob_blend"] = 0.80 * df["prob_dino_tex"] + 0.15 * df["prob_conv"]
        final_prob = df["prob_blend"].values
    else:
        print("ConvNeXtV2 probs not found, using DINO+Tex only.")
        final_prob = dino_tex_prob

    # ── Generate submission (Top-K) ──
    sorted_idx = np.argsort(final_prob)[::-1]
    top_k_ids = set(test_df.iloc[sorted_idx[:args.top_k]]["id"].tolist())

    submission = pd.read_csv(sample_csv)
    submission["label"] = submission["id"].map(lambda x: 1 if x in top_k_ids else 0).astype(int)
    submission.to_csv(args.output, index=False)

    n_pos = int(submission["label"].sum())
    print(f"\nSaved: {args.output}")
    print(f"  Positives: {n_pos}, Negatives: {len(submission) - n_pos}")
    print(f"  Top-{args.top_k} threshold: {final_prob[sorted_idx[args.top_k - 1]]:.6f}")


if __name__ == "__main__":
    main()
