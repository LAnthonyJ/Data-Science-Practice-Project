"""
=============================================================================
Final Training Pipeline — Best Score: 0.85 F1 (submission_blend_top86.csv)
=============================================================================
Architecture: DINOv3-L(frozen) + 21-dim Texture → MLP(1045→512→2) + ConvNeXtV2 blend
Training data: original 5098 + Dataset 2639 + pseudo-labels 194 = 7931 images
Output: submission_blend_top86.csv ONLY

Key config:
  - DINOv3-L: vit_large_patch16_dinov3.lvd1689m, 512x512, frozen
  - Texture: 21-dim via cv2 (multi-scale variance, gradient, FFT, entropy)
  - MLP head: LayerNorm(1045) → Linear(512) → GELU → Dropout(0.3) → Linear(2)
  - Training: LR=4e-5, WD=2e-2, LabelSmoothing=0.1, F1-based selection
  - Aug: RandomPerspective + RandomAffine + RandomErasing(p=0.7)
  - TTA: 4-view Flip (orig/hflip/vflip/hvflip) probability averaging
  - Blend: 0.80 * DINO+Tex + 0.15 * ConvNeXtV2-Base
  - Output: Top-86 submission only
=============================================================================
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json, time, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.backends.cudnn as cudnn, timm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
from pathlib import Path
from PIL import Image, ImageOps
import cv2

# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = Path("/data/数据科学实践project")
OUTPUT_BASE = DATA_ROOT / "runs" / "final_top86"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"

# Model
DINO_MODEL = "vit_large_patch16_dinov3.lvd1689m"
DINO_WEIGHTS = "/data/models/dinov3_large/model.safetensors"
CONVNEXT_PROB = "/data/models/convnextv2_base_22k_384_ema.pt"
IMG_SIZE = 512
TEXTURE_DIM = 21
DINO_DIM = 1024
TOTAL_DIM = DINO_DIM + TEXTURE_DIM  # 1045

# Training
SEED = 35
EPOCHS = 20
BATCH_SIZE = 8
LR = 4e-5
WEIGHT_DECAY = 2e-2
LABEL_SMOOTHING = 0.1
PATIENCE = 8

# Blend
W_DINO_TEX = 0.80
W_CONVNEXT = 0.15
TARGET_K = 86  # Best submission

torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = True

print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Output: {OUTPUT_BASE}")
print(f"Target: Top-{TARGET_K}")


# ============================================================
# 21-DIM TEXTURE FEATURE EXTRACTOR
# ============================================================
def compute_texture_features(image):
    """Extract 21-dim handcrafted texture features from PIL RGB image."""
    image = image.resize((256, 256), Image.LANCZOS)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY).astype(np.float64)
    feats = []

    # Multi-scale local variance (15 dims): 3 windows x 5 statistics
    for ws in [7, 15, 31]:
        mean = cv2.blur(gray, (ws, ws))
        mean_sq = cv2.blur(gray ** 2, (ws, ws))
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
        feats.extend([float(np.mean(std)), float(np.std(std)),
                      float(np.percentile(std, 90)), float(np.percentile(std, 50)),
                      float(np.max(std))])

    # Global contrast (2 dims)
    feats.append(float(np.std(gray)))
    feats.append(float(np.max(gray) - np.min(gray)))

    # Gradient energy (2 dims)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.sqrt(gx ** 2 + gy ** 2)
    feats.extend([float(np.mean(gm)), float(np.std(gm))])

    # FFT high-frequency energy ratio (1 dim)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = mag.shape
    ch, cw = h // 2, w // 2
    r = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask = (y - ch) ** 2 + (x - cw) ** 2 > r ** 2
    feats.append(float((mag[mask] ** 2).sum() / ((mag ** 2).sum() + 1e-8)))

    # Histogram entropy (1 dim)
    hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
    hist = hist[hist > 0]
    feats.append(float(-np.sum(hist * np.log2(hist))))

    return np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# DATASET
# ============================================================
class TexDataset(Dataset):
    def __init__(self, df, texture_feats, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tex = texture_feats.astype(np.float32)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        tex = torch.from_numpy(self.tex[idx])
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, tex, row["id"]
        return img, tex, int(row["label"])


# ============================================================
# TRANSFORMS
# ============================================================
def build_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.6),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.7, scale=(0.02, 0.2), ratio=(0.3, 3.0), value=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_val_transform():
    return transforms.Compose([
        transforms.Resize(512), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_tta_transforms():
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def make(extra):
        return transforms.Compose([
            transforms.Resize(512), transforms.CenterCrop(512),
            *extra, transforms.ToTensor(), norm,
        ])

    return {
        "orig": make([]),
        "hflip": make([transforms.RandomHorizontalFlip(p=1.0)]),
        "vflip": make([transforms.RandomVerticalFlip(p=1.0)]),
        "hvflip": make([transforms.RandomHorizontalFlip(p=1.0),
                         transforms.RandomVerticalFlip(p=1.0)]),
    }


# ============================================================
# MODEL: DINOv3-L + Texture MLP
# ============================================================
class DINOv3TexClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.freeze = True
        self.backbone = timm.create_model(
            DINO_MODEL, pretrained=True, num_classes=0, img_size=IMG_SIZE,
            pretrained_cfg_overlay={"file": DINO_WEIGHTS},
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.head = nn.Sequential(
            nn.LayerNorm(TOTAL_DIM),
            nn.Linear(TOTAL_DIM, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x, tex):
        with torch.no_grad():
            feat = self.backbone(x)
        return self.head(torch.cat([feat, tex], dim=1))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self


# ============================================================
# THRESHOLD SEARCH (F1-based)
# ============================================================
def threshold_search(y_true, prob, start=0.05, end=0.95, step=0.005):
    best = {"threshold": 0.5, "f1": -1.0}
    for th in np.arange(start, end + 1e-9, step):
        pred = (prob >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": float(th), "f1": float(f1),
                "accuracy": float(accuracy_score(y_true, pred)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
            }
    return best


# ============================================================
# BUILD AUGMENTED DATASET
# ============================================================
def build_augmented_dataset():
    train_dir = DATA_ROOT / "train_images" / "train_images"
    if not train_dir.is_dir():
        train_dir = DATA_ROOT / "train_images"

    # 1. Original training data
    train_df = pd.read_csv(DATA_ROOT / "train_labels.csv")
    train_df["img_path"] = train_df["id"].apply(lambda x: str(train_dir / x))
    train_df["source"] = "original"

    # 2. Dataset/ folder augmentation
    ds_dir = DATA_ROOT / "Dataset"
    ds_entries = []
    if ds_dir.is_dir():
        for category in sorted(ds_dir.iterdir()):
            if not category.is_dir():
                continue
            if category.name == "Meteorite":
                for f in category.iterdir():
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        ds_entries.append({
                            "id": f"ds/meteorite/{f.name}", "img_path": str(f),
                            "label": 1, "source": "dataset",
                        })
            else:
                for subcat in category.iterdir():
                    if subcat.is_dir():
                        for f in subcat.iterdir():
                            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                                ds_entries.append({
                                    "id": f"ds/{category.name}/{subcat.name}/{f.name}",
                                    "img_path": str(f), "label": 0, "source": "dataset",
                                })
    ds_df = pd.DataFrame(ds_entries)

    # 3. Pseudo-labels: all test images (top-84 positive, rest negative)
    test_dir = DATA_ROOT / "test_images" / "test_images"
    if not test_dir.is_dir():
        test_dir = DATA_ROOT / "test_images"

    pseudo_pos = {
        "000001.jpg", "000005.jpg", "000009.jpg", "000012.jpg", "000013.jpg",
        "000014.jpg", "000015.jpg", "000017.jpg", "000019.jpg", "000023.jpg",
        "000027.jpg", "000028.jpg", "000029.jpg", "000030.jpg", "000034.jpg",
        "000038.jpg", "000042.jpg", "000047.jpg", "000048.jpg", "000052.jpg",
        "000055.jpg", "000056.jpg", "000058.jpg", "000060.jpg", "000061.jpg",
        "000063.jpg", "000064.jpg", "000067.jpg", "000068.jpg", "000076.jpg",
        "000077.jpg", "000078.jpg", "000080.jpg", "000081.jpg", "000085.jpg",
        "000089.jpg", "000092.jpg", "000093.jpg", "000094.jpg", "000098.jpg",
        "000099.jpg", "000102.jpg", "000103.jpg", "000104.jpg", "000105.jpg",
        "000106.jpg", "000108.jpg", "000109.jpg", "000110.jpg", "000111.jpg",
        "000121.jpg", "000124.jpg", "000125.jpg", "000126.jpg", "000128.jpg",
        "000130.jpg", "000133.jpg", "000134.jpg", "000139.jpg", "000141.jpg",
        "000142.jpg", "000144.jpg", "000146.jpg", "000147.jpg", "000149.jpg",
        "000154.jpg", "000155.jpg", "000157.jpg", "000158.jpg", "000159.jpg",
        "000161.jpg", "000164.jpg", "000166.jpg", "000169.jpg", "000170.jpg",
        "000172.jpg", "000173.jpg", "000175.jpg", "000181.jpg", "000182.jpg",
        "000184.jpg", "000185.jpg", "000188.jpg", "000191.jpg",
    }

    all_ids = pd.read_csv(DATA_ROOT / "sample_submission.csv")["id"].tolist()
    ps_entries = []
    for tid in all_ids:
        p = test_dir / tid
        if p.exists():
            ps_entries.append({
                "id": f"ps/{tid}", "img_path": str(p),
                "label": 1 if tid in pseudo_pos else 0, "source": "pseudo",
            })
    ps_df = pd.DataFrame(ps_entries)

    # Combine
    full_df = pd.concat([train_df, ds_df, ps_df], ignore_index=True)
    full_df = full_df[full_df["img_path"].apply(lambda p: Path(p).exists())].copy()

    print(f"Original: {len(train_df)} ({train_df['label'].value_counts().to_dict()})")
    print(f"Dataset:  {len(ds_df)} ({ds_df['label'].value_counts().to_dict()})")
    print(f"Pseudo:   {len(ps_df)} ({ps_df['label'].value_counts().to_dict()})")
    print(f"Combined: {len(full_df)} ({full_df['label'].value_counts().to_dict()})")

    return full_df


# ============================================================
# PRE-COMPUTE TEXTURE FEATURES
# ============================================================
def precompute_texture(full_df, test_df, test_dir):
    tex_cache = OUTPUT_BASE / "texture_cache.npz"

    if tex_cache.exists():
        print("Loading cached texture features...")
        data = np.load(tex_cache)
        return data["full_tex"], data["test_tex"]

    print("Pre-computing 21-dim texture features...")

    full_feats = []
    for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Train tex"):
        try:
            img = ImageOps.exif_transpose(Image.open(row["img_path"])).convert("RGB")
            full_feats.append(compute_texture_features(img))
        except Exception:
            full_feats.append(np.zeros(TEXTURE_DIM, dtype=np.float32))
    full_tex = np.stack(full_feats)
    full_tex = np.nan_to_num(full_tex, nan=0.0, posinf=0.0, neginf=0.0)

    test_feats = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test tex"):
        try:
            p = test_dir / row["id"]
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            test_feats.append(compute_texture_features(img))
        except Exception:
            test_feats.append(np.zeros(TEXTURE_DIM, dtype=np.float32))
    test_tex = np.stack(test_feats)
    test_tex = np.nan_to_num(test_tex, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = RobustScaler().fit(full_tex)
    full_tex = scaler.transform(full_tex).astype(np.float32)
    test_tex = scaler.transform(test_tex).astype(np.float32)

    np.savez_compressed(tex_cache, full_tex=full_tex, test_tex=test_tex)
    print(f"Texture features cached: train={full_tex.shape}, test={test_tex.shape}")
    return full_tex, test_tex


# ============================================================
# STRATIFIED SPLIT
# ============================================================
def stratified_split(full_df):
    labels = full_df["label"].values
    rng = np.random.default_rng(SEED)
    tr_idx, val_idx = [], []
    for c in range(2):
        ci = np.where(labels == c)[0]
        rng.shuffle(ci)
        nv = max(1, int(len(ci) * 0.1))
        val_idx.extend(ci[:nv])
        tr_idx.extend(ci[nv:])
    tr_idx = np.array(tr_idx)
    val_idx = np.array(val_idx)
    rng.shuffle(tr_idx)
    rng.shuffle(val_idx)
    return tr_idx, val_idx


# ============================================================
# TRAIN ONE EPOCH
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, total_n, total_correct = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for x, tex, y in pbar:
        x, tex, y = x.to(DEVICE), tex.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(x, tex)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


# ============================================================
# VALIDATE
# ============================================================
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    all_probs, all_labels, vloss, vn = [], [], 0.0, 0
    for x, tex, y in loader:
        x, tex, y = x.to(DEVICE), tex.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast("cuda"):
            logits = model(x, tex)
            l = criterion(logits, y)
        prob = torch.softmax(logits.float(), dim=1)[:, 1]
        bs = y.size(0)
        vloss += l.item() * bs
        vn += bs
        all_probs.append(prob.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels), vloss / max(vn, 1)


# ============================================================
# 4-VIEW TTA INFERENCE
# ============================================================
@torch.no_grad()
def predict_tta(model, test_df, test_tex, tta_transforms):
    view_probs = {}
    ids_ref = None
    for view_name, tfm in tta_transforms.items():
        ds = TexDataset(
            test_df.assign(img_path=test_df["id"].apply(
                lambda x: str(DATA_ROOT / "test_images" / "test_images" / x))),
            test_tex, tfm, is_test=True,
        )
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        ids, pl = [], []
        for x, tex, img_ids in tqdm(loader, desc=f"TTA {view_name}", leave=False):
            x, tex = x.to(DEVICE), tex.to(DEVICE)
            with torch.amp.autocast("cuda"):
                logits = model(x, tex)
            prob = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            ids.extend(list(img_ids))
            pl.append(prob)
        pa = np.concatenate(pl)
        if ids_ref is None:
            ids_ref = ids
        view_probs[view_name] = pa
    mean_prob = np.mean(list(view_probs.values()), axis=0)
    return ids_ref, mean_prob, view_probs


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()

    # ---- Load data ----
    print("\n" + "=" * 60)
    print("Step 1: Build augmented dataset")
    print("=" * 60)
    full_df = build_augmented_dataset()

    test_dir = DATA_ROOT / "test_images" / "test_images"
    if not test_dir.is_dir():
        test_dir = DATA_ROOT / "test_images"
    test_df = pd.read_csv(DATA_ROOT / "sample_submission.csv")

    # ---- Texture features ----
    print("\n" + "=" * 60)
    print("Step 2: Pre-compute texture features")
    print("=" * 60)
    full_tex, test_tex = precompute_texture(full_df, test_df, test_dir)

    # ---- Split ----
    print("\n" + "=" * 60)
    print("Step 3: Stratified split")
    print("=" * 60)
    tr_idx, val_idx = stratified_split(full_df)
    tr_df = full_df.iloc[tr_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)
    tr_tex = full_tex[tr_idx]
    val_tex = full_tex[val_idx]
    print(f"Train: {len(tr_df)}, Val: {len(val_df)}")
    print(f"Val dist: {val_df['label'].value_counts().to_dict()}")

    # ---- DataLoaders ----
    train_ds = TexDataset(tr_df, tr_tex, build_train_transform())
    val_ds = TexDataset(val_df, val_tex, build_val_transform())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ---- Model ----
    print("\n" + "=" * 60)
    print("Step 4: Train DINOv3 + Texture MLP")
    print("=" * 60)
    model = DINOv3TexClassifier().to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable}")
    print(f"Input dim: {TOTAL_DIM} (DINO={DINO_DIM} + Texture={TEXTURE_DIM})")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    best_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    rows = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        scheduler.step()
        val_probs, val_labels, val_loss = validate(model, val_loader, criterion)
        th_best = threshold_search(val_labels, val_probs)
        pred_05 = (val_probs >= 0.5).astype(int)

        row = {
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,
            "val_f1_05": f1_score(val_labels, pred_05, zero_division=0),
            "val_best_f1": th_best["f1"], "val_best_th": th_best["threshold"],
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_BASE / "metrics.csv", index=False)

        print(f"  E{epoch:2d}: train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_f1@0.5={row['val_f1_05']:.4f} best_f1={th_best['f1']:.4f} th={th_best['threshold']:.3f}")

        hs = model.head.state_dict()
        torch.save({"head": hs, "epoch": epoch, "metrics": th_best}, OUTPUT_BASE / "last_head.pth")
        if th_best["f1"] > best_f1:
            best_f1 = th_best["f1"]
            best_epoch = epoch
            no_improve = 0
            torch.save({"head": hs, "epoch": epoch, "metrics": th_best}, OUTPUT_BASE / "best_head.pth")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ---- Load best head ----
    ckpt = torch.load(OUTPUT_BASE / "best_head.pth", map_location=DEVICE)
    model.head.load_state_dict(ckpt["head"])
    best_th = float(ckpt["metrics"]["threshold"])
    model.eval()
    print(f"\nBest: epoch={best_epoch}, val_f1={best_f1:.4f}, threshold={best_th:.4f}")

    # ---- TTA Inference ----
    print("\n" + "=" * 60)
    print("Step 5: 4-view TTA inference")
    print("=" * 60)
    tta_transforms = build_tta_transforms()
    test_ids, test_prob, _ = predict_tta(model, test_df, test_tex, tta_transforms)

    clean_ids = [Path(p).name for p in test_ids]
    out = pd.DataFrame({"id": clean_ids, "prob": test_prob})
    out.to_csv(OUTPUT_BASE / "test_prob_dino_tex.csv", index=False)
    print(f"DINO+Tex prob: mean={test_prob.mean():.4f}, range=[{test_prob.min():.4f}, {test_prob.max():.4f}]")

    # ---- Blend with ConvNeXtV2 ----
    print("\n" + "=" * 60)
    print("Step 6: Blend with ConvNeXtV2-Base")
    print("=" * 60)

    # Use ConvNeXtV2 probabilities from the fixed training
    conv_path = DATA_ROOT / "runs" / "v13_proper" / "convnextv2_base_frozen_head_v2" / "test_prob.csv"
    if not conv_path.exists():
        # Fallback: train ConvNeXtV2 on the fly (requires pretrained weights on server)
        print("ConvNeXtV2 probs not found. Run convnextv2_base training first.")
        print(f"Expected at: {conv_path}")
        return

    conv = pd.read_csv(conv_path)
    df = out[["id", "prob"]].rename(columns={"prob": "prob_dino_tex"})
    df = df.merge(conv[["id", "prob"]].rename(columns={"prob": "prob_conv"}), on="id")
    df["prob_blend"] = W_DINO_TEX * df["prob_dino_tex"] + W_CONVNEXT * df["prob_conv"]
    df = df.sort_values("prob_blend", ascending=False).reset_index(drop=True)

    df.to_csv(OUTPUT_BASE / "test_prob_blend.csv", index=False)
    print(f"Blend prob: mean={df['prob_blend'].mean():.4f}, "
          f"range=[{df['prob_blend'].min():.4f}, {df['prob_blend'].max():.4f}]")

    # ---- Generate submission (Top-86 ONLY) ----
    print("\n" + "=" * 60)
    print(f"Step 7: Generate submission_blend_top{TARGET_K}.csv")
    print("=" * 60)

    top_ids = set(df.head(TARGET_K)["id"].tolist())
    template = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    submission = template.copy()
    submission["label"] = submission["id"].map(lambda x: 1 if x in top_ids else 0).astype(int)

    output_path = OUTPUT_BASE / f"submission_blend_top{TARGET_K}.csv"
    submission.to_csv(output_path, index=False)

    n_pos = int(submission["label"].sum())
    print(f"Saved: {output_path}")
    print(f"Positives: {n_pos}, Negatives: {len(submission) - n_pos}")

    # ---- Summary ----
    elapsed = time.time() - t_start
    summary = {
        "method": "DINOv3-L + 21-dim Texture MLP + ConvNeXtV2-Base blend",
        "architecture": f"DINO({DINO_DIM}) + Texture({TEXTURE_DIM}) = {TOTAL_DIM} -> MLP({TOTAL_DIM}->512->2)",
        "training_samples": len(full_df),
        "train_distribution": full_df["label"].value_counts().to_dict(),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "best_val_threshold": best_th,
        "blend_weights": {"dino_tex": W_DINO_TEX, "convnextv2_base": W_CONVNEXT},
        "tta_views": 4,
        "target_k": TARGET_K,
        "n_positives": n_pos,
        "elapsed_minutes": elapsed / 60,
    }
    with open(OUTPUT_BASE / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Done! Elapsed: {elapsed / 60:.1f} min")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
