# 陨石图像识别 — 数据科学实践期末项目

**第11组：张航源 | 陈杭楠 | 程卓 | 林敬知 | 潘昊阳**

**Best Result: F1 = 0.86046** (Stage 2 Private Test Set)

> GitHub: https://github.com/LAnthonyJ/Data-Science-Practice-Project
> 模型与数据: https://pan.baidu.com/s/11ZQfDjY0uJAMb_7RTkR_kQ?pwd=dn2w (提取码: dn2w)

## Overview

This project implements a hybrid deep learning pipeline that combines DINOv3-Large visual features with 21-dimensional handcrafted texture descriptors for meteorite identification. The core idea is that meteorites exhibit distinctive surface textures (fusion crust, regmaglypts) that complement global semantic features from vision transformers.

**Architecture:**

```
Input Image (512x512)
    ├── DINOv3-L (frozen) ──→ 1024-dim ──┐
    │                                      ├── Concat(1045) ──→ MLP Head ──→ Output
    └── Texture Extractor ──→ 21-dim  ────┘
    
    + ConvNeXtV2-Base branch (probability-level fusion, weight=0.15)
```

## Key Features

- **DINOv3-Large backbone** (ViT-L/16, 303M params) with frozen weights
- **21-dim handcrafted texture features** (multi-scale variance, gradient energy, FFT, entropy)
- **2-layer MLP classification head** with LayerNorm, GELU activation, and Dropout
- **Strong data augmentation**: RandomPerspective, RandomAffine, RandomErasing (p=0.7)
- **4-view Flip TTA** (original, horizontal, vertical, diagonal) with probability averaging
- **Multi-model blending**: 0.80 × DINO+Tex + 0.15 × ConvNeXtV2-Base

## Requirements

```
Python >= 3.10
PyTorch >= 2.0
torchvision >= 0.15
timm >= 1.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### 1. Download the competition data

Place the following files in your data directory:

```
data/
├── train_labels.csv          # Training labels (id, label columns)
├── sample_submission.csv      # Submission template
├── train_images/
│   └── train_images/          # 5098 training images (.jpg)
└── test_images/
    └── test_images/           # 194 test images (.jpg)
```

### 2. (Optional) Rock dataset augmentation

For improved non-meteorite discrimination, add rock type images:

```
data/Dataset/
├── Igneous/{Basalt, Granite}/
├── Metamorphic/{Marble, Quartzite}/
├── Sedimentary/{Coal, Limestone, Sandstone}/
└── Meteorite/                # Additional meteorite samples
```

### 3. Download pretrained backbone weights

```bash
# DINOv3-Large (1.2 GB)
pip install huggingface_hub
huggingface-cli download timm/vit_large_patch16_dinov3.lvd1689m \
    --local-dir ./pretrained/dinov3_large/

# ConvNeXtV2-Base (355 MB) — optional, for ensemble blending
# Download from: https://dl.fbaipublicfiles.com/convnext/convnextv2/base/convnextv2_base_22k_384_ema.pt
# Place at: ./pretrained/convnextv2_base_22k_384_ema.pt
```

**Note:** Due to file size limits, backbone weights are NOT included in this repository. They will be downloaded automatically by `timm` on first run (requires internet), or you can pre-download them as shown above.

## Quick Start

### Inference Only (Recommended)

Use our pre-trained model to generate predictions directly:

```bash
python infer.py \
    --data_path ./data \
    --output submission.csv
```

This will:
1. Load the pre-trained DINOv3+Tex MLP head from `checkpoints/`
2. Compute texture features for the test set (auto-cached)
3. Run 4-view TTA inference
4. Blend with ConvNeXtV2-Base probabilities
5. Output `submission_blend_top86.csv`

### Full Training

To reproduce the entire training pipeline:

```bash
python train.py --data_path ./data --output_dir ./runs
```

The training script will:
1. Build augmented dataset (original + Dataset/ + pseudo-labels)
2. Pre-compute 21-dim texture features
3. Train the DINOv3+Tex MLP head (~15-20 min on NVIDIA L20)
4. Run 4-view TTA inference
5. Blend with ConvNeXtV2-Base
6. Output final submission

## Model Checkpoints

| File | Description | Size |
|------|-------------|------|
| `checkpoints/dinov3_tex_mlp_head.pth` | Trained DINOv3+Tex MLP head (epoch 7, val_f1=0.975) | 2.1 MB |
| `checkpoints/convnextv2_base_head.pth` | Trained ConvNeXtV2-Base head (epoch 5, val_acc=0.980) | 9.9 KB |
| `checkpoints/convnextv2_base_test_prob.csv` | Pre-computed ConvNeXtV2 test probabilities | 12 KB |
| `checkpoints/texture_cache.npz` | Pre-computed texture features (RobustScaler fitted) | 605 KB |

**Cloud download (完整离线包，含训练好的模型头 + 纹理缓存 + 补充数据集)：**
- [百度网盘](https://pan.baidu.com/s/11ZQfDjY0uJAMb_7RTkR_kQ?pwd=dn2w) (提取码: dn2w)

压缩包 `project_models_and_data.zip` (240 MB) 包含：
- `checkpoints/` — 训练好的 DINOv3+Tex MLP 头、ConvNeXtV2 头、纹理缓存
- `Dataset_supplement.tar.gz` — 2639 张岩石补充图片

## Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Backbone | DINOv3-L (ViT-L/16, registers) |
| Image size | 512 × 512 |
| MLP head | LayerNorm(1045) → Linear(512) → GELU → Dropout(0.3) → Linear(2) |
| Learning rate | 4 × 10⁻⁵ |
| Weight decay | 2 × 10⁻² |
| Label smoothing | 0.1 |
| Batch size | 8 |
| Epochs | 20 (patience=8, F1-based) |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR (η_min = 10⁻⁶) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |

## Data Augmentation

```python
transforms.Compose([
    RandomResizedCrop(512, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomPerspective(distortion_scale=0.3, p=0.6),
    RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    ToTensor(),
    RandomErasing(p=0.7, scale=(0.02, 0.2)),
    Normalize(ImageNet stats),
])
```

## 21-Dimensional Texture Features

| Dims | Source | Physical Meaning |
|------|--------|-----------------|
| 1-5 | Window 7×7 std stats | Fine grain (micro-cracks) |
| 6-10 | Window 15×15 std stats | Medium texture (regmaglypt edges) |
| 11-15 | Window 31×31 std stats | Coarse texture (large indentations) |
| 16 | Global gray std | Overall brightness dispersion |
| 17 | Global contrast (max-min) | Light-dark gap |
| 18 | Sobel gradient mean | Average edge intensity |
| 19 | Sobel gradient std | Edge distribution non-uniformity |
| 20 | FFT high-freq ratio (r > h/4) | Surface detail amount |
| 21 | Histogram entropy (32 bins) | Texture complexity |

All computed on 256×256 grayscale images, normalized with `RobustScaler` (median/IQR resistant to outliers).

## Results

| Version | Method | val_f1 | Public F1 |
|---------|--------|--------|-----------|
| V1 | SVM+kNN baseline | — | ~0.78 |
| V2 | MLP head (corrected LR/aug) | 0.994 | — |
| V4 | +Dataset augmentation | 0.982 | improved |
| **V8** | **+21-dim texture + ConvNeXtV2 blend** | **0.975** | **0.85** |

## Citation

If you find this work useful, please cite:

```bibtex
@misc{meteorite-classification-2026,
  title  = {Meteorite Image Classification with DINOv3 and Handcrafted Texture Features},
  author = {STA326 Project},
  year   = {2026},
}
```

## References

- Oquab, M. et al. *DINOv3: All are DINOs now.* (2025)
- Woo, S. et al. *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders.* CVPR 2023.
- Carion, N. et al. *Emerging Properties in Self-Supervised Vision Transformers.* ICCV 2021.

## License

This project is for educational purposes as part of the STA326 Data Science Practice course.
