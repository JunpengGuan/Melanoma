# Melanoma vs Benign Lesion Classification (EE 5271)

Two pipelines on **ISIC / ISBI 2016** dermoscopy data:

| Method | Idea |
|--------|------|
| **Method 1** | End-to-end **EfficientNet-B0** (or ViT) binary classifier — no lesion mask. |
| **Method 2** | **U-Net** lesion segmentation → **ABCD-style** features → **Logistic Regression** / **XGBoost**. |

---

## 1. Environment

```bash
cd FinalProject   # repository root
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires **PyTorch** with CUDA if you use GPU (install from [pytorch.org](https://pytorch.org) if the default wheels do not match your system).

---

## 2. Data layout

Place challenge files under the **repository root** so defaults in `melanoma/config.py` resolve:

| Path | Contents |
|------|----------|
| `ISBI2016_ISIC_Part1_Training_Data/` | Training images `ISIC_*.jpg` (~900) |
| `ISBI2016_ISIC_Part1_Training_GroundTruth/ISBI2016_ISIC_Part1_Training_GroundTruth/` | Training masks `ISIC_*_Segmentation.png` |
| `ISBI2016_ISIC_Part3_Training_GroundTruth.csv` | Training labels (`benign` / `malignant` or `0`/`1`) |
| `ISBI2016_ISIC_Part3B_Test_Data/` | Test images + optional masks |
| `ISBI2016_ISIC_Part3B_Test_GroundTruth.csv` | Test labels (`0.0` / `1.0`) |

**Note:** Image folders are **gitignored** (too large). Clone the repo, then download and unzip ISIC 2016 assets into the paths above.

---

## 3. Method 1 — train & validate

**Train** (stratified **train/val** split inside the script; metrics printed each epoch):

```bash
python -m melanoma.method1.train \
  --epochs 5 \
  --batch-size 16 \
  --backbone efficientnet_b0
```

- Weights: `checkpoints/efficientnet_b0_last.pt` (default `--checkpoint-dir`).
- **Validation:** same run — look for `val_loss`, `val_acc` in the console (hold-out ratio `--val-ratio`, default `0.15`).

**Test / held-out evaluation** (379 test images + CSV):

```bash
python -m melanoma.method1.eval_test \
  --checkpoint checkpoints/efficientnet_b0_last.pt \
  --backbone efficientnet_b0
```

Defaults point to `ISBI2016_ISIC_Part3B_Test_Data` and `ISBI2016_ISIC_Part3B_Test_GroundTruth.csv`.  
Output: accuracy, sensitivity, specificity, confusion matrix, **AUC-ROC**.

**Optional backbone:**

```bash
python -m melanoma.method1.train --backbone vit_b_16
```

---

## 4. Method 2 — train & validate

### Stage A — U-Net segmentation (CNN)

**Train** (prints **val Dice** and **val IoU** each epoch):

```bash
python -m melanoma.method2.train_seg \
  --epochs 30 \
  --batch-size 8
```

- Weights: `checkpoints/unet_last.pt`.
- **Validation:** `val_dice` (soft Dice) and `val_iou` (binary IoU at 0.5) on the stratified val split.

**Segmentation-only metrics** (reload checkpoint, same val split):

```bash
python -m melanoma.method2.eval_seg --checkpoint checkpoints/unet_last.pt
```

### Stage B — ABCD features + tabular classifiers

**Train** (uses **train split only** for labels; masks from U-Net unless `--use-gt-mask`):

```bash
python -m melanoma.method2.train_tabular
```

Saves:

- `checkpoints/method2_lr.joblib` — `StandardScaler` + Logistic Regression  
- `checkpoints/method2_xgb.json` — XGBoost model  

**Oracle upper bound** (ground-truth masks instead of U-Net):

```bash
python -m melanoma.method2.train_tabular --use-gt-mask
```

### Stage C — Test evaluation (segmentation + classifier)

```bash
python -m melanoma.method2.eval_test --classifier lr
python -m melanoma.method2.eval_test --classifier xgb
```

Loads `unet_last.pt` + the chosen classifier; same test paths as Method 1 by default.

---

## 5. Project layout

```
melanoma/
  config.py              # default paths
  method1/               # CNN classification
  method2/               # U-Net, ABCD, tabular, eval_seg
scripts/
  build_presentation.py  # optional: English deck (needs python-pptx)
requirements.txt
checkpoints/             # created when training (.gitignored weights)
```

---

## 6. Push to GitHub

From the repository root (after `git init` if needed):

```bash
git add .
git commit -m "Initial commit: Method 1 & 2 melanoma classification"
git branch -M main
git remote add origin https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git push -u origin main
```

Create an empty repository on GitHub first, then use its URL as `origin`.  
If checkpoints or data were accidentally staged, run `git reset` and rely on `.gitignore`.

---

## 7. Citation / data

Use the **ISIC 2016 / ISBI challenge** citation required by the archive when you publish or present results.
