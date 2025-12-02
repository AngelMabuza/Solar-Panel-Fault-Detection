# SPF‑Net Replication (Segmentation + Classification)

This repository scaffolds a reproducible replication of the paper's methodology:
- U‑Net (MobileNetV2 encoder) for segmentation (IoU/Dice trends, qualitative masks)
- Multiple CNN baselines + proposed InceptionV3‑Net (with SE blocks & residuals) for 6‑class panel fault classification
- 60/20/20 **stratified** splits preserved per class
- Logging to `reports/` (tables/figs/logs) with LaTeX-ready exports

## 1) Environment
Choose one:
```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate spfn

# or pip
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```
> Tested with Python 3.11, TensorFlow 2.15+.

## 2) Expected data layout
```
data/
  segmentation/
    images/            # *.jpg|png (RGB)
    masks/             # *.png     (single-channel binary; same basename as images)
  classification/
    images/
      Clean/ *.jpg
      Dusty/ *.jpg
      Bird-drop/ *.jpg
      Electrical-damage/ *.jpg
      Physical-Damage/ *.jpg
      Snow-Covered/ *.jpg
```

## 3) Make stratified splits (60/20/20)
```bash
# Classification
python -m src.data_prep.make_splits \
  --mode cls \
  --root data/classification/images \
  --out data/classification/splits.csv

# Segmentation
python -m src.data_prep.make_splits \
  --mode seg \
  --root data/segmentation \
  --out data/segmentation/splits.csv
```

## 4) Train & evaluate
**Segmentation (U‑Net MobileNetV2):**
```bash
python -m src.seg.train_seg \
  --csv data/segmentation/splits.csv \
  --out reports/seg_unet \
  --epochs 8 --batch 24 --img 128

python -m src.seg.eval_seg \
  --csv data/segmentation/splits.csv \
  --model reports/seg_unet/best.keras \
  --out reports/seg_unet/eval
```

**Classification (Baselines & Proposed):**
```bash
# e.g., InceptionV3‑Net (proposed)
python -m src.cls.train_cls \
  --csv data/classification/splits.csv \
  --model_name inceptionv3_net_proposed \
  --out reports/cls_incv3p \
  --epochs 30 --batch 32 --img 256

# Evaluate on test set + export confusion matrix & LaTeX tables
python -m src.cls.eval_cls \
  --csv data/classification/splits.csv \
  --model reports/cls_incv3p/best.keras \
  --out reports/cls_incv3p/eval
```

## 5) Outputs
- `reports/**/history.csv` — per‑epoch metrics
- `reports/**/test_metrics.json` — final test set metrics
- `reports/**/confusion_matrix.png` — classification
- `reports/**/tables/*.tex` — LaTeX tables matching paper’s summaries

## 6) Repro tips
- Fix seeds via `--seed` (training scripts do this for TF/Numpy/Python).
- Ensure **no leakage**: splits script guards by filename basenames.
- If MobileNetV3 is unavailable in your TF build, skip that baseline or switch to Small/Large variants.

---
Happy replicating!
