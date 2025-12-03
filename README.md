**Project:** SPF‑Net Replication (Segmentation + Classification)

This repository contains code and notebooks used to (1) train a U-Net segmentation model that localizes solar panels in aerial/rooftop imagery, (2) export classification-ready datasets derived from segmentation results (masked and cropped variants), and (3) train and evaluate classification models across three dataset variants (base, masked, cropped).

**Short summary:**
- **Base**: original full-frame images (no segmentation applied).
- **Masked**: original images where pixels outside the panel are zeroed (or saved with alpha). Keeps image size, removes background signal.
- **Cropped**: images cropped tightly to the detected panel bounding box (derived from segmentation), yielding smaller inputs focused on the panel.

**Why three variants?**
- The project compares how segmentation-derived preprocessing affects classification performance. Cropped inputs typically remove irrelevant background and often yield better results; masked inputs remove background signal while preserving spatial context.

**Environment**
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
**Repository layout (important paths)**
- `src/` : implementation for segmentation and classification training/eval utilities.
- `data/` : source images and generated datasets.
  - `data/classification/splits_spfnet.csv` : original classification split CSV.
  - `data/classification/localized/` : exporter outputs with `masked` and `cropped` subfolders and CSVs (`splits_localized_masked.csv`, `splits_localized_cropped.csv`).
- `reports/` : training runs, saved models, learning curves and evaluation metrics.
- `COS 801 Project_v3.ipynb` : notebook used as the main exploratory / pipeline notebook; contains exporter (`run_exporter()`), training and evaluation code paths.
-**data root structur:**
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
**Quick start (requirements)**
- **Python**: create environment from `environment.yml` or install `requirements.txt`.

Example (PowerShell):
```
conda env create -f environment.yml -n deeplearning_env
conda activate deeplearning_env
pip install -r requirements.txt
```

**Generating masked / cropped classification datasets**
1. Train or obtain a segmentation model (U-Net) and set `SEG_MODEL` path in the notebook or pass path to exporter.
2. Run the exporter from the notebook or call the exporter function to create `masked` and `cropped` datasets and CSVs.

Example (from `COS 801 Project_v3.ipynb` cells):
```
# in notebook: run_exporter(CSV_BASE, SEG_MODEL, out_root=OUT_LOCALIZED, mode="both", ...)
```

The exporter performs:
- segmentation inference to get probability masks
- thresholding + morphological postprocessing to produce binary masks
- `masked`: applies mask to zero-out pixels outside panel
- `cropped`: extracts bounding boxes around detected panels and saves cropped images

After a successful run the exporter writes:
- `data/classification/localized/splits_localized_masked.csv`
- `data/classification/localized/splits_localized_cropped.csv`

**Training classification models**
- The classification training and multi-model runner is in `src/cls` and the notebook contains helper wrappers.
- Models used: `cnn_plain`, `densenet`, `ineptionv3` (alias), `inceptionv3_proposed`, `resnet50`, `vgg16`, `vgg19`, `mobilenetv3`.

Example commands shown in the notebook (PowerShell / Windows):
```
# Train on cropped localized CSV
python -m src.cls.train_cls --csv data/classification/localized/splits_localized_cropped.csv --model_name cnn_plain --out reports\cls_runs\cnn_plain --epochs 30 --batch 32 --img 256

# Evaluate a trained model
python -m src.cls.eval_cls --csv data/classification/localized/splits_localized_cropped.csv --model reports\cls_runs\cnn_plain\best.keras --out reports\cls_runs\cnn_plain\eval --img 256
```

Notes:
- The `--csv` argument points to a CSV with columns: `split,filepath,label`.
- Use the exported CSVs for `masked` or `cropped` runs; use the original split CSV for `base` runs.

**Running the multi-variant experiments (notebook flow)**
- The notebook includes code to train/evaluate three variants sequentially. It uses `CSV_BASE`, `CSV_MASKED`, `CSV_CROPPED` variables and will run models on each CSV it finds.

**Results & logs**
- Training outputs, best checkpoints and metrics are in `reports/`.
- `final_results.csv` contains summarized results across runs and models.

**Reproducing the paper/figures**
- Use the notebook cells that wrap `train_many()` and `run_exporter()` to recreate experiments. The notebook saves learning curves and JSON metrics for each run.

**Extending / ideas**
- Use the segmentation mask as an additional channel (RGBA / 4-channel input) or build a dual-branch network to explicitly fuse segmentation features.
- Test soft masks (probabilities) rather than hard thresholding to provide attention-like input.

**Troubleshooting**
- If exporter reports `segmentation model not found`, ensure `SEG_MODEL` points to a saved Keras model file in `reports/seg_unet` or equivalent.
- If images fail to read, check file paths in the CSV and ensure relative paths are correct from the notebook/workdir.

**Repro tips**
- Fix seeds via `--seed` (training scripts do this for TF/Numpy/Python).
- Ensure **no leakage**: splits script guards by filename basenames.
- If MobileNetV3 is unavailable in your TF build, skip that baseline or switch to Small/Large variants.

---
Happy replicating!
