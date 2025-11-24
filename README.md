# Solar-Panel-Fault-Detection

## Project Overview
The project replicates the results of [Rudro et al., 2024](https://doi.org/10.1016/j.egyr.2024.07.044). The main goal is to develop the proposed **SPF-Net** model to detect and monitor solar panel faults in real-time and benchmark it against existing pre-trained models.
 
The methodology follows a **two-stage computer vision pipeline**:
1. **Segmentation** – U-Net with MobileNetV2 encoder isolates PV regions in satellite imagery.
2. **Classification** – InceptionV3-based classifier (with SE blocks and residual connections) categorizes panel surface conditions.


---

## Datasets
-We will utilize the multi-resolution [dataset](https://zenodo.org/records/5171712) for photovoltaic panel segmentation from satellite and aerial imagery. 
-The dataset includes three groups of PV samples collected at the spatial resolution of 0.8m, 0.3m and 0.1m and total size of the data is  about 7GB.

---
## Methodology


### Stage 1: Segmentation
- **Model**: U-Net with MobileNetV2 encoder  
- **Input**: 256×256 RGB images  
- **Output**: 128×128 binary mask (PV vs background)  
- **Metrics**: Dice coefficient, Intersection-over-Union (IoU), binary accuracy  

### Stage 2: Classification
- **Model**: InceptionV3 backbone + SE blocks + residual connections  
- **Classes**:  
  - Clean  
  - Dusty  
  - Bird droppings  
  - Electrical damage  
  - Physical damage  
  - Snow-covered  
- **Metrics**: Accuracy, Precision, Recall, Macro-F1  

---

## ⚙️ Training Protocol
- **Split**: 60/20/20 (train/val/test)  
- **Optimizer**: Adam (lr = 1e-4)  
- **Losses**:  
  - Segmentation – Binary cross-entropy  
  - Classification – Categorical cross-entropy  
- **Evaluation**: Per-epoch logging of IoU, Dice, Accuracy, Precision, Recall, F1  

---

## Expected Performance
Replication targets reported ranges:
- Segmentation IoU ≈ 0.76  
- Classification Validation Accuracy ≈ 98.3%  
- Test Accuracy ≈ 94.4%  
- Macro-F1 ≈ 0.94  

---

## 📂 Repository Structure
-Benchmark models are set in various .ipynb files
