# 🛍️ Product Image Classification Framework

## 📋 Project Overview

This project addresses multi-class image classification for fashion e-commerce using Convolutional Neural Networks. Given a product image, the model automatically assigns it to one of **five high-level product categories**:

> `Accessories` · `Apparel` · `Footwear` · `Free Items` · `Personal Care`

Three models are implemented and compared:
- **Custom CNN** — trained from scratch (652K parameters)
- **MobileNetV2** — lightweight transfer learning
- **EfficientNetB0** — state-of-the-art transfer learning

---

## 📊 Results Summary

| Model | Test Accuracy | Weighted F1 | Macro F1 | Parameters |
|---|---|---|---|---|
| Custom CNN | **98.33%** | 98.23% | 78.20% | 652,069 |
| MobileNetV2 | 97.15% | 97.04% | 76.70% | 2,587,205 |
| EfficientNetB0 | **98.86%** | 98.75% | 78.80% | 4,378,792 |

> **Key Finding:** The Custom CNN delivers strong performance with only 652K parameters — roughly **6× more parameter-efficient** than EfficientNetB0.

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| **Name** | Fashion Products Images |
| **Source** | https://www.kaggle.com/datasets/bhavikjikadara/e-commerce-products-images |
| **Total Images** | ~44,000+ |
| **Classes** | 5 |
| **Image Type** | RGB JPEG |
| **Split** | 70% train / 15% validation / 15% test (stratified) |
| **Input Resolution** | 128 × 128 pixels |

---

## ⚙️ How to Run

### Kaggle Notebook (Recommended)

1. Open the Kaggle Notebook link: https://www.kaggle.com/code/rajsinghh/product-image-classification-framework
2. Click **Copy & Edit** to fork the notebook
3. Enable GPU accelerator: **Settings → Accelerator → GPU P100**
4. Attach the dataset: search for `https://www.kaggle.com/datasets/bhavikjikadara/e-commerce-products-images`
5. Click **Run All**

---

## 📈 Key Output Figures

| Figure | Description |
|---|---|
| `class_distribution.png` | Train/val/test class imbalance overview |
| `augmented_samples.png` | Data augmentation examples per class |
| `cnn_training_curves.png` | Custom CNN accuracy & loss over epochs |
| `Custom_CNN_confusion_matrix.png` | Per-class confusion on test set |
| `gradcam_visualisations.png` | Grad-CAM saliency maps |
| `model_comparison_chart.png` | Accuracy vs. parameter count comparison |

---

## 🛠️ Tech Stack

- **Language:** Python 3.10
- **Framework:** TensorFlow 2.x / Keras
- **Environment:** Kaggle Notebooks (GPU)
- **Libraries:** NumPy · Pandas · Scikit-learn · Matplotlib · Seaborn · OpenCV · Pillow

---