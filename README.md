# 🛍️ Product Image Classification

An end-to-end fashion product image classification project: a **Kaggle training pipeline** (Custom CNN, MobileNetV2, ResNet50 + Grad-CAM/SHAP explainability) and a **Streamlit web app** that serves the best model (fine-tuned ResNet50) with Grad-CAM visualisations for each prediction.

🔗 **Live app:** [pic-ml.streamlit.app](https://pic-ml.streamlit.app/)
📓 **Kaggle notebook:** [product-image-classification-framework](https://www.kaggle.com/code/rajsinghh/product-image-classification-framework)

**Dataset:** [Fashion Products Images – Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/e-commerce-products-images)

## Categories

| Icon | Class |
|------|-------|
| 👜 | Accessories |
| 👕 | Apparel |
| 👟 | Footwear |
| 🎁 | Free Items |
| 🧴 | Personal Care |

## Repo Structure

```
.
├── app.py                        # Streamlit entrypoint (redirects to Home)
├── style.css                     # Global stylesheet
├── requirements.txt              # Streamlit app dependencies
├── .streamlit/config.toml
├── models/
│   └── resnet50_final.keras      # Trained model served by the app
├── pages/
│   ├── 1_🏠_Home.py
│   ├── 2_🧠_Explainability.py
│   └── 3_ℹ️_About.py
├── utils/
│   ├── config.py                 # Centralised constants
│   ├── model.py                  # Model loader (@st.cache_resource)
│   ├── predictor.py              # Inference logic
│   ├── gradcam.py                # Grad-CAM heatmap generation
│   └── ui.py                     # Shared sidebar, CSS, chart helpers
└── kaggle/
    ├── notebooks/
    │   └── product-image-classification-framework.ipynb
    ├── reports/                  # Per-model classification reports (CSV)
    └── figures/                  # Training curves, confusion matrices, Grad-CAM & SHAP plots (PDF)
```

## 1. Model Training (Kaggle)

The full training pipeline lives in `kaggle/notebooks/product-image-classification-framework.ipynb` and was run on Kaggle. It:

1. Loads and explores the Fashion Products Images dataset
2. Preprocesses and splits data into train/val/test
3. Trains a **Custom CNN** from scratch
4. Fine-tunes **MobileNetV2** and **ResNet50** via transfer learning
5. Evaluates all three models and compares them
6. Generates **Grad-CAM** visualisations (including on misclassified images) and **SHAP** explanations
7. Saves the final trained models

### Model comparison

| Model | Accuracy | Precision | Recall | Macro F1 | Training Time | Parameters |
|---|---|---|---|---|---|---|
| Custom CNN | 0.9766 | 0.7651 | 0.7822 | 0.7732 | 5538.4s | 652,069 |
| MobileNetV2 | 0.9878 | 0.7876 | 0.7871 | 0.7873 | 2499.6s | 2,587,205 |
| **ResNet50** | **0.9886** | 0.7876 | 0.7886 | 0.7881 | 3603.1s | 24,113,541 |

ResNet50 gave the best overall accuracy/F1 and is the model shipped with the Streamlit app. Full per-class reports are in `kaggle/reports/`, and training curves, confusion matrices, and Grad-CAM/SHAP figures are in `kaggle/figures/`.

### Running the notebook

The notebook is published on Kaggle at [kaggle.com/code/rajsinghh/product-image-classification-framework](https://www.kaggle.com/code/rajsinghh/product-image-classification-framework) — open it there and hit "Copy & Edit" to run it with the [Fashion Products Images dataset](https://www.kaggle.com/datasets/bhavikjikadara/e-commerce-products-images) already attached. 

All required libraries (TensorFlow, scikit-learn, OpenCV, SHAP, etc.) are preinstalled in the Kaggle notebook environment. 

Alternatively, run `kaggle/notebooks/product-image-classification-framework.ipynb` locally in Jupyter after attaching the dataset yourself.

## 2. Web App (Streamlit)

A Streamlit app that classifies an uploaded product image into one of the five categories using the fine-tuned ResNet50 model, with Grad-CAM overlays explaining each prediction.

🔗 **Try it live:** [pic-ml.streamlit.app](https://pic-ml.streamlit.app/)

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure the trained model is present at:
#    models/resnet50_final.keras

# 3. Run the app
streamlit run app.py
```

### Tech Stack

- **TensorFlow / Keras** — ResNet50 inference
- **Streamlit** — web UI
- **OpenCV** — Grad-CAM heatmap overlay
- **NumPy / Pillow** — image preprocessing
