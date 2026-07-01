import streamlit as st

from utils.ui import load_css, render_sidebar

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="About the Project",
    page_icon="ℹ️",
    layout="wide",
)

load_css()
render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; color:#FF4B4B;'>ℹ️ About the Project</h1>
    <h4 style='text-align:center; color:gray;'>
        Fashion Product Classification using Deep Learning
    </h4>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Overview ──────────────────────────────────────────────────────────────────
with st.expander("📌 Project Overview", expanded=True):
    st.write(
        """
        This project presents a **Fashion Product Classification System** built
        with Deep Learning.

        A fine-tuned **ResNet50 Convolutional Neural Network (CNN)** classifies
        fashion product images into five categories. **Grad-CAM** is integrated
        to explain which image regions drove each prediction, making the model
        transparent and interpretable.
        """
    )

# ── Dataset ───────────────────────────────────────────────────────────────────
with st.expander("📂 Dataset"):
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Images", "44,415")
        st.metric("Categories", "5")

    with col2:
        st.metric("Image Size", "128 × 128")
        st.metric("Dataset Type", "Fashion Products")

    st.markdown("### Product Categories")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("👜 Accessories")
        st.success("👕 Apparel")

    with c2:
        st.success("👟 Footwear")
        st.success("🎁 Free Items")

    with c3:
        st.success("🧴 Personal Care")

# ── Workflow ──────────────────────────────────────────────────────────────────
with st.expander("🔄 Project Workflow"):
    st.code(
        """
Image Upload
      │
      ▼
Resize (128×128)
      │
      ▼
ResNet50 Preprocessing
      │
      ▼
ResNet50 Inference
      │
      ▼
Prediction + Confidence Scores
      │
      ▼
Grad-CAM Heatmap
        """,
        language="text",
    )

# ── Model ─────────────────────────────────────────────────────────────────────
with st.expander("🧠 Deep Learning Model"):
    st.write(
        """
        ### ResNet50

        ResNet50 is a 50-layer deep convolutional neural network that introduces
        **residual (skip) connections** to overcome the vanishing gradient problem.

        It was selected for this project because it achieved the highest validation
        accuracy among all evaluated architectures and generalised well to unseen
        fashion product images.
        """
    )

# ── Technologies ──────────────────────────────────────────────────────────────
with st.expander("🛠 Technologies Used"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🐍 Python")
        st.info("🧠 TensorFlow / Keras")

    with col2:
        st.info("🎨 Streamlit")
        st.info("📷 OpenCV")

    with col3:
        st.info("📊 NumPy")
        st.info("🖼 Pillow")

# ── Future Scope ──────────────────────────────────────────────────────────────
with st.expander("🚀 Future Scope"):
    st.write(
        """
        - Expand to more product categories.
        - Deploy as a mobile application.
        - Add real-time webcam prediction.
        - Integrate additional XAI techniques (SHAP, LIME).
        - Connect with e-commerce recommendation systems.
        - Upgrade to 224 × 224 input size to fully leverage ResNet50 pretraining.
        """
    )

# ── Navigation ────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("🏠 Back to Home"):
        st.switch_page("pages/1_🏠_Home.py")

with col2:
    if st.button("🧠 Explainability"):
        st.switch_page("pages/2_🧠_Explainability.py")

st.markdown("---")
st.caption(
    "Developed as part of a Machine Learning Project using TensorFlow, ResNet50, Streamlit, and Grad-CAM."
)
