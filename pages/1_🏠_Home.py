import streamlit as st
from PIL import Image

from utils.config import CLASS_NAMES
from utils.model import load_resnet
from utils.predictor import predict
from utils.ui import load_css, render_sidebar, prediction_card, confidence_chart

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="Fashion Product Classification",
    page_icon="🛍️",
    layout="wide",
)

load_css()
render_sidebar()

model = load_resnet()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; color:#FF4B4B;'>
        🛍 Fashion Product Classification
    </h1>
    <h4 style='text-align:center; color:inherit;'>
        Deep Learning Image Classification using ResNet50
    </h4>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📤 Upload a Fashion Product Image",
    type=["jpg", "jpeg", "png"],
)

if not uploaded_file:
    st.info("👆 Upload a product image to begin.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.image(image, caption="Uploaded Image", width=500)

with col2:
    st.subheader("Prediction")

    if st.button("🚀 Predict Product"):
        with st.spinner("Analyzing image…"):
            category, confidence, probs = predict(model, image)

        # Persist results for the Explainability page
        st.session_state.update(
            image=image,
            prediction=category,
            confidence=confidence,
            probs=probs,
            gradcam=None,           # will be generated lazily on the next page
            prediction_done=True,
        )

        prediction_card(category, confidence)
        st.write("")
        st.markdown("**📊 Class Probabilities**")
        confidence_chart(CLASS_NAMES, probs)

    # Show the "Explain" button whenever a prediction exists in this session
    if st.session_state.get("prediction_done"):
        st.write("")
        if st.button("🧠 Explain Prediction"):
            st.switch_page("pages/2_🧠_Explainability.py")

st.markdown("---")
st.caption("Developed using TensorFlow • ResNet50 • Streamlit • Grad-CAM")
