import streamlit as st

from utils.config import CLASS_NAMES
from utils.model import load_resnet
from utils.gradcam import generate_gradcam
from utils.ui import load_css, render_sidebar, confidence_chart

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="Explainability",
    page_icon="🧠",
    layout="wide",
)

load_css()
render_sidebar()

# ── Guard: require a prediction ───────────────────────────────────────────────
if "prediction" not in st.session_state:
    st.warning("⚠️ Please predict an image first.")
    if st.button("🏠 Back to Home"):
        st.switch_page("pages/1_🏠_Home.py")
    st.stop()

# ── Pull state ────────────────────────────────────────────────────────────────
image      = st.session_state["image"]
prediction = st.session_state["prediction"]
confidence = st.session_state["confidence"]
probs      = st.session_state["probs"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; color:#FF4B4B;'>🧠 Explainable AI</h1>
    <h4 style='text-align:center; color:inherit;'>Understanding the Model's Decision</h4>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Grad-CAM (cached in session to avoid recomputing on reruns) ───────────────
model = load_resnet()

if st.session_state.get("gradcam") is None:
    with st.spinner("Generating Grad-CAM…"):
        st.session_state["gradcam"] = generate_gradcam(model, image)

gradcam = st.session_state["gradcam"]

# ── Images ────────────────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("🖼 Original Image")
    st.image(image, width=500)

with right:
    st.subheader("🔥 Grad-CAM Heatmap")
    st.image(gradcam, width=500)

st.markdown("---")

# ── Summary ───────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1E3A8A, #2563EB);
            padding: 25px;
            border-radius: 18px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
            color: white;
        ">
            <h2 style="margin-bottom: 20px;">📋 Prediction Summary</h2>
            <table style="width:100%; color:white; font-size:18px;">
                <tr>
                    <td><b>📦 Category</b></td>
                    <td style="text-align:right;"><b>{prediction}</b></td>
                </tr>
                <tr>
                    <td><b>📈 Confidence</b></td>
                    <td style="text-align:right;"><b>{confidence * 100:.2f}%</b></td>
                </tr>
                <tr>
                    <td><b>🤖 Model</b></td>
                    <td style="text-align:right;">ResNet50</td>
                </tr>
                <tr>
                    <td><b>🧠 Explainability</b></td>
                    <td style="text-align:right;">Grad-CAM</td>
                </tr>
                <tr>
                    <td><b>📏 Input Size</b></td>
                    <td style="text-align:right;">128 × 128</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown("**📊 Class Probabilities**")
    confidence_chart(CLASS_NAMES, probs)

st.write("")

# ── Dynamic interpretation ────────────────────────────────────────────────────
if confidence >= 0.90:
    certainty_note = "The model is **highly confident** in this prediction."
elif confidence >= 0.65:
    certainty_note = "The model is **moderately confident** — the Grad-CAM may reveal ambiguous features."
else:
    certainty_note = (
        "The model is **uncertain** about this image. "
        "Review the heatmap carefully; the product may belong to an overlapping category."
    )

st.info(
    f"### 💡 Interpretation\n\n"
    f"The product has been classified as **{prediction}** with "
    f"**{confidence * 100:.1f}%** confidence. {certainty_note}\n\n"
    f"The Grad-CAM heatmap highlights the regions that drove this decision. "
    f"Warm colours (red/yellow) indicate high influence; cool colours (blue) indicate low influence."
)

st.markdown("---")

# ── Navigation ────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    if st.button("🏠 Predict Another Image"):
        st.switch_page("pages/1_🏠_Home.py")

with c2:
    if st.button("ℹ️ About Project"):
        st.switch_page("pages/3_ℹ️_About.py")

st.markdown("---")
st.caption("Grad-CAM highlights the most influential image regions used by the ResNet50 model.")
