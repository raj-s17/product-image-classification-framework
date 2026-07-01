import streamlit as st
from tensorflow.keras.models import load_model

from utils.config import MODEL_PATH


@st.cache_resource(show_spinner="Loading model…")
def load_resnet():
    """Load and cache the ResNet50 model. Shows a user-friendly error if the file is missing."""
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(
            f"❌ Could not load model from `{MODEL_PATH}`.\n\n"
            f"Make sure the file exists and is a valid Keras model.\n\n"
            f"**Error:** {e}"
        )
        st.stop()
