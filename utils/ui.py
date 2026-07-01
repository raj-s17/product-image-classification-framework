import streamlit as st

from utils.config import CLASS_NAMES, CLASS_ICONS


def load_css(path: str = "style.css") -> None:
    """Inject the project stylesheet into the page."""
    try:
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS is cosmetic; a missing file should never crash the app


def render_sidebar() -> None:
    """Render the shared sidebar present on every page."""
    with st.sidebar:
        st.title("🛍 Fashion Classifier")
        st.write("---")

        st.success("🤖 Model")
        st.write("ResNet50")

        st.success("🧠 Explainability")
        st.write("Grad-CAM")

        st.success("📂 Classes")
        items = "\n".join(
            f"- {CLASS_ICONS[c]} {c}" for c in CLASS_NAMES
        )
        st.write(items)

        st.write("---")
        st.info("Machine Learning Project")


def prediction_card(category: str, confidence: float, model_name: str = "ResNet50") -> None:
    """Render the blue gradient prediction summary card."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1E3A8A, #2563EB);
            padding: 25px;
            border-radius: 18px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
            color: white;
        ">
            <h2 style="margin-bottom: 20px;">🎯 Prediction Result</h2>
            <table style="width:100%; color:white; font-size:18px;">
                <tr>
                    <td><b>📦 Category</b></td>
                    <td style="text-align:right;"><b>{category}</b></td>
                </tr>
                <tr>
                    <td><b>📈 Confidence</b></td>
                    <td style="text-align:right;"><b>{confidence * 100:.2f}%</b></td>
                </tr>
                <tr>
                    <td><b>🤖 Model</b></td>
                    <td style="text-align:right;">{model_name}</td>
                </tr>
                <tr>
                    <td><b>📏 Input Size</b></td>
                    <td style="text-align:right;">128 × 128</td>
                </tr>
                <tr>
                    <td><b>🟢 Status</b></td>
                    <td style="text-align:right;">Prediction Successful</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_chart(class_names: list[str], probs: list[float]) -> None:
    """Bar chart showing softmax probabilities for all classes."""
    import pandas as pd

    df = pd.DataFrame({"Class": class_names, "Confidence": probs})
    df = df.sort_values("Confidence", ascending=False)

    st.bar_chart(df.set_index("Class")["Confidence"], height=220)
