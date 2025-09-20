import streamlit as st
from PIL import Image
import io
import pandas as pd

st.set_page_config(page_title="OMR Evaluator Demo", layout="wide")
st.title("📄 OMR Evaluator Demo (Frontend Only)")

# 1️⃣ Answer key selection (for future backend)
key_set = st.selectbox("Choose Answer Key", ["Set A", "Set B"])

# 2️⃣ File uploader
uploaded_file = st.file_uploader("Upload OMR Sheet Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to PIL Image
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Show the uploaded image
    st.subheader("Uploaded OMR Sheet:")
    st.image(image, caption="OMR Sheet", use_container_width=True)

    # -------------------------------
    # 3️⃣ Dummy results (replace with backend later)
    scores = {
        "Maths": 15,
        "Reasoning": 12,
        "English": 10,
        "GK": 14,
        "Aptitude": 13,
        "Total": 64
    }

    # Display table
    st.subheader("Evaluation Results:")
    df_scores = pd.DataFrame(list(scores.items()), columns=["Subject", "Marks"])
    st.table(df_scores)

    # 4️⃣ CSV Download
    csv = df_scores.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name="omr_results.csv",
        mime="text/csv"
    )

    # 5️⃣ Overlay placeholder (optional, for future backend)
    st.subheader("Detected Bubbles Overlay (Placeholder):")
    st.image(image, caption="Overlay will appear here", use_container_width=True)
