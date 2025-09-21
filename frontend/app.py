# frontend/app.py
import streamlit as st
import pandas as pd
from PIL import Image
import io
import os
import sys
import cv2
import numpy as np

# -------------------------------
# Ensure Python can find backend folder (one level up)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import backend functions
from backend.omr_eval import (
    preprocess_image,
    correct_perspective,
    get_bubbles,
    classify_bubbles_to_options,
    load_answer_key,
    score_answers
)

# -------------------------------
# Page setup
st.set_page_config(page_title="📄 OMR Evaluator", layout="wide")
st.title("📄 OMR Evaluator")

# -------------------------------
# File uploader
uploaded_file = st.file_uploader("Upload OMR Sheet Image", type=["jpg", "jpeg", "png"])
answer_key_file = st.file_uploader("Upload Answer Key (Excel)", type=["xlsx"])

if uploaded_file is not None and answer_key_file is not None:
    # Convert uploaded file to PIL Image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.subheader("Uploaded OMR Sheet:")
    st.image(image, caption="OMR Sheet", use_container_width=True)

    # Save uploaded files temporarily
    img_path = "temp_omr_image.png"
    image.save(img_path)
    answer_key_path = "temp_answer_key.xlsx"
    with open(answer_key_path, "wb") as f:
        f.write(answer_key_file.getbuffer())

    # -------------------------------
    # Backend processing
    st.info("Processing image and scoring...")

    try:
        # Preprocess image
        thresh, original_img = preprocess_image(img_path)
        warped_img = correct_perspective(thresh, original_img)

        # Detect bubbles
        warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        _, warped_thresh = cv2.threshold(
            warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        bubble_contours = get_bubbles(warped_thresh)

        # Classify answers
        detected_answers = classify_bubbles_to_options(warped_thresh, bubble_contours)

        # Load answer key and score
        answer_key = load_answer_key(answer_key_path)
        scores, total_score = score_answers(detected_answers, answer_key)

    except Exception as e:
        st.error(f"Error during evaluation: {e}")
        scores, total_score = {}, 0

    # -------------------------------
    # Display results
    if scores:
        st.subheader("Evaluation Results:")
        df_scores = pd.DataFrame(list(scores.items()), columns=["Subject", "Marks"])
        st.table(df_scores)
        st.markdown(f"**Total Score:** {total_score}")

        # CSV Download
        csv = df_scores.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv,
            file_name="omr_results.csv",
            mime="text/csv"
        )

        # Overlay detected bubbles
        overlay_img = warped_img.copy()
        for idx, c in enumerate(bubble_contours):
            x, y, w, h = cv2.boundingRect(c)
            color = (0, 255, 0)  # green for detected bubble
            thickness = 2
            cv2.rectangle(overlay_img, (x, y), (x + w, y + h), color, thickness)

            # Mark selected answer
            option = detected_answers[idx] if idx < len(detected_answers) else None
            if option not in [None, 'ambiguous']:
                cv2.putText(
                    overlay_img, option, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                )

        overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
        st.subheader("Detected Bubbles Overlay:")
        st.image(overlay_img_rgb, caption="Bubbles with selections", use_container_width=True)

    # -------------------------------
    # Clean up temporary files
    if os.path.exists(img_path):
        os.remove(img_path)
    if os.path.exists(answer_key_path):
        os.remove(answer_key_path)
