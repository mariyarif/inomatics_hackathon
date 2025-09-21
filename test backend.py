# test_backend.py
import os
import cv2
import numpy as np
from backend.omr_eval import (
    evaluate_omr,
    preprocess_image,
    correct_perspective,
    get_bubbles,
    classify_bubbles_to_options,
    load_answer_key,
    score_answers
)

# === CONFIG: edit these paths to match your files ===
IMG_PATH = os.path.abspath(r"C:\Users\User\Documents\OMR_EVALUATOR\data\img1.jpeg")
ANSWER_KEY_PATH = os.path.abspath(r"C:\Users\User\Documents\OMR_EVALUATOR\data\answer_key.xlsx")
OVERLAY_OUT = os.path.abspath("overlay_test.jpg")
# ===================================================

def run_test():
    print("Running backend test...")
    print(f"Image: {IMG_PATH}")
    print(f"Answer key: {ANSWER_KEY_PATH}")

    # 1) High-level evaluate_omr() call
    try:
        scores, total = evaluate_omr(IMG_PATH, ANSWER_KEY_PATH)
        print("\n== EVALUATE_OMR() RESULT ==")
        for k, v in scores.items():
            print(f"{k}: {v}")
        print("Total Score:", total)
    except Exception as e:
        print("evaluate_omr() raised an error:", e)

    # 2) Lower-level pipeline for debugging
    try:
        thresh, original = preprocess_image(IMG_PATH)
        warped = correct_perspective(thresh, original)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, warped_thresh = cv2.threshold(
            warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        contours = get_bubbles(warped_thresh)
        print(f"\nDetected bubble contours: {len(contours)}")

        detected_answers = classify_bubbles_to_options(warped_thresh, contours)
        print(f"Detected answers (first 40 shown): {detected_answers[:40]}")

        ak = load_answer_key(ANSWER_KEY_PATH)
        print(f"Loaded answer key subjects: {list(ak.keys())}")

        scores, total = score_answers(detected_answers, ak)
        print("\n== SCORES (lower-level) ==")
        for s, sc in scores.items():
            print(f"{s}: {sc}")
        print("Total:", total)

        # Save overlay image showing detected bubbles
        overlay = warped.copy()
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(OVERLAY_OUT, overlay)
        print(f"\nOverlay image saved to: {OVERLAY_OUT}")

    except Exception as e:
        print("Lower-level pipeline error:", e)

if __name__ == "__main__":
    run_test()
