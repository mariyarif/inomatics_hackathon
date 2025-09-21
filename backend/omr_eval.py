# backend/omr_eval.py

import cv2
import numpy as np
from PIL import Image
import pandas as pd

# -----------------------
# Step 1: Preprocessing
# -----------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh, img

def correct_perspective(thresh_img, original_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) == 0:
        return original_img
    sheet_contour = contours[0]
    peri = cv2.arcLength(sheet_contour, True)
    approx = cv2.approxPolyDP(sheet_contour, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4,2)
        rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(original_img, M, (maxWidth, maxHeight))
        return warped
    else:
        return original_img

# -----------------------
# Step 2: Bubble Detection
# -----------------------
def get_bubbles(thresh_img, min_area=100, max_area=1500):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    bubble_contours = sorted(bubble_contours, key=lambda c: cv2.boundingRect(c)[1])
    rows = []
    current_row = []
    previous_y = -100
    for c in bubble_contours:
        x, y, w, h = cv2.boundingRect(c)
        if abs(y - previous_y) > 10 and current_row:
            rows.append(sorted(current_row, key=lambda cc: cv2.boundingRect(cc)[0]))
            current_row = [c]
        else:
            current_row.append(c)
        previous_y = y
    if current_row:
        rows.append(sorted(current_row, key=lambda cc: cv2.boundingRect(cc)[0]))
    sorted_bubbles = [c for row in rows for c in row]
    return sorted_bubbles

# -----------------------
# Step 3: Classify Bubbles (FIXED)
# -----------------------
def classify_bubbles_to_options(thresh, contours, options_per_question=4, fill_thresh=0.3):
    detected_answers = []
    for i in range(0, len(contours), options_per_question):
        question_contours = contours[i:i+options_per_question]
        selected_option = None
        multiple_filled = False

        for idx, cnt in enumerate(question_contours):
            area = cv2.contourArea(cnt)
            if area == 0:  # Skip invalid contours
                continue

            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            filled_ratio = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask)) / area

            if filled_ratio > fill_thresh:
                if selected_option is None:
                    selected_option = chr(ord('a') + idx)
                else:
                    multiple_filled = True

        if multiple_filled:
            detected_answers.append("ambiguous")
        else:
            detected_answers.append(selected_option)

    return detected_answers

# -----------------------
# Step 4: Load Answer Key
# -----------------------
def load_answer_key(excel_path):
    df = pd.read_excel(excel_path)
    answer_key = {}
    for col in df.columns:
        answer_key[col] = df[col].tolist()
    return answer_key

# -----------------------
# Step 5: Score Answers
# -----------------------
def score_answers(detected_answers, answer_key, questions_per_subject=20):
    scores = {}
    total_score = 0
    start_idx = 0
    for subject, correct in answer_key.items():
        subject_answers = detected_answers[start_idx:start_idx+questions_per_subject]
        subject_score = 0
        for det, ans in zip(subject_answers, correct):
            # Remove numbering in answer key if any: "1 - a" -> "a"
            ans_clean = ans.split("-")[-1].strip() if isinstance(ans, str) else ans
            if det == ans_clean:
                subject_score += 1
        scores[subject] = subject_score
        total_score += subject_score
        start_idx += questions_per_subject
    return scores, total_score

# -----------------------
# Step 6: Evaluate Function
# -----------------------
def evaluate_omr(img_path, answer_key_path):
    thresh, original_img = preprocess_image(img_path)
    warped_img = correct_perspective(thresh, original_img)
    warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, warped_thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = get_bubbles(warped_thresh)
    detected_answers = classify_bubbles_to_options(warped_thresh, contours)
    answer_key = load_answer_key(answer_key_path)
    scores, total_score = score_answers(detected_answers, answer_key)
    return scores, total_score

# -----------------------
# Test with sample files
# -----------------------
if __name__ == "__main__":
    img_path = r"C:\Users\User\Documents\OMR_EVALUATOR\data\img1.jpeg"          # your OMR image file
    excel_path = r"C:\Users\User\Documents\OMR_EVALUATOR\data\answer_key.xlsx"   # your answer key file

    print("🔹 Preprocessing image...")
    thresh, img = preprocess_image(img_path)

    print("🔹 Correcting perspective...")
    warped = correct_perspective(thresh, img)

    print("🔹 Detecting bubbles...")
    thresh_warped, _ = preprocess_image(img_path)
    contours = get_bubbles(thresh_warped)

    print("🔹 Classifying answers...")
    detected_answers = classify_bubbles_to_options(thresh_warped, contours)

    print("🔹 Loading answer key...")
    answer_key = load_answer_key(excel_path)

    print("🔹 Scoring...")
    scores, total = score_answers(detected_answers, answer_key)

    print("✅ Scores per subject:", scores)
    print("✅ Total score:", total)
