import cv2
import numpy as np

# =========================
# Step 2: Preprocessing
# =========================
def preprocess_image(img_path):
    """
    Preprocess an OMR sheet image:
    1. Convert to grayscale
    2. Apply Gaussian blur
    3. Apply thresholding
    Returns thresholded image and original image
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh, img

# =========================
# Step 2.5: Perspective Correction
# =========================
def correct_perspective(thresh_img, original_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if len(contours) == 0:
        return original_img

    sheet_contour = contours[0]
    peri = cv2.arcLength(sheet_contour, True)
    approx = cv2.approxPolyDP(sheet_contour, 0.02 * peri, True)
    
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
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

# =========================
# Step 3: Bubble Detection
# =========================
def get_bubbles(thresh_img, min_area=100, max_area=1500):
    """
    Detect bubble contours in thresholded image.
    Filters by area to remove noise.
    Returns list of contours sorted top-to-bottom, left-to-right per row
    """
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

    # Sort contours by vertical position
    bubble_contours = sorted(bubble_contours, key=lambda c: cv2.boundingRect(c)[1])

    # Optional: refine sorting within rows (left-to-right)
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

# =========================
# Step 4: Bubble Classification
# =========================
def classify_bubbles_to_options(thresh_img, contours, options_per_question=4, fill_thresh=0.5):
    """
    For each question, determines which option (a,b,c,d) is filled
    Returns list of answers. If multiple bubbles filled, returns 'ambiguous'.
    """
    filled_options = []

    for i in range(0, len(contours), options_per_question):
        question_contours = contours[i:i+options_per_question]
        selected_option = None
        multiple_filled = False

        for idx, c in enumerate(question_contours):
            x, y, w, h = cv2.boundingRect(c)
            roi = thresh_img[y:y+h, x:x+w]
            filled_ratio = cv2.countNonZero(roi) / (w * h)

            if filled_ratio > fill_thresh:
                if selected_option is None:
                    selected_option = chr(ord('a') + idx)
                else:
                    multiple_filled = True

        if multiple_filled:
            filled_options.append('ambiguous')
        else:
            filled_options.append(selected_option)  # None if no bubble filled

    return filled_options

import pandas as pd

def load_answer_key(excel_path):
    """
    Reads Excel answer key.
    Returns a dictionary: {subject_name: [answers]}
    """
    df = pd.read_excel(excel_path)
    answer_key = {}
    for col in df.columns:
        answer_key[col] = df[col].tolist()
    return answer_key


def score_answers(detected_answers, answer_key, questions_per_subject=20):
    """
    detected_answers: list of all answers in order
    answer_key: dict {subject: [correct_answers]}
    
    Returns: dict of {subject: score}, total score
    """
    scores = {}
    total_score = 0
    num_subjects = len(answer_key)
    
    start_idx = 0
    for subject, correct in answer_key.items():
        subject_answers = detected_answers[start_idx:start_idx+questions_per_subject]
        subject_score = 0
        for det, ans in zip(subject_answers, correct):
            if det == ans:
                subject_score += 1  # 1 mark per correct answer
        scores[subject] = subject_score
        total_score += subject_score
        start_idx += questions_per_subject
    
    return scores, total_score


