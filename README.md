# 📄 OMR Evaluator

## 🔹 Problem Statement
In many exams, OMR (Optical Mark Recognition) sheets are used to collect student answers.  
Manually checking these sheets is time-consuming, error-prone, and inefficient.  

The need is for a system that can:
- Automatically detect filled bubbles in OMR sheets  
- Compare them against a predefined answer key  
- Provide scores quickly and accurately  

**Our solution:** An automated OMR evaluation system that scans, processes, and scores answer sheets from images.  
Built with **Streamlit (frontend)** and **OpenCV, NumPy, Pandas (backend)**.

---

## 🚀 Features
- Upload OMR sheet image (`.jpg`, `.jpeg`, `.png`)  
- Upload Answer Key (`.xlsx`)  
- Automatic bubble detection & classification using OpenCV  
- Scoring & evaluation with subject-wise breakdown  
- Download results as CSV  
- Visual bubble overlay showing detected answers  
- Clean, responsive UI powered by Streamlit  

---

## 🛠️ Tech Stack
**Frontend:** Streamlit, PIL, Pandas  
**Backend:** Python, OpenCV, NumPy  
**Data Handling:** Pandas, OpenPyXL  
**Deployment:** Streamlit Cloud  

---

## 🔹 Approach
Our solution combines computer vision and data handling to automate OMR evaluation:

1. **Image Preprocessing**  
   - Noise removal, thresholding, and perspective correction using OpenCV  

2. **Bubble Detection**  
   - Contour extraction and classification of marked answers  

3. **Answer Matching**  
   - Comparing detected responses with the Excel-based answer key  

4. **Result Generation**  
   - Subject-wise scoring, total score calculation  

5. **Export Results**  
   - Generate CSV file with scores  
   - Visual overlay of detected bubbles for verification  

6. **Frontend (Streamlit)**  
   - Simple UI for uploading sheets & answer keys  
   - Interactive results and downloads  

---

## 🔹 Installation Steps
Clone the repository:
```bash
git clone https://github.com/your-username/omr_evaluator.git
cd omr_evaluator

## create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


