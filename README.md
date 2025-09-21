📄 OMR Evaluator
🔹 Problem Statement
In many exams, OMR (Optical Mark Recognition) sheets are used to collect student answers.
Manually checking these sheets is time-consuming, error-prone, and inefficient.
The need is for a system that can:
->Automatically detect filled bubbles in OMR sheets
->Compare them against a predefined answer key
->Provide scores quickly and accurately
->An automated OMR (Optical Mark Recognition) evaluation system that scans, processes, and scores answer sheets from images.
->Built with Streamlit (frontend) and OpenCV, NumPy, and Pandas (backend).
🚀 Features
*Upload OMR sheet image (.jpg, .jpeg, .png)
*Upload Answer Key (.xlsx)
*Automatic bubble detection & classification using OpenCV
*Scoring & evaluation with subject-wise breakdown
*Download results as CSV
*Visual bubble overlay showing detected answers
*Clean, responsive UI powered by Streamlit
🛠️ Tech Stack
Frontend: Streamlit, PIL, Pandas
Backend: Python, OpenCV, NumPy
Data Handling: Pandas, OpenPyXL
Deployment: Streamlit Cloud
🔹 Approach
Our solution combines computer vision and data handling to automate OMR evaluation:
1.Image Preprocessing
   .Noise removal, thresholding, and perspective correction using OpenCV
2.Bubble Detection
  .Contour extraction and classification of marked answers
3.Answer Matching
  .Comparing detected responses with the Excel-based answer key
4.Result Generation
   .Subject-wise scoring, total score calculation
5.Export results to CSV
   .Visual overlay of detected bubbles for verification
5.Frontend (Streamlit)
  .Simple UI for uploading sheets & answer keys
  .Interactive results and downloads
 🔹 Installation Steps
Clone the repository:
      git clone https://github.com/your-username/omr_evaluator.git
      cd omr_evaluator
Create and activate a virtual environment:
  python -m venv venv
  source venv/bin/activate      # On Windows: venv\Scripts\activate
Install dependencies:
   pip install -r requirements.txt
Run the application:
  streamlit run frontend/app.py
  🔹 Usage
Open the app in your browser (Streamlit will show a local URL).
Upload:
An OMR sheet image (.jpg, .jpeg, .png)
The Answer Key (.xlsx)
Wait for the system to process the image.
View results:
  .Subject-wise and total scores
  .Bubble overlay visualization
  .Download results as CSV
📦 Requirements
See requirements.txt
🤝 Contributing
Contributions, issues, and feature requests are welcome!
Feel free to fork this repo and open a PR.
✨ Built with ❤️ using Python, OpenCV, and Streamlit
