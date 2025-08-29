Spam Email Classifier Project
Project Overview

This project classifies emails as spam or ham (not spam) using Machine Learning. It includes a training pipeline, a Flask backend, and a web-based frontend for real-time predictions. The goal is to help users automatically identify unwanted emails and improve email security.

Business Requirements & KPIs

Spam Detection Accuracy – Correctly identifying spam emails

Precision – Proportion of correctly predicted spam emails

Recall – Proportion of actual spam emails correctly detected

F1-Score – Harmonic mean of precision and recall

Confusion Matrix – True Positive, True Negative, False Positive, False Negative

Granular Analysis

Classification by Keyword Patterns – How certain words influence predictions

Performance by Email Length – Compare short vs long emails

Dataset Balance Analysis – Check spam/ham ratio

Model Evaluation Metrics – Accuracy, Precision, Recall, F1-score

Interactive Frontend Testing – User can input email text and get prediction

Tools & Technologies

Python → Data preprocessing, model training, prediction (Pandas, scikit-learn, Joblib)

Flask + Flask-CORS → Web backend and API

HTML, CSS, JavaScript → Frontend interface for email input and predictions

Streamlit (optional) → Alternative interactive demo

Joblib → Saving and loading trained models

Dashboard / Frontend Preview

Web Interface: Enter an email text and get instant spam/ham prediction

CLI: Command-line testing for quick predictions

Insights & Findings

Shorter emails with certain trigger words are often classified as spam

Classifier can achieve high precision for the current dataset

TF-IDF + Logistic Regression provides a simple yet effective baseline

Performance can improve with larger datasets and advanced preprocessing

Folder Structure
spam-email-classifier-full/
│
├── data/                 # Dataset CSV (text,label)
├── backend/              # Flask backend + training scripts
│   ├── train.py
│   ├── predict.py
│   ├── app.py
│   ├── run_all.py        # optional: train + run backend together
│   └── models/           # Trained model & vectorizer
├── frontend/             # Web interface
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

How to Use

Clone the repository

git clone <your-repo-url>
cd spam-email-classifier-full


Set up virtual environment

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r backend/requirements.txt


Prepare Dataset

Place dataset.csv in data/ folder

CSV must have columns: text,label

Labels must be either spam or ham

Train the Model

cd backend
python train.py --data ../data/dataset.csv --model_dir models


Predict from CLI

python predict.py --model models/spam_model.pkl --text "Win a FREE iPhone now! Click here"


Run Web App

cd backend
python app.py


Open browser: http://127.0.0.1:5000

Enter email text → click Predict → see spam/ham

Future Improvements

Use Multinomial Naive Bayes or Linear SVM for higher accuracy

Add bigrams/trigrams for better TF-IDF representation

Automate dataset preprocessing pipeline

Deploy backend on Render / Heroku / AWS and frontend on Netlify / Vercel

Add probability scores and confidence levels for predictions

Author

👨‍💻 Shivam Kushwah
🔗 LinkedIn : https://www.linkedin.com/in/kushwahshivam/

📧 Email: your-email@example.com

✨ This project demonstrates a complete spam detection pipeline with ML, web integration, and deployment-ready architecture.
