# Spam Email Classifier (Minor Project)

An end-to-end mini project that trains a **machine learning model** to classify emails as **spam** or **ham** (not spam), with a **web-based frontend** for testing.

---

## Project Structure

spam-email-classifier-full/
├── data/
│   └── dataset.csv               # sample dataset with 'text' and 'label' columns
├── backend/
│   ├── train.py                  # trains TF-IDF + Logistic Regression
│   ├── predict.py                # CLI prediction helper
│   ├── app.py                    # Flask backend + serves frontend
│   ├── run_all.py                # optional: train + run backend in one command
│   └── models/                   # trained model and vectorizer files
├── frontend/
│   ├── index.html                # web interface
│   ├── style.css                 # styling
│   └── script.js                 # JS to call backend API
├── requirements.txt              # Python dependencies
└── README.md

---

## Quick Start

### 1️⃣ Set up virtual environment

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

---

### 2️⃣ Prepare dataset

- Place your dataset CSV at `data/dataset.csv`  
- **CSV format**: two columns → `text,label`  
- **Labels** must be exactly `spam` or `ham` (lowercase)  
- You can start with the small sample dataset provided, then expand for better accuracy.

Example:

label,text
ham,"Hey! Are we still meeting for lunch today?"
spam,"Win a brand new car! Click here to claim your prize now."

---

### 3️⃣ Train the model

cd backend
python train.py --data ../data/dataset.csv --model_dir models

This will:

- Split data into train/test sets
- Train a **TF-IDF + Logistic Regression** pipeline
- Save the trained pipeline to `backend/models/`  
  (`spam_model.pkl` and `vectorizer.pkl`)
- Print evaluation metrics (accuracy, precision, recall, F1, confusion matrix)

---

### 4️⃣ Run backend + frontend

#### Option A: Run separately

cd backend
python app.py

Open browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)

#### Option B: Run combined (train + backend automatically)

cd backend
python run_all.py

Frontend will load automatically, and you can classify emails directly.

---

### 5️⃣ Quick predictions from CLI

python predict.py --model models/spam_model.pkl --text "Win a FREE iPhone now! Click here"

---

## Project Details

### Methodology

- **Features**: TF-IDF vectors of email text  
- **Classifier**: Logistic Regression (fast, strong baseline)  
- **Preprocessing**: lowercase, strip spaces  

---

### Evaluation

- Metrics reported: Accuracy, Precision, Recall, F1-score  
- Confusion matrix included  
- Works best with a **larger dataset**; small sample datasets may misclassify some emails.

---

### Possible Extensions

- Use **Multinomial Naive Bayes** or **Linear SVM**  
- Include **bigrams or trigrams** in TF-IDF (`ngram_range=(1,2)`)  
- Handle **imbalanced datasets** with class weights  
- Add **probability scores** for spam predictions  
- Add **ROC/PR curves** for evaluation  

---

### Notes

- Make sure your virtual environment is active before running scripts  
- Ensure dataset encoding is **UTF-8**  
- Frontend communicates with backend via **Flask API** (`/api/predict`)  
