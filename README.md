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

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd spam-email-classifier-full
