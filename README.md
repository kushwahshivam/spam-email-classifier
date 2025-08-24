# Spam Email Classifier (Minor Project)

An end-to-end mini project that trains a simple machine learning model to classify emails as **spam** or **ham** (not spam).

## Project Structure
```
spam-email-classifier/
├── data/
│   └── sample_dataset.csv         # tiny sample; replace with a larger dataset
├── models/
│   └── (created after training)   # saved model pipeline
├── train.py                       # training + evaluation script
├── predict.py                     # CLI prediction script
├── app.py                         # Streamlit demo app
├── requirements.txt
└── README.md
```

## Quick Start

1. **Create a virtual environment & install dependencies**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

2. **Prepare your dataset**
- Put a CSV file at `data/dataset.csv` with **two columns**: `text` and `label`.
- `label` must be either `spam` or `ham`.
- You can start with `data/sample_dataset.csv` (tiny) and then replace it with a bigger dataset (e.g., from Kaggle).

3. **Train the model**
```bash
python train.py --data data/dataset.csv --model_dir models
```
This will:
- Split into train/test
- Train a TF-IDF + Logistic Regression pipeline
- Save the trained pipeline to `models/spam_model.joblib`
- Print accuracy, precision, recall, F1, and confusion matrix

4. **Run quick predictions from CLI**
```bash
python predict.py --model models/spam_model.joblib --text "Win a FREE iPhone now! Click here"
```

5. **Run the Streamlit web app**
```bash
streamlit run app.py
```
Then open the local URL shown in the terminal.

## Tips for a Good Minor Project Report
- **Problem Statement**: Classify emails as spam/ham to improve user safety and save time.
- **Dataset**: Briefly describe source, size, class balance, and preprocessing.
- **Methodology**: TF-IDF features + Logistic Regression (why: fast, strong baseline).
- **Evaluation**: Accuracy + Precision/Recall/F1; add confusion matrix.
- **Result**: Report metrics and show a few example predictions.
- **Extensions** (optional):
  - Try **Multinomial Naive Bayes** or **Linear SVM**.
  - Add bigrams (`ngram_range=(1,2)` already enabled).
  - Use **class weights** if your dataset is imbalanced.
  - Add **model versioning** and **ROC/PR curves** (if using a classifier with probabilities).

Good luck!
