#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')  # <-- fix encoding
    df = df.rename(columns={'v2':'text', 'v1':'label'})  # <-- fix column names if needed
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")
    df = df.dropna(subset=['text', 'label']).copy()
    # Strip spaces and lowercase
    df['label'] = df['label'].astype(str).str.strip().str.lower()

# Map any common alternatives to spam/ham
    df['label'] = df['label'].replace({
    '1':'spam',
    '0':'ham',
    'junk':'spam',
    'non-spam':'ham'
    })
    df = df[df['label'].isin(['spam','ham'])]

    return df


def build_pipeline(class_weight=None):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1,2),
            min_df=2
        )),
        ('clf', LogisticRegression(
            max_iter=200,
            class_weight=class_weight,
            solver='liblinear'
        ))
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='data/sample_dataset.csv', help='Path to dataset CSV with columns: text,label')
    ap.add_argument('--model_dir', type=str, default='models', help='Directory to save trained model')
    ap.add_argument('--test_size', type=float, default=0.2, help='Test split size')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    df = load_data(args.data)
    X = df['text'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    import numpy as np
    # Compute class weights (useful if imbalanced)
    classes = np.array(['ham', 'spam'])  # must be a numpy array
    class_weight_vals = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
    cw = {cls: w for cls, w in zip(classes, class_weight_vals)}

    pipe = build_pipeline(class_weight=cw)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='spam')
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print("=== Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion Matrix [rows=true, cols=pred] (labels: ham, spam):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    model_path = os.path.join(args.model_dir, 'spam_model.joblib')
    joblib.dump(pipe, model_path)
    print(f"\nSaved model to: {model_path}")

if __name__ == '__main__':
    main()
