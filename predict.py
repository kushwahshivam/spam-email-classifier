#!/usr/bin/env python3
import argparse
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='models/spam_model.joblib', help='Path to saved model')
    ap.add_argument('--text', type=str, required=True, help='Email text to classify')
    args = ap.parse_args()

    pipe = joblib.load(args.model)
    pred = pipe.predict([args.text])[0]
    print(f"Prediction: {pred}")

if __name__ == '__main__':
    main()
