#!/usr/bin/env python3
import streamlit as st
import joblib

@st.cache_resource
def load_model(path='models/spam_model.joblib'):
    return joblib.load(path)

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")
st.title("📧 Spam Email Classifier")
st.write("Enter an email below and the model will predict whether it is **spam** or **ham**.")

model_path = 'models/spam_model.joblib'
try:
    pipe = load_model(model_path)
    model_ready = True
except Exception as e:
    model_ready = False
    st.warning("Model not found. Please run `python train.py` first to create `models/spam_model.joblib`.")

email_text = st.text_area("Email content", height=200, placeholder="Paste email text here...")
if st.button("Predict"):
    if not model_ready:
        st.error("Model not loaded. Train the model first.")
    elif not email_text.strip():
        st.warning("Please enter some text.")
    else:
        pred = pipe.predict([email_text])[0]
        st.success(f"Prediction: **{pred.upper()}**")
