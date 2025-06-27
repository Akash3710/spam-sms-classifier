import streamlit as st
import pickle
import pandas as pd

# Load model and vectorizer
try:
    with open("model/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("model/vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)

except FileNotFoundError:
    st.error("❌ Model or vectorizer not found. Make sure files are correctly placed.")
    st.stop()

# ---------- Page Config & Custom CSS ----------
st.set_page_config(page_title="Spam SMS Classifier", page_icon="📩", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .stTextArea textarea {
        font-size: 16px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- App Layout ----------
st.markdown("<h1 style='text-align: center;'>📩 Spam Message Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Check whether a message is spam or not using a trained ML model</p>",
    unsafe_allow_html=True)
st.markdown("---")

# Input section
message = st.text_area("✉️ Enter your SMS message here:", height=150)

# Predict
if st.button("🔍 Predict"):
    if message.strip():
        transformed = vectorizer.transform([message])
        prediction = model.predict(transformed)[0]
        result = "🚫 Spam" if prediction == 1 else "✅ Not Spam"

        # Show result with color
        if prediction == 1:
            st.error(f"**Prediction:** {result}")
        else:
            st.success(f"**Prediction:** {result}")
    else:
        st.warning("Please enter a message first.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>Built with ❤️ using Streamlit | © 2025 Spam Detector</p>",
            unsafe_allow_html=True)
