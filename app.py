import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import math
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Disease Prediction System",
    layout="wide"
)

# ---------------- DARK MODE SAFE CSS ----------------
st.markdown("""
<style>
/* General text visibility fix */
html, body, [class*="css"]  {
    color: inherit !important;
}

/* Card styling */
.disease-card {
    background-color: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1rem 0;
    border-left: 5px solid #4CAF50;
}

/* Header */
.main-header {
    font-size: 2.2rem;
    text-align: center;
    padding: 1rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white !important;
}

/* Buttons */
.stButton>button {
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: bold;
}

/* Text area fix */
textarea {
    color: inherit !important;
}

/* Sidebar fix */
section[data-testid="stSidebar"] {
    color: inherit !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ML ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ---------------- TEXT PREPROCESS ----------------
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    stop_words = {'the','and','or','is','in','to','of','for','with','on','at'}
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# ---------------- SIMILARITY ----------------
def calculate_similarity(text1, text2):
    words1 = text1.split()
    words2 = text2.split()

    if not words1 or not words2:
        return 0.0

    count1 = Counter(words1)
    count2 = Counter(words2)

    all_words = set(words1 + words2)

    dot, mag1, mag2 = 0, 0, 0

    for w in all_words:
        dot += count1[w] * count2[w]
        mag1 += count1[w]**2
        mag2 += count2[w]**2

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (math.sqrt(mag1) * math.sqrt(mag2))

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("Diseases_Symptoms.csv")

# ---------------- FIND MATCHES ----------------
def find_similar(symptoms, df):
    cleaned = preprocess_text(symptoms)
    results = []

    for _, row in df.iterrows():
        sim = calculate_similarity(cleaned, preprocess_text(row["Symptoms"]))
        if sim > 0.1:
            results.append((row["Name"], row["Symptoms"], row["Treatments"], sim))

    results.sort(key=lambda x: x[3], reverse=True)
    return results[:5]

# ---------------- ML PREDICTION ----------------
def ml_predict(symptoms):
    vec = vectorizer.transform([symptoms])
    return model.predict(vec)[0]

# ---------------- SEVERITY ----------------
def classify_severity(text):
    text = text.lower()
    if "chest pain" in text or "breath" in text:
        return "HIGH"
    elif "fever" in text or "vomit" in text:
        return "MEDIUM"
    return "LOW"

# ---------------- MAIN APP ----------------
def main():
    st.markdown('<div class="main-header">🏥 AI Disease Prediction System</div>', unsafe_allow_html=True)

    df = load_data()

    col1, col2 = st.columns([3,1])

    with col1:
        symptoms = st.text_area("Enter your symptoms:")

        if st.button("🔍 Analyze", use_container_width=True):

            if not symptoms:
                st.warning("Please enter symptoms")
                return

            prediction = ml_predict(symptoms)
            results = find_similar(symptoms, df)
            severity = classify_severity(symptoms)

            st.subheader("🤖 ML Prediction")
            st.success(prediction)

            st.subheader("⚠️ Severity Level")
            st.write(severity)

            if severity == "HIGH":
                st.error("🚨 Seek immediate medical attention!")

            st.subheader("🔍 Similar Diseases")

            for name, sym, treat, sim in results:
                st.markdown(f"""
                <div class="disease-card">
                    <h4>{name} ({sim*100:.1f}%)</h4>
                    <p><b>Symptoms:</b> {sym}</p>
                    <p><b>Treatment:</b> {treat}</p>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.info("AI-based symptom analysis system using ML + NLP")

if __name__ == "__main__":
    main()
