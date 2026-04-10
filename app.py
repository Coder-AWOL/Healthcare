# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import math

# Set page config
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    .disease-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .similarity-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .category-badge {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.8rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove special characters
    words = text.split()
    # Common stop words
    stop_words = {'the', 'and', 'or', 'but', 'is', 'in', 'it', 'to', 'of', 'for', 
                 'with', 'on', 'at', 'by', 'an', 'a', 'this', 'that', 'are', 'was',
                 'has', 'have', 'had', 'be', 'been', 'being'}
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# Disease categorization
def categorize_disease(disease_name):
    disease_name = disease_name.lower()
    categories = {
        'Cancer': ['cancer', 'carcinoma', 'tumor', 'lymphoma', 'leukemia', 'melanoma'],
        'Infection': ['infection', 'itis', 'abscess', 'fever', 'pox', 'flu', 'sepsis'],
        'Syndrome': ['syndrome', 'disorder'],
        'Deficiency': ['deficiency', 'anemia', 'avitaminosis'],
        'Poisoning': ['poisoning', 'overdose', 'intoxication', 'toxicity'],
        'Injury': ['fracture', 'injury', 'dislocation', 'sprain', 'trauma', 'wound'],
        'Eye Condition': ['glaucoma', 'cataract', 'vision', 'eye', 'retina', 'cornea'],
        'Cardiovascular': ['heart', 'cardio', 'hypertension', 'blood pressure', 'artery', 'vein'],
        'Endocrine': ['diabetes', 'thyroid', 'hormone', 'metabolic', 'gland'],
        'Neurological': ['neuro', 'brain', 'nerve', 'neural', 'cephal', 'psych'],
        'Respiratory': ['lung', 'pulmonary', 'breath', 'respiratory', 'asthma'],
        'Gastrointestinal': ['gastro', 'stomach', 'intestinal', 'colon', 'liver', 'pancreas'],
        'Pain Condition': ['arthritis', 'pain', 'ache', 'neuralgia', 'migraine']
    }
    
    for category, keywords in categories.items():
        if any(keyword in disease_name for keyword in keywords):
            return category
    return 'Other'

# Similarity calculation
def calculate_similarity(text1, text2):
    words1 = text1.split()
    words2 = text2.split()
    
    if not words1 or not words2:
        return 0.0
    
    count1 = Counter(words1)
    count2 = Counter(words2)
    
    all_words = set(words1 + words2)
    
    dot_product = 0
    mag1 = 0
    mag2 = 0
    
    for word in all_words:
        dot_product += count1[word] * count2[word]
        mag1 += count1[word] ** 2
        mag2 += count2[word] ** 2
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (math.sqrt(mag1) * math.sqrt(mag2))

# Load data with caching
@st.cache_data
def load_data():
    """Load the disease data from CSV"""
    try:
        df = pd.read_csv('Diseases_Symptoms.csv')
        st.success("✅ Disease database loaded successfully!")
        return df
    except FileNotFoundError:
        st.error("❌ CSV file not found. Please make sure 'Diseases_Symptoms.csv' is in the same directory.")
        return pd.DataFrame()

# Find similar diseases
def find_similar_diseases(symptoms, df, top_n=10):
    cleaned_symptoms = preprocess_text(symptoms)
    
    if not cleaned_symptoms:
        return []
    
    results = []
    for _, row in df.iterrows():
        disease_symptoms = preprocess_text(row['Symptoms'])
        similarity = calculate_similarity(cleaned_symptoms, disease_symptoms)
        
        if similarity > 0.1:  # Only include meaningful matches
            results.append({
                'disease': row['Name'],
                'symptoms': row['Symptoms'],
                'treatments': row['Treatments'],
                'similarity': similarity,
                'category': categorize_disease(row['Name'])
            })
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]

# Main application
def main():
    st.markdown('<h1 class="main-header">🏥 AI Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Database Statistics")
        st.metric("Total Diseases", len(df))
        
        # Category distribution
        df['Category'] = df['Name'].apply(categorize_disease)
        st.write("**Category Distribution:**")
        for category, count in df['Category'].value_counts().items():
            st.write(f"• {category}: {count}")
        
        st.header("💡 Example Symptoms")
        examples = [
            "headache fever fatigue",
            "chest pain shortness of breath", 
            "joint pain swelling stiffness",
            "abdominal pain nausea vomiting",
            "skin rash itching redness"
        ]
        
        for example in examples:
            if st.button(example, key=example):
                st.session_state.symptoms_input = example
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Symptoms input
        symptoms_input = st.text_area(
            "**Describe your symptoms:**",
            height=120,
            placeholder="Enter symptoms separated by commas (e.g., headache, fever, fatigue, cough...)",
            value=st.session_state.get('symptoms_input', ''),
            help="Be as specific as possible for better results"
        )
        
        # Analyze button
        if st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True):
            if not symptoms_input.strip():
                st.error("⚠️ Please enter some symptoms to analyze")
            else:
                with st.spinner("🔬 Analyzing symptoms and searching database..."):
                    results = find_similar_diseases(symptoms_input, df)
                
                if not results:
                    st.warning("❌ No significant matches found. Try different symptoms or be more specific.")
                else:
                    st.success(f"✅ Found {len(results)} potential matches")
                    
                    # Display results
                    for i, disease in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="disease-card">
                                <h3>{i}. {disease['disease']} 
                                <span class="similarity-badge">{disease['similarity']*100:.1f}% match</span>
                                </h3>
                                <p><strong>📋 Category:</strong> <span class="category-badge">{disease['category']}</span></p>
                                <p><strong>🧬 Symptoms:</strong> {disease['symptoms']}</p>
                                <p><strong>💊 Treatments:</strong> {disease['treatments']}</p>
                            </div>
                            """, unsafe_allow_html=True)
    
    with col2:
        st.header("ℹ️ About")
        st.info("""
        This AI-powered system analyzes your symptoms and matches them 
        with diseases in our medical database using advanced text similarity algorithms.
        """)
        
        st.header("📈 Top Symptoms")
        # Analyze common symptoms in database
        all_symptoms = ' '.join(df['Symptoms'].astype(str)).lower()
        words = [word for word in all_symptoms.split() if len(word) > 4]
        common_words = Counter(words).most_common(10)
        
        for word, count in common_words:
            st.write(f"• {word}: {count}")
    
    # Data exploration section
    with st.expander("🔍 Explore Disease Database"):
        st.dataframe(df[['Name', 'Symptoms', 'Treatments']], use_container_width=True)
        
        # Search functionality
        search_term = st.text_input("Search diseases:", placeholder="Enter disease name...")
        if search_term:
            filtered_df = df[df['Name'].str.contains(search_term, case=False)]
            st.dataframe(filtered_df, use_container_width=True)
    
    
   
if __name__ == "__main__":
    main()