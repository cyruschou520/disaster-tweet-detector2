# ================================================================
# COMPLETE DISASTER TWEET AI DETECTOR DASHBOARD
# ================================================================

import streamlit as st
import pandas as pd
import requests
import urllib.parse
import json
import re
import math
import time
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import deque
from statistics import mean, stdev
import hashlib
import base64
import random
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import nltk
from nltk.corpus import stopwords

# ================================================================
# FIREBASE IMPORTS (with graceful fallback)
# ================================================================

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Disaster Tweet AI Detector - HuggingFace BERT",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌪️"
)

# ================================================================
# ENHANCED CSS WITH MODERN DESIGN
# ================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
    }
    
    /* Main container with glass morphism */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        margin: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Header with disaster theme gradient */
    .main-header {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 50%, #48bb78 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.4);
        animation: slideDown 0.5s ease-out;
    }
    
    @keyframes slideDown {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* HuggingFace badge */
    .huggingface-badge {
        background: linear-gradient(135deg, #6e40ff, #a371f7);
        color: white;
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 1em;
        font-weight: 600;
        display: inline-block;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(110, 64, 255, 0.4);
        animation: pulse 2s infinite;
    }
    
    .bert-badge {
        background: linear-gradient(135deg, #667eea, #5a67d8);
        color: white;
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 1em;
        font-weight: 600;
        display: inline-block;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.9; transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Alert animations */
    .fake-alert, .real-alert, .bert-alert, .disaster-alert {
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-size: 1.2em;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        animation: pulse 2s infinite, slideIn 0.5s ease-out;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    .disaster-alert {
        background: linear-gradient(135deg, #ff6b6b, #feca57);
    }
    
    .bert-alert {
        background: linear-gradient(135deg, #667eea, #5a67d8);
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .fake-alert {
        background: linear-gradient(135deg, #ff6b6b, #ee5253);
    }
    
    .real-alert {
        background: linear-gradient(135deg, #10ac84, #1dd1a1);
    }
    
    /* Probability bar with gradient */
    .probability-bar-container {
        background: rgba(0,0,0,0.05);
        border-radius: 15px;
        padding: 5px;
        margin: 20px 0;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .probability-bar {
        height: 40px;
        border-radius: 12px;
        overflow: hidden;
        display: flex;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .fake-bar {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b, #ee5253);
        color: white;
        text-align: center;
        line-height: 40px;
        font-weight: bold;
        transition: width 0.5s ease-in-out;
    }
    
    .real-bar {
        height: 100%;
        background: linear-gradient(90deg, #10ac84, #1dd1a1);
        color: white;
        text-align: center;
        line-height: 40px;
        font-weight: bold;
        transition: width 0.5s ease-in-out;
    }
    
    /* Metric cards with hover effects */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 107, 107, 0.2);
        margin: 10px 0;
        animation: fadeIn 0.5s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(255, 107, 107, 0.3);
        border-color: #ff6b6b;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .metric-card h4 {
        color: #666;
        font-size: 0.9em;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card h2 {
        font-size: 2em;
        margin: 0;
        font-weight: 700;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 30px;
        font-size: 0.9em;
        font-weight: 600;
        margin: 5px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-out;
    }
    
    .live-badge {
        background: linear-gradient(135deg, #ff6b6b, #ee5253);
        color: white;
    }
    
    .local-badge {
        background: linear-gradient(135deg, #95a5a6, #7f8c8d);
        color: white;
    }
    
    .huggingface-model-badge {
        background: linear-gradient(135deg, #6e40ff, #a371f7);
        color: white;
    }
    
    .model-badge {
        background: linear-gradient(135deg, #667eea, #5a67d8);
        color: white;
    }
    
    /* Input area with modern design */
    .stTextArea textarea {
        border-radius: 15px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 15px !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        background: white !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05) !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #ff6b6b !important;
        box-shadow: 0 5px 20px rgba(255, 107, 107, 0.3) !important;
        transform: scale(1.02);
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 12px !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border: none !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(255, 107, 107, 0.4) !important;
    }
    
    /* Primary button */
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #ff6b6b, #feca57) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-163ttbj {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%) !important;
    }
    
    .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 20px;
        margin: 20px;
    }
    
    /* Live feed table */
    .dataframe {
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1) !important;
    }
    
    /* Alert container */
    .alert-container {
        background: linear-gradient(135deg, #fff5f5, #fff0f0);
        border-left: 5px solid #ff6b6b;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        animation: slideInRight 0.5s ease-out;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Live indicator */
    .live-dot {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #10ac84;
        border-radius: 50%;
        margin-right: 8px;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(255, 107, 107, 0.3);
    }
    
    .stat-value {
        font-size: 2.5em;
        font-weight: 700;
        color: #ff6b6b;
        margin: 10px 0;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress bar */
    .progress-container {
        width: 100%;
        height: 10px;
        background: #f0f0f0;
        border-radius: 5px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b, #feca57);
        border-radius: 5px;
        transition: width 0.5s ease-in-out;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5em;
        font-weight: 700;
        color: #333;
        margin: 30px 0 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #ff6b6b;
        display: inline-block;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            padding: 15px;
            margin: 10px;
        }
        
        .metric-card h2 {
            font-size: 1.5em;
        }
        
        .stat-value {
            font-size: 2em;
        }
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# DOWNLOAD NLTK DATA
# ================================================================
@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))

stop_words = download_nltk_data()

# ================================================================
# FIREBASE INITIALIZATION (with graceful fallback)
# ================================================================

def initialize_firebase():
    """Initialize Firebase with fallback to local mode"""
    if not FIREBASE_AVAILABLE:
        return None, False
    
    if not firebase_admin._apps:
        try:
            # Try to get credentials from Streamlit secrets
            if "firebase" in st.secrets:
                firebase_config = {
                    "type": st.secrets["firebase"]["type"],
                    "project_id": st.secrets["firebase"]["project_id"],
                    "private_key_id": st.secrets["firebase"]["private_key_id"],
                    "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
                    "client_email": st.secrets["firebase"]["client_email"],
                    "client_id": st.secrets["firebase"]["client_id"],
                    "auth_uri": st.secrets["firebase"]["auth_uri"],
                    "token_uri": st.secrets["firebase"]["token_uri"]
                }
                cred = credentials.Certificate(firebase_config)
                firebase_admin.initialize_app(cred)
                db = firestore.client()
                return db, True
            else:
                return None, False
        except Exception as e:
            return None, False
    else:
        return firestore.client(), True

# Initialize Firebase
db, FIREBASE_ACTIVE = initialize_firebase()

# ================================================================
# SESSION STATE INITIALIZATION
# ================================================================
if "session_id" not in st.session_state:
    st.session_state["session_id"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "huggingface_bert"
if "input_key_counter" not in st.session_state:
    st.session_state["input_key_counter"] = 0
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True
if "local_analyses" not in st.session_state:
    st.session_state["local_analyses"] = []
if "local_stats" not in st.session_state:
    st.session_state["local_stats"] = {
        "total_analyses": 0,
        "total_fake": 0,
        "total_real": 0,
        "locations": {},
        "disaster_types": {},
        "models_used": {}
    }

# ================================================================
# CONSTANTS
# ================================================================
MALAYSIA_LOCATIONS = [
    'Kampar', 'Ipoh', 'Kuala Lumpur', 'KL', 'Penang', 'Pulau Pinang',
    'Johor', 'Johor Bahru', 'Shah Alam', 'Selangor', 'Perak', 'Kedah',
    'Kelantan', 'Terengganu', 'Pahang', 'Negeri Sembilan', 'Melaka',
    'Sabah', 'Sarawak', 'Langkawi', 'Kuantan', 'Kota Bharu', 'Alor Setar',
    'George Town', 'Butterworth', 'Taiping', 'Petaling Jaya', 'Subang Jaya',
    'Klang', 'Putrajaya', 'Cameron Highlands', 'Kota Kinabalu', 'Kuching'
]

DISASTER_KEYWORDS = {
    "flood": ['flood', 'banjir', 'inundation', 'water level', 'banjir kilat'],
    "earthquake": ['earthquake', 'gempa', 'tremor', 'seismic', 'gempa bumi'],
    "storm": ['storm', 'thunderstorm', 'ribut', 'petir', 'lightning', 'ribut petir'],
    "landslide": ['landslide', 'tanah runtuh', 'mudslide', 'runtuh'],
    "fire": ['fire', 'kebakaran', 'burning', 'api'],
    "tsunami": ['tsunami', 'tidal wave', 'gelombang tsunami'],
    "wind": ['wind', 'angin', 'gale', 'tornado', 'angin kencang'],
    "haze": ['haze', 'jerebu', 'smoke', 'asap'],
    "drought": ['drought', 'kemarau', 'dry spell'],
    "heatwave": ['heatwave', 'gelombang panas', 'heat stroke']
}

# ================================================================
# HUGGINGFACE BERT MODEL LOADING (EASIER FOR DEPLOYMENT)
# ================================================================

@st.cache_resource(show_spinner="🔄 Loading BERT model from HuggingFace Hub...")
def load_huggingface_bert():
    """
    Load BERT model from HuggingFace Hub
    This is the recommended approach for deployment - no large files in GitHub!
    """
    try:
        # TODO: Replace with your HuggingFace model ID after uploading
        # Format: "username/model-name"
        MODEL_NAME = "cyruschou520/bert-disaster-model"  # You need to upload your model here
        
        with st.spinner(f"Downloading model from HuggingFace Hub (this may take a minute)..."):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            model.eval()
            
        st.sidebar.success("✅ BERT model loaded from HuggingFace Hub!")
        return model, tokenizer, True
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load from HuggingFace: {e}")
        st.sidebar.info("Falling back to local model if available...")
        
        # Fallback to local model
        try:
            local_path = "bert_disaster_model_fine_tuned"
            if os.path.exists(local_path):
                tokenizer = AutoTokenizer.from_pretrained(local_path)
                model = AutoModelForSequenceClassification.from_pretrained(local_path)
                model.eval()
                st.sidebar.success("✅ Loaded local BERT model")
                return model, tokenizer, True
        except:
            pass
        
        return None, None, False

# Load the HuggingFace BERT model
bert_model, bert_tokenizer, bert_loaded = load_huggingface_bert()

# ================================================================
# LOAD LOGISTIC REGRESSION MODEL (as backup)
# ================================================================

@st.cache_resource
def load_lr_model():
    """Load trained Logistic Regression model as backup"""
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        lr_model = joblib.load('logistic_regression_model.pkl')
        return vectorizer, lr_model, True
    except:
        return None, None, False

vectorizer, lr_model, lr_loaded = load_lr_model()

# ================================================================
# PREPROCESSING FUNCTION
# ================================================================

def preprocess_tweet(text):
    """Clean and preprocess tweet text (same as training)"""
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# ================================================================
# BERT PREDICTION FUNCTION (HuggingFace version)
# ================================================================

def predict_with_bert(text):
    """Predict using BERT model from HuggingFace"""
    if not bert_loaded or bert_model is None:
        return None
    
    start_time = time.time()
    
    # Preprocess
    clean_text = preprocess_tweet(text)
    
    if not clean_text:
        return {
            "is_fake": False,
            "fake_probability": 0.5,
            "real_probability": 0.5,
            "confidence": 0.5,
            "reasons": ["Empty tweet after preprocessing"],
            "detected_disasters": [],
            "model_used": "BERT (HuggingFace)",
            "response_time": time.time() - start_time,
            "word_count": 0,
            "exclamation_count": 0,
            "question_count": 0,
            "caps_words": 0,
            "has_url": False,
            "number_count": 0
        }
    
    # Tokenize
    inputs = bert_tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    # Predict
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        prob_not_disaster = probabilities[0][0].item()
        prob_disaster = probabilities[0][1].item()
    
    # Detect disaster types
    text_lower = text.lower()
    detected_disasters = []
    for disaster, keywords in DISASTER_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_disasters.append(disaster)
    
    # Text metrics
    words = text.split()
    exclamation_count = text.count('!')
    question_count = text.count('?')
    caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
    has_url = bool(re.search(r'http[s]?://', text_lower))
    numbers = re.findall(r'\d+', text)
    
    # Generate reasons
    reasons = []
    if predicted_class == 1:
        reasons.append("🔴 BERT model classified as DISASTER tweet")
        if confidence > 0.8:
            reasons.append("✅ High confidence prediction")
        elif confidence > 0.6:
            reasons.append("⚠️ Medium confidence prediction")
        else:
            reasons.append("❓ Low confidence prediction")
    else:
        reasons.append("✅ BERT model classified as NOT DISASTER")
        if confidence > 0.8:
            reasons.append("✅ High confidence prediction")
        elif confidence > 0.6:
            reasons.append("⚠️ Medium confidence prediction")
        else:
            reasons.append("❓ Low confidence prediction")
    
    if detected_disasters:
        reasons.append(f"🌪️ Detected disaster type(s): {', '.join(detected_disasters)}")
    
    return {
        "is_fake": predicted_class == 1,
        "fake_probability": prob_disaster,
        "real_probability": prob_not_disaster,
        "confidence": confidence,
        "reasons": reasons,
        "detected_disasters": detected_disasters,
        "model_used": "BERT (HuggingFace)",
        "response_time": time.time() - start_time,
        "word_count": len(words),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "caps_words": caps_words,
        "has_url": has_url,
        "number_count": len(numbers)
    }

# ================================================================
# LOGISTIC REGRESSION PREDICTION FUNCTION (backup)
# ================================================================

def predict_with_lr(text):
    """Predict using Logistic Regression (backup)"""
    if not lr_loaded:
        return None
    
    start_time = time.time()
    
    clean_text = preprocess_tweet(text)
    X = vectorizer.transform([clean_text])
    probs = lr_model.predict_proba(X)[0]
    pred = lr_model.predict(X)[0]
    
    text_lower = text.lower()
    detected_disasters = []
    for disaster, keywords in DISASTER_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_disasters.append(disaster)
    
    words = text.split()
    
    return {
        "is_fake": pred == 1,
        "fake_probability": probs[1],
        "real_probability": probs[0],
        "confidence": probs[pred],
        "reasons": ["📊 Logistic Regression model prediction", "TF-IDF + Linear Classification"],
        "detected_disasters": detected_disasters,
        "model_used": "Logistic Regression",
        "response_time": time.time() - start_time,
        "word_count": len(words),
        "exclamation_count": text.count('!'),
        "question_count": text.count('?'),
        "caps_words": sum(1 for word in words if word.isupper() and len(word) > 2),
        "has_url": bool(re.search(r'http[s]?://', text_lower)),
        "number_count": len(re.findall(r'\d+', text))
    }

# ================================================================
# DISASTER AI FUNCTION (simulated - always available)
# ================================================================

def predict_with_disaster_ai(text):
    """Simulated Disaster AI (always available as fallback)"""
    start_time = time.time()
    
    # Simple keyword-based detection
    text_lower = text.lower()
    
    fake_score = 0
    real_score = 0
    
    # Fake indicators
    fake_keywords = ['urgent', 'breaking', 'share', 'viral', '!!!', 'omg', 'shocking', 'massive']
    for kw in fake_keywords:
        if kw in text_lower:
            fake_score += 1
    
    # Real indicators
    real_keywords = ['according to', 'official', 'authorities', 'reported', 'confirmed', 'jps', 'bomba']
    for kw in real_keywords:
        if kw in text_lower:
            real_score += 1.5
    
    # Disaster detection
    detected_disasters = []
    for disaster, keywords in DISASTER_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_disasters.append(disaster)
    
    # Calculate probability
    total = fake_score + real_score
    if total > 0:
        fake_prob = fake_score / total
    else:
        fake_prob = 0.5
    
    real_prob = 1 - fake_prob
    
    words = text.split()
    
    return {
        "is_fake": fake_prob > 0.5,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "confidence": abs(fake_prob - 0.5) * 2,
        "reasons": ["🌪️ Disaster AI (simulated) - No actual AI model loaded"],
        "detected_disasters": detected_disasters,
        "model_used": "Disaster AI (Simulated)",
        "response_time": time.time() - start_time,
        "word_count": len(words),
        "exclamation_count": text.count('!'),
        "question_count": text.count('?'),
        "caps_words": sum(1 for word in words if word.isupper() and len(word) > 2),
        "has_url": bool(re.search(r'http[s]?://', text_lower)),
        "number_count": len(re.findall(r'\d+', text))
    }

# ================================================================
# MAIN ANALYSIS FUNCTION
# ================================================================

def analyze_tweet(text, model_choice):
    """Main analysis function routing"""
    
    if model_choice == "huggingface_bert" and bert_loaded:
        result = predict_with_bert(text)
        if result is not None:
            return result
    
    if model_choice == "lr" and lr_loaded:
        result = predict_with_lr(text)
        if result is not None:
            return result
    
    # Always fall back to Disaster AI
    return predict_with_disaster_ai(text)

# ================================================================
# DISPLAY FUNCTIONS
# ================================================================

def display_probability_bar(fake_prob, real_prob):
    """Display visual probability bar"""
    fake_percent = fake_prob * 100
    real_percent = real_prob * 100
    
    st.markdown(f"""
    <div class="probability-bar-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span style="color: #ff6b6b; font-weight: 600;">DISASTER: {fake_percent:.1f}%</span>
            <span style="color: #10ac84; font-weight: 600;">NOT DISASTER: {real_percent:.1f}%</span>
        </div>
        <div class="probability-bar">
            <div class="fake-bar" style="width: {fake_percent}%;">
                {fake_percent:.1f}%
            </div>
            <div class="real-bar" style="width: {real_percent}%;">
                {real_percent:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_comprehensive_metrics(analysis):
    """Display all analysis metrics"""
    
    st.markdown('<h3 class="section-header">📊 Disaster Tweet Analysis</h3>', unsafe_allow_html=True)
    
    # Probability Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Disaster Probability</h4>
            <h2 style="color: {'#ff6b6b' if analysis['fake_probability'] > 0.5 else '#666'}">
                {analysis['fake_probability']*100:.1f}%
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Non-Disaster Probability</h4>
            <h2 style="color: {'#10ac84' if analysis['real_probability'] > 0.5 else '#666'}">
                {analysis['real_probability']*100:.1f}%
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Decision</h4>
            <h2>{'🔴 DISASTER' if analysis['is_fake'] else '✅ NOT DISASTER'}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Confidence Analysis
    st.markdown('<h4 style="margin-top: 20px;">🎯 Confidence Analysis</h4>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Confidence", f"{analysis.get('confidence', 0.5)*100:.1f}%")
    
    with col2:
        st.metric("Words", analysis.get('word_count', 0))
    
    with col3:
        st.metric("Exclamation", analysis.get('exclamation_count', 0))
    
    with col4:
        st.metric("Caps Words", analysis.get('caps_words', 0))
    
    # URL and Numbers
    if analysis.get('has_url'):
        st.info("🔗 Contains reference link")
    
    if analysis.get('number_count', 0) > 0:
        st.info(f"🔢 Contains {analysis['number_count']} numbers")
    
    # Disaster Type
    if analysis.get('detected_disasters'):
        st.markdown("#### 🌪️ Detected Disaster Types")
        cols = st.columns(len(analysis['detected_disasters']))
        for i, disaster in enumerate(analysis['detected_disasters']):
            with cols[i]:
                st.info(disaster.upper())
    
    # Reasons
    if analysis.get('reasons'):
        st.markdown("#### 🔍 Detection Reasons")
        for reason in analysis['reasons']:
            if "DISASTER" in reason or "disaster" in reason:
                st.error(f"• {reason}")
            elif "NOT" in reason:
                st.success(f"• {reason}")
            else:
                st.warning(f"• {reason}")

def create_location_map(location, lat, lon, is_fake):
    """Create a location map"""
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode='markers+text',
        marker=dict(
            size=20,
            color='red' if is_fake else 'green',
            symbol='marker'
        ),
        text=[location],
        textposition="top center"
    ))
    
    radius_km = 5
    radius_deg = radius_km / 111.0
    circle_points = 50
    circle_lats = [lat + radius_deg * math.cos(2 * math.pi * i / circle_points) 
                  for i in range(circle_points + 1)]
    circle_lons = [lon + radius_deg * math.sin(2 * math.pi * i / circle_points) 
                  for i in range(circle_points + 1)]
    
    fig.add_trace(go.Scattermapbox(
        lat=circle_lats,
        lon=circle_lons,
        mode='lines',
        line=dict(width=2, color='red' if is_fake else 'green'),
        name='5km Radius'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat, lon=lon),
            zoom=10
        ),
        height=400,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    return fig

# ================================================================
# REAL-TIME DATA MANAGER (simplified)
# ================================================================

class RealtimeDataManager:
    """Simplified data manager"""
    
    def __init__(self, firebase_active):
        self.firebase_active = firebase_active
        
    def save_analysis(self, analysis_data):
        st.session_state["local_analyses"].append(analysis_data)
        stats = st.session_state["local_stats"]
        stats["total_analyses"] += 1
        if analysis_data.get("is_fake"):
            stats["total_fake"] += 1
        else:
            stats["total_real"] += 1
        
        if analysis_data.get("location"):
            loc = analysis_data["location"]
            stats["locations"][loc] = stats["locations"].get(loc, 0) + 1
        
        for disaster in analysis_data.get("detected_disasters", []):
            stats["disaster_types"][disaster] = stats["disaster_types"].get(disaster, 0) + 1
        
        return "local"
    
    def get_live_stats(self):
        return st.session_state["local_stats"]
    
    def get_live_analyses(self, limit=50):
        return st.session_state["local_analyses"][-limit:]

# Initialize real-time manager
rt_manager = RealtimeDataManager(FIREBASE_ACTIVE)

# ================================================================
# MAIN UI STARTS HERE
# ================================================================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================
badge_class = "live-badge" if FIREBASE_ACTIVE else "local-badge"
badge_text = "🔴 LIVE" if FIREBASE_ACTIVE else "⚫ LOCAL"
connection_status = "Connected" if FIREBASE_ACTIVE else "Offline Mode"

# Determine which model is active
if bert_loaded:
    model_display = "🧠 HuggingFace BERT"
    model_badge_class = "bert-badge"
else:
    model_display = "🌪️ Disaster AI"
    model_badge_class = "disaster-badge"

st.markdown(
    f'''
    <div class="main-header">
        <h1 style="font-size: 3em; margin-bottom: 10px;">🌪️ Disaster Tweet AI Detector</h1>
        <p style="font-size: 1.2em; opacity: 0.9;">Powered by HuggingFace 🤗 BERT Model</p>
        <div style="margin-top: 20px;">
            <span class="{model_badge_class}">{model_display}</span>
            <span class="status-badge {badge_class}">{badge_text}</span>
            <span class="status-badge huggingface-model-badge">🤗 HuggingFace Hub</span>
        </div>
        <p style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">Session: {st.session_state["session_id"]}</p>
    </div>
    ''',
    unsafe_allow_html=True
)

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    # HuggingFace Info
    st.markdown("### 🤗 HuggingFace Integration")
    st.markdown("""
    <div style="background: #6e40ff20; padding: 15px; border-radius: 12px; border-left: 4px solid #6e40ff;">
        <strong style="color: #6e40ff;">MODEL FROM HUGGINGFACE HUB</strong><br>
        <small>Model downloads automatically during deployment</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Status
    st.markdown("### 🧠 Model Status")
    
    if bert_loaded:
        st.success("✅ HuggingFace BERT: LOADED")
        st.info("Model downloaded from HuggingFace Hub")
    else:
        st.error("❌ HuggingFace BERT: NOT LOADED")
        st.warning("Using Disaster AI fallback")
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    
    # Model Selection
    model_options = []
    if bert_loaded:
        model_options.append("huggingface_bert")
    if lr_loaded:
        model_options.append("lr")
    model_options.append("disaster_ai")
    
    model_labels = {
        "huggingface_bert": "🧠 HuggingFace BERT (Recommended)",
        "lr": "📊 Logistic Regression",
        "disaster_ai": "🌪️ Disaster AI (Fallback)"
    }
    
    model_choice = st.radio(
        "Select Model",
        model_options,
        format_func=lambda x: model_labels.get(x, x),
        index=0
    )
    st.session_state["model_choice"] = model_choice
    
    # Auto-refresh toggle
    st.session_state["auto_refresh"] = st.toggle(
        "🔄 Auto-refresh",
        value=True
    )
    
    st.markdown("---")
    
    # Live Statistics
    st.markdown("### 📊 Statistics")
    stats = rt_manager.get_live_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", stats.get('total_analyses', 0))
    with col2:
        st.metric("Disasters", stats.get('total_fake', 0))
    
    # Clear data button
    if st.button("🗑️ Clear Data", use_container_width=True):
        st.session_state["local_analyses"] = []
        st.session_state["local_stats"] = {
            "total_analyses": 0,
            "total_fake": 0,
            "total_real": 0,
            "locations": {},
            "disaster_types": {},
            "models_used": {}
        }
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
# AUTO-REFRESH LOGIC
# ================================================================
if st.session_state["auto_refresh"]:
    if time.time() - st.session_state["last_refresh"] > 5:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

# ================================================================
# INPUT SECTION
# ================================================================
st.markdown("### 📝 Enter Tweet")

input_col1, input_col2 = st.columns([6, 1])

with input_col1:
    input_key = f"tweet_input_{st.session_state['input_key_counter']}"
    tweet = st.text_area(
        "Enter tweet:",
        height=100,
        placeholder="Example: Heavy rain in Kampar causing flash floods...",
        key=input_key,
        label_visibility="collapsed"
    )

with input_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["input_key_counter"] += 1
        st.rerun()

# Quick Examples
st.markdown("#### 📋 Examples")
ex_col1, ex_col2, ex_col3 = st.columns(3)

with ex_col1:
    if st.button("📰 Real Disaster", use_container_width=True):
        st.session_state["tweet_input"] = "Heavy rain in Kampar causing flash floods. According to JPS, water levels rising. 150 evacuated."
        st.session_state["input_key_counter"] += 1
        st.rerun()

with ex_col2:
    if st.button("🚨 Fake Disaster", use_container_width=True):
        st.session_state["tweet_input"] = "URGENT! BREAKING: MASSIVE earthquake in KL! Thousands DEAD! SHARE NOW! 😱"
        st.session_state["input_key_counter"] += 1
        st.rerun()

with ex_col3:
    if st.button("📍 Location Test", use_container_width=True):
        st.session_state["tweet_input"] = "Landslide in Cameron Highlands - rescue ongoing"
        st.session_state["input_key_counter"] += 1
        st.rerun()

# Analyze Button
col1, col2 = st.columns([1, 5])
with col1:
    analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ================================================================
# ANALYSIS EXECUTION
# ================================================================
if analyze_clicked and tweet:
    model_name = {
        "huggingface_bert": "HuggingFace BERT",
        "lr": "Logistic Regression",
        "disaster_ai": "Disaster AI"
    }.get(st.session_state["model_choice"], "Model")
    
    with st.spinner(f"🤖 {model_name} analyzing..."):
        
        result = analyze_tweet(tweet, st.session_state["model_choice"])
        
        if result:
            # Extract location
            location = None
            for loc in MALAYSIA_LOCATIONS:
                if loc.lower() in tweet.lower():
                    location = loc
                    break
            
            # Prepare data for storage
            analysis_data = {
                "tweet": tweet[:100] + "...",
                "location": location,
                "is_fake": result.get("is_fake"),
                "fake_probability": result.get("fake_probability"),
                "real_probability": result.get("real_probability"),
                "detected_disasters": result.get("detected_disasters", []),
                "model_used": result.get("model_used", model_name)
            }
            
            rt_manager.save_analysis(analysis_data)
            
            # Display results
            st.markdown("---")
            
            # Alert
            alert_class = "bert-alert" if "BERT" in result.get("model_used", "") else "disaster-alert"
            
            if result["is_fake"]:
                st.markdown(
                    f'<div class="{alert_class}">🔴 DISASTER TWEET DETECTED<br>'
                    f'Confidence: {result.get("confidence", 0.5)*100:.1f}%<br>'
                    f'<small>Model: {result.get("model_used", "Unknown")}</small></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="{alert_class}">✅ NOT A DISASTER TWEET<br>'
                    f'Confidence: {result.get("confidence", 0.5)*100:.1f}%<br>'
                    f'<small>Model: {result.get("model_used", "Unknown")}</small></div>',
                    unsafe_allow_html=True
                )
            
            # Display probability bar
            display_probability_bar(result["fake_probability"], result["real_probability"])
            
            # Display metrics
            display_comprehensive_metrics(result)
            
            # Location map
            if location:
                st.info(f"📍 Location: {location}")
                try:
                    url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
                    response = requests.get(url, headers={'User-Agent': 'Disaster-Detector'}, timeout=5)
                    data = response.json()
                    if data:
                        lat, lon = float(data[0]['lat']), float(data[0]['lon'])
                        st.markdown("### 🗺️ Map")
                        fig = create_location_map(location, lat, lon, result["is_fake"])
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    pass

elif analyze_clicked and not tweet:
    st.warning("⚠️ Please enter a tweet")

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown(
    f'''
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #ff6b6b10, #feca5710); border-radius: 15px;">
        <p style="color: #666;">
            🌪️ Disaster Tweet AI Detector | Powered by 🤗 HuggingFace BERT<br>
            Session: {st.session_state["session_id"]} | Model auto-downloads from HuggingFace Hub
        </p>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
