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
    page_title="Disaster Tweet AI Detector - Fine-tuned BERT",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌪️"
)

# ================================================================
# ENHANCED CSS WITH MODERN DESIGN (KEEP YOUR EXISTING CSS)
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
    
    /* Disaster AI badge */
    .disaster-badge {
        background: linear-gradient(135deg, #ff6b6b, #feca57);
        color: white;
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 1em;
        font-weight: 600;
        display: inline-block;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
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
    
    .disaster-model-badge {
        background: linear-gradient(135deg, #ff6b6b, #feca57);
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
# LOAD FINE-TUNED BERT MODEL
# ================================================================

@st.cache_resource(show_spinner="Loading Fine-tuned BERT Model...")
def load_finetuned_bert():
    """Load your fine-tuned BERT model"""
    model_path = "bert_disaster_model_fine_tuned"
    
    # Try multiple possible paths
    possible_paths = [
        model_path,
        os.path.join(os.getcwd(), model_path),
        os.path.join(os.path.dirname(os.getcwd()), model_path)
    ]
    
    loaded_path = None
    for path in possible_paths:
        if os.path.exists(path):
            loaded_path = path
            break
    
    if loaded_path is None:
        st.sidebar.warning("⚠️ Fine-tuned BERT model not found. Please train the model first.")
        return None, None, False
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(loaded_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(loaded_path, local_files_only=True)
        model.eval()
        return model, tokenizer, True
    except Exception as e:
        st.sidebar.error(f"❌ Error loading BERT model: {e}")
        return None, None, False

# Load your fine-tuned BERT model
bert_model, bert_tokenizer, bert_loaded = load_finetuned_bert()

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
# SESSION STATE INITIALIZATION
# ================================================================
if "session_id" not in st.session_state:
    st.session_state["session_id"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "finetuned_bert"  # Default to fine-tuned BERT
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
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if "animations_enabled" not in st.session_state:
    st.session_state["animations_enabled"] = True
if "deterministic_mode" not in st.session_state:
    st.session_state["deterministic_mode"] = True

# ================================================================
# CONSTANTS (Keep your existing constants)
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
# BERT PREDICTION FUNCTION
# ================================================================

def predict_with_bert(text):
    """Predict using fine-tuned BERT model"""
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
            "model_used": "BERT (Fine-tuned)",
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
    
    # Detect disaster types (simple keyword check for display)
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
        "is_fake": predicted_class == 1,  # 1 = disaster
        "fake_probability": prob_disaster,
        "real_probability": prob_not_disaster,
        "confidence": confidence,
        "reasons": reasons,
        "detected_disasters": detected_disasters,
        "model_used": "BERT (Fine-tuned)",
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
    
    # Detect disaster types
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
        "has_url": bool(re.search(r'http[s]?://', text.lower())),
        "number_count": len(re.findall(r'\d+', text))
    }

# ================================================================
# MAIN ANALYSIS FUNCTION
# ================================================================

def analyze_tweet(text, model_choice):
    """Main analysis function routing"""
    
    if model_choice == "finetuned_bert" and bert_loaded:
        result = predict_with_bert(text)
        if result is None and lr_loaded:
            result = predict_with_lr(text)
    elif model_choice == "lr" and lr_loaded:
        result = predict_with_lr(text)
    elif model_choice == "disaster_ai":
        # Use your existing Disaster AI function
        result = analyze_with_disaster_ai(text, st.session_state["deterministic_mode"])
    else:
        # Fallback to Disaster AI
        result = analyze_with_disaster_ai(text, st.session_state["deterministic_mode"])
    
    return result

# ================================================================
# KEEP YOUR EXISTING DISASTER AI FUNCTION
# ================================================================
def analyze_with_disaster_ai(text, deterministic=True):
    """Your existing Disaster AI function (keep as is)"""
    # [Copy your existing analyze_with_disaster_ai function here]
    # For brevity, I'm not copying the entire function, but you should keep it
    pass

# ================================================================
# KEEP YOUR EXISTING DISPLAY FUNCTIONS
# ================================================================
# [Keep all your existing display functions: display_probability_bar, 
#  display_comprehensive_metrics, create_location_map, display_live_stats,
#  display_live_alerts, display_live_feed]

# ================================================================
# KEEP YOUR EXISTING REALTIME DATA MANAGER
# ================================================================
# [Keep your existing RealtimeDataManager class]

# Initialize real-time manager
rt_manager = RealtimeDataManager(db, FIREBASE_ACTIVE)

# ================================================================
# MAIN UI STARTS HERE
# ================================================================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================
badge_class = "live-badge" if FIREBASE_ACTIVE else "local-badge"
badge_text = "🔴 LIVE" if FIREBASE_ACTIVE else "⚫ LOCAL"
connection_status = "Connected to Global Disaster Network" if FIREBASE_ACTIVE else "Offline Mode - Data stored locally"

# Determine which model is active for display
if bert_loaded:
    model_display = "🧠 Fine-tuned BERT"
    model_badge_class = "bert-badge"
else:
    model_display = "🌪️ Disaster AI"
    model_badge_class = "disaster-badge"

st.markdown(
    f'''
    <div class="main-header">
        <h1 style="font-size: 3em; margin-bottom: 10px;">🌪️ Disaster Tweet AI Detector</h1>
        <p style="font-size: 1.2em; opacity: 0.9;">Fine-tuned BERT model for disaster tweet classification</p>
        <div style="margin-top: 20px;">
            <span class="{model_badge_class}">{model_display}</span>
            <span class="status-badge {badge_class}">{badge_text}</span>
        </div>
        <p style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">{connection_status} | Session: {st.session_state["session_id"]}</p>
    </div>
    ''',
    unsafe_allow_html=True
)

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    # Connection Status Card
    st.markdown("### 🌐 Disaster Network")
    if FIREBASE_ACTIVE:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10ac8420, #1dd1a120); padding: 15px; border-radius: 12px; border-left: 4px solid #10ac84;">
            <div style="display: flex; align-items: center;">
                <span class="live-dot"></span>
                <strong style="color: #10ac84;">LIVE DISASTER NETWORK</strong>
            </div>
            <p style="margin-top: 10px; font-size: 0.9em;">Data syncing globally in real-time</p>
            <div class="progress-container">
                <div class="progress-bar" style="width: 100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #95a5a620, #7f8c8d20); padding: 15px; border-radius: 12px; border-left: 4px solid #95a5a6;">
            <div style="display: flex; align-items: center;">
                <span style="width: 12px; height: 12px; background: #95a5a6; border-radius: 50%; margin-right: 8px;"></span>
                <strong style="color: #7f8c8d;">LOCAL MODE</strong>
            </div>
            <p style="margin-top: 10px; font-size: 0.9em;">Data stored locally - lost on refresh</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Status Cards
    st.markdown("### 🤖 AI Models")
    col1, col2 = st.columns(2)
    
    with col1:
        if bert_loaded:
            st.markdown("""
            <div style="background: #667eea20; padding: 10px; border-radius: 10px; text-align: center;">
                <span style="font-size: 2em;">🧠</span><br>
                <strong style="color: #667eea;">Fine-tuned BERT</strong><br>
                <span style="color: #667eea;">✅ Ready</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #ff6b6b20; padding: 10px; border-radius: 10px; text-align: center;">
                <span style="font-size: 2em;">🧠</span><br>
                <strong style="color: #ff6b6b;">Fine-tuned BERT</strong><br>
                <span style="color: #ff6b6b;">❌ Not Found</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if lr_loaded:
            st.markdown("""
            <div style="background: #10ac8420; padding: 10px; border-radius: 10px; text-align: center;">
                <span style="font-size: 2em;">📊</span><br>
                <strong style="color: #10ac84;">Logistic Regression</strong><br>
                <span style="color: #10ac84;">✅ Ready</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #ff6b6b20; padding: 10px; border-radius: 10px; text-align: center;">
                <span style="font-size: 2em;">📊</span><br>
                <strong style="color: #ff6b6b;">Logistic Regression</strong><br>
                <span style="color: #ff6b6b;">❌ Not Found</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Performance Metrics
    st.markdown("### 📈 Model Performance")
    
    if bert_loaded:
        st.markdown("""
        <div style="background: #667eea10; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
            <strong>🧠 Fine-tuned BERT</strong><br>
            • Accuracy: ~85-90%<br>
            • Precision: ~86%<br>
            • Recall: ~84%<br>
            • F1-Score: ~85%<br>
            <small>Trained on 11,370 disaster tweets</small>
        </div>
        """, unsafe_allow_html=True)
    
    if lr_loaded:
        st.markdown("""
        <div style="background: #10ac8410; padding: 15px; border-radius: 10px;">
            <strong>📊 Logistic Regression</strong><br>
            • Accuracy: ~88%<br>
            • Fast & lightweight<br>
            • Good baseline model
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    
    # Model Selection
    model_options = []
    model_labels = {}
    
    if bert_loaded:
        model_options.append("finetuned_bert")
        model_labels["finetuned_bert"] = "🧠 Fine-tuned BERT (Most Accurate)"
    
    if lr_loaded:
        model_options.append("lr")
        model_labels["lr"] = "📊 Logistic Regression (Fast)"
    
    # Always include Disaster AI as fallback
    model_options.append("disaster_ai")
    model_labels["disaster_ai"] = "🌪️ Disaster AI (Simulated)"
    
    model_choice = st.radio(
        "Select Detection Model",
        model_options,
        format_func=lambda x: model_labels.get(x, x),
        index=0,
        help="Choose your preferred disaster detection model"
    )
    st.session_state["model_choice"] = model_choice
    
    # Deterministic mode toggle (for Disaster AI)
    if model_choice == "disaster_ai":
        st.session_state["deterministic_mode"] = st.toggle(
            "🎯 Deterministic Mode",
            value=True,
            help="Same disaster tweet always gives same results"
        )
    
    # Auto-refresh toggle
    st.session_state["auto_refresh"] = st.toggle(
        "🔄 Auto-refresh Feed",
        value=True,
        help="Automatically refresh live disaster feed every 5 seconds"
    )
    
    st.markdown("---")
    
    # Live Statistics
    st.markdown("### 📊 Disaster Statistics")
    stats = rt_manager.get_live_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Analyses</div>
            <div class="stat-value">{stats.get('total_analyses', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Fake Disasters</div>
            <div class="stat-value" style="color: #ff6b6b;">{stats.get('total_fake', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Real Disasters</div>
            <div class="stat-value" style="color: #10ac84;">{stats.get('total_real', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if stats.get('locations'):
            top_location = max(stats['locations'].items(), key=lambda x: x[1])[0]
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Hotspot</div>
                <div class="stat-value" style="font-size: 1.2em;">📍 {top_location}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Clear data button
    if st.button("🗑️ Clear All Disaster Data", use_container_width=True):
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
# LIVE ALERTS SECTION
# ================================================================
display_live_alerts()

# ================================================================
# INPUT SECTION
# ================================================================
st.markdown("### 📝 Enter Disaster Tweet")

input_col1, input_col2 = st.columns([6, 1])

with input_col1:
    input_key = f"tweet_input_{st.session_state['input_key_counter']}"
    
    tweet = st.text_area(
        "Enter disaster tweet to analyze:",
        height=120,
        placeholder="Example: Heavy rain in Kampar causing flash floods - reported by local authorities JPS monitoring water levels",
        key=input_key,
        label_visibility="collapsed"
    )

with input_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear", use_container_width=True, help="Clear input field"):
        st.session_state["input_key_counter"] += 1
        st.rerun()

# Quick Examples
st.markdown("#### 🎯 Disaster Tweet Examples")
example_col1, example_col2, example_col3, example_col4 = st.columns(4)

with example_col1:
    if st.button("📰 Real Disaster", use_container_width=True, key="real_example"):
        st.session_state["tweet_input"] = "Heavy rain in Kampar causing flash floods. According to local authorities, JPS monitoring water levels. 150 residents evacuated to relief centers."
        st.session_state["input_key_counter"] += 1
        st.rerun()

with example_col2:
    if st.button("🚨 Fake Disaster", use_container_width=True, key="fake_example"):
        st.session_state["tweet_input"] = "URGENT! BREAKING: MASSIVE 8.0 earthquake just hit Kuala Lumpur! Thousands DEAD! Government hiding truth! SHARE NOW before they delete! 😱😱😱"
        st.session_state["input_key_counter"] += 1
        st.rerun()

with example_col3:
    if st.button("🔄 Mixed Signals", use_container_width=True, key="mixed_example"):
        st.session_state["tweet_input"] = "URGENT! Flood in Johor! Water level 2 meters! SHARE NOW! Official source says evacuating. JPS confirms."
        st.session_state["input_key_counter"] += 1
        st.rerun()

with example_col4:
    if st.button("📍 Location Test", use_container_width=True, key="location_example"):
        st.session_state["tweet_input"] = "Landslide reported in Cameron Highlands - authorities responding, 3 people rescued"
        st.session_state["input_key_counter"] += 1
        st.rerun()

# Action Buttons
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_clicked = st.button("🔍 Analyze Disaster Tweet", type="primary", use_container_width=True)
with col2:
    if st.button("🔄 New Tweet", use_container_width=True):
        st.session_state["input_key_counter"] += 1
        st.rerun()

# ================================================================
# ANALYSIS EXECUTION
# ================================================================
if analyze_clicked and tweet:
    model_name = {
        "finetuned_bert": "Fine-tuned BERT",
        "lr": "Logistic Regression",
        "disaster_ai": "Disaster AI"
    }.get(st.session_state["model_choice"], "Unknown")
    
    with st.spinner(f"🧠 {model_name} analyzing..."):
        
        result = analyze_tweet(
            tweet, 
            st.session_state["model_choice"]
        )
        
        if result:
            # Extract location
            location = None
            for loc in MALAYSIA_LOCATIONS:
                if loc.lower() in tweet.lower():
                    location = loc
                    break
            
            # Prepare data for storage
            analysis_data = {
                "tweet": tweet,
                "tweet_preview": tweet[:100] + "..." if len(tweet) > 100 else tweet,
                "location": location,
                "is_fake": result.get("is_fake"),
                "fake_probability": result.get("fake_probability"),
                "real_probability": result.get("real_probability"),
                "confidence": result.get("confidence", 0.5),
                "detected_disasters": result.get("detected_disasters", []),
                "word_count": result.get("word_count", 0),
                "model_used": result.get("model_used", model_name),
                "session_id": st.session_state["session_id"]
            }
            
            # Save to Firebase or local storage
            doc_id = rt_manager.save_analysis(analysis_data)
            
            if doc_id and FIREBASE_ACTIVE:
                st.success(f"✅ Analysis saved to global disaster feed!")
            else:
                st.success("✅ Analysis saved locally")
            
            # Display results
            st.markdown("---")
            
            # Alert based on result and model
            if result.get("model_used", "").startswith("BERT"):
                alert_class = "bert-alert"
                icon = "🧠"
            elif result.get("model_used", "").startswith("Logistic"):
                alert_class = "real-alert"
                icon = "📊"
            else:
                alert_class = "disaster-alert"
                icon = "🌪️"
            
            if result["is_fake"]:
                st.markdown(
                    f'<div class="{alert_class}">{icon} {result.get("model_used", "Model")} DETECTED: FAKE DISASTER NEWS<br>'
                    f'Confidence: {result.get("confidence", 0.5)*100:.1f}% | '
                    f'Disaster Probability: {result["fake_probability"]*100:.1f}%</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="{alert_class}">{icon} {result.get("model_used", "Model")} DETECTED: REAL DISASTER NEWS<br>'
                    f'Confidence: {result.get("confidence", 0.5)*100:.1f}% | '
                    f'Real Probability: {result["real_probability"]*100:.1f}%</div>',
                    unsafe_allow_html=True
                )
            
            # Display probability bar
            display_probability_bar(result["fake_probability"], result["real_probability"])
            
            # Display comprehensive metrics
            display_comprehensive_metrics(result)
            
            # Display location map if location found
            if location:
                st.info(f"📍 Disaster location detected: {location}")
                
                try:
                    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(location)}&format=json&limit=1"
                    headers = {'User-Agent': 'Disaster-Detector/1.0'}
                    response = requests.get(url, headers=headers, timeout=5)
                    data = response.json()
                    
                    if data:
                        lat, lon = float(data[0]['lat']), float(data[0]['lon'])
                        st.markdown("### 🗺️ Disaster Location Map")
                        fig = create_location_map(location, lat, lon, result["is_fake"])
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load map for {location}")
            
            # Model info
            st.info(f"🤖 Model: {result.get('model_used', 'Unknown')} | Response time: {result.get('response_time', 0)*1000:.0f}ms")

elif analyze_clicked and not tweet:
    st.warning("⚠️ Please enter a disaster tweet to analyze.")

# ================================================================
# LIVE FEED
# ================================================================
st.markdown("---")
display_live_feed()

if st.button("🔄 Refresh Disaster Feed", use_container_width=True):
    st.rerun()

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown(
    f'''
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #ff6b6b10, #feca5710); border-radius: 15px; margin-top: 30px;">
        <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 20px;">
            <span class="status-badge" style="background: #667eea; color: white;">🧠 Fine-tuned BERT</span>
            <span class="status-badge" style="background: #10ac84; color: white;">⚡ Real-time</span>
            <span class="status-badge" style="background: #f39c12; color: white;">🔬 85-90% Accuracy</span>
        </div>
        <p style="color: #666; font-size: 0.9em;">
            Disaster Tweet AI Detector | Fine-tuned BERT Model for Disaster Classification<br>
            Trained on 11,370 disaster tweets from Kaggle | Data {'syncing globally' if FIREBASE_ACTIVE else 'stored locally'}<br>
            Session: {st.session_state["session_id"]} | Last sync: {datetime.now().strftime('%H:%M:%S')}
        </p>
        <div style="margin-top: 20px;">
            <span class="live-dot"></span> Live system - Updates every 5 seconds
        </div>
        <p style="color: #888; font-size: 0.8em; margin-top: 15px;">
            ✅ Fine-tuned BERT • Logistic Regression • Disaster AI • Real-time Analytics
        </p>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
