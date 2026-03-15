# ================================================================
# DISASTER TWEET AI DETECTOR - BERT MODEL + MOCK API ONLY
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
import zipfile
import io

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Disaster Tweet AI Detector - BERT + Mock",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    
    /* Header with gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #48bb78 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        animation: slideDown 0.5s ease-out;
    }
    
    @keyframes slideDown {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Model badges */
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
        animation: pulse 2s infinite;
    }
    
    .mock-badge {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 1em;
        font-weight: 600;
        display: inline-block;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(72, 187, 120, 0.4);
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.9; transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Alert animations */
    .bert-alert, .mock-alert {
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
    
    .bert-alert {
        background: linear-gradient(135deg, #667eea, #5a67d8);
    }
    
    .mock-alert {
        background: linear-gradient(135deg, #48bb78, #38a169);
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
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
    
    .disaster-bar {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b, #ee5253);
        color: white;
        text-align: center;
        line-height: 40px;
        font-weight: bold;
        transition: width 0.5s ease-in-out;
    }
    
    .non-disaster-bar {
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
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 10px 0;
        animation: fadeIn 0.5s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
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
    
    .local-badge {
        background: linear-gradient(135deg, #95a5a6, #7f8c8d);
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
        border-color: #667eea !important;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3) !important;
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
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Primary button */
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #667eea, #5a67d8) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-163ttbj {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
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
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.3);
    }
    
    .stat-value {
        font-size: 2.5em;
        font-weight: 700;
        color: #667eea;
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
        background: linear-gradient(90deg, #667eea, #5a67d8);
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
        border-bottom: 3px solid #667eea;
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
# SESSION STATE INITIALIZATION
# ================================================================
if "session_id" not in st.session_state:
    st.session_state["session_id"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "bert"  # Default to BERT
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
        "total_disaster": 0,
        "total_non_disaster": 0,
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
# DOWNLOAD MODEL FUNCTION
# ================================================================

@st.cache_resource(show_spinner="🔄 Loading BERT model...")
def download_and_load_model():
    """
    Download BERT model from cloud storage if not present locally
    """
    model_path = "bert_disaster_model_fine_tuned"
    
    # First, check if model already exists locally
    if os.path.exists(model_path):
        try:
            with st.spinner("Loading existing BERT model..."):
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
                model.eval()
            st.sidebar.success("✅ BERT model loaded from local storage!")
            return model, tokenizer, True
        except Exception as e:
            st.warning(f"Existing model corrupted: {e}. Will re-download.")
    
    # If not found or corrupted, download from cloud
    st.info("📥 BERT model not found locally. Downloading from cloud storage...")
    
    # Your Google Drive file ID
    file_id = "1iUBsn-eNIBftXxzzW65Z8wYXg1IgYhRt"
    download_url = f"https://drive.google.com/file/d/1iUBsn-eNIBftXxzzW65Z8wYXg1IgYhRt/view?usp=sharing"
    
    try:
        # Create a session to handle cookies
        session = requests.Session()
        
        # First request might get a warning page
        response = session.get(download_url, stream=True, allow_redirects=True)
        
        # Check if we need to confirm the download (Google Drive's virus check)
        if "confirm" in response.url:
            # Extract confirm code
            import re
            confirm_match = re.search(r'confirm=([0-9A-Za-z]+)', response.url)
            if confirm_match:
                confirm_code = confirm_match.group(1)
                download_url = f"https://drive.google.com/file/d/1iUBsn-eNIBftXxzzW65Z8wYXg1IgYhRt/view?usp=sharing"
                response = session.get(download_url, stream=True)
        
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Check if it's actually a zip file (not an HTML page)
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            st.error("❌ Google Drive is returning an HTML page instead of the file.")
            st.info("This often happens with large files. Trying alternative download method...")
            
            # Alternative method: use the download URL with confirm=1
            alt_url = f"https://drive.google.com/uc?export=download&confirm=1&id={file_id}"
            response = session.get(alt_url, stream=True)
            
            if 'text/html' in response.headers.get('content-type', ''):
                st.error("❌ Still getting HTML. The file might be too large or not publicly shared.")
                return None, None, False
        
        # Download with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        zip_path = "bert_model.zip"
        with open(zip_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded: {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB")
        
        status_text.text("Download complete! Verifying zip file...")
        
        # Verify it's a valid zip file
        if not zipfile.is_zipfile(zip_path):
            st.error("❌ Downloaded file is not a valid zip file.")
            st.info("The file may be corrupted or incomplete.")
            os.remove(zip_path)
            return None, None, False
        
        # Extract
        status_text.text("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Clean up
        os.remove(zip_path)
        progress_bar.empty()
        status_text.empty()
        
        # Verify the extracted folder exists
        if not os.path.exists(model_path):
            st.error(f"❌ Extracted folder '{model_path}' not found.")
            return None, None, False
        
        # Load the model
        with st.spinner("Loading BERT model..."):
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            model.eval()
        
        st.success("✅ Model downloaded and loaded successfully!")
        st.sidebar.success("✅ BERT model ready from cloud download!")
        return model, tokenizer, True
        
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.info("Falling back to Mock API...")
        return None, None, False

# ================================================================
# LOAD THE MODEL - THIS MUST BE BEFORE ANY CODE THAT USES bert_loaded
# ================================================================
bert_model, bert_tokenizer, bert_loaded = download_and_load_model()

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
    """Predict using BERT model"""
    if not bert_loaded or bert_model is None:
        return None
    
    start_time = time.time()
    
    # Preprocess
    clean_text = preprocess_tweet(text)
    
    if not clean_text:
        return {
            "is_disaster": False,
            "disaster_probability": 0.5,
            "non_disaster_probability": 0.5,
            "confidence": 0.5,
            "reasons": ["Empty tweet after preprocessing"],
            "detected_disasters": [],
            "model_used": "BERT",
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
        
        prob_non_disaster = probabilities[0][0].item()
        prob_disaster = probabilities[0][1].item()
    
    # Detect disaster types (for display)
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
        reasons.append("✅ BERT model classified as NON-DISASTER tweet")
        if confidence > 0.8:
            reasons.append("✅ High confidence prediction")
        elif confidence > 0.6:
            reasons.append("⚠️ Medium confidence prediction")
        else:
            reasons.append("❓ Low confidence prediction")
    
    if detected_disasters:
        reasons.append(f"🌪️ Detected disaster type(s): {', '.join(detected_disasters)}")
    
    if has_url:
        reasons.append("🔗 Contains URL - may provide evidence")
    
    if numbers:
        reasons.append(f"🔢 Contains {len(numbers)} numbers/data points")
    
    return {
        "is_disaster": predicted_class == 1,
        "disaster_probability": prob_disaster,
        "non_disaster_probability": prob_non_disaster,
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
# MOCK API PREDICTION FUNCTION (ALWAYS AVAILABLE)
# ================================================================

def predict_with_mock(text):
    """Mock API prediction - always available, no dependencies"""
    start_time = time.time()
    
    # Simple keyword-based detection
    text_lower = text.lower()
    
    disaster_score = 0
    non_disaster_score = 0
    
    # Disaster indicators (keywords that suggest disaster)
    disaster_keywords = [
        'flood', 'banjir', 'earthquake', 'gempa', 'fire', 'kebakaran', 
        'storm', 'ribut', 'tsunami', 'landslide', 'runtuh', 'urgent', 
        'breaking', 'emergency', 'warning', 'alert', 'disaster', 'bencana',
        'evacuate', 'pindah', 'rescue', 'saving', 'casualty', 'victim',
        'damage', 'destroyed', 'collapsed', 'runtuh', 'injured', 'terjejas'
    ]
    
    # Non-disaster indicators (keywords that suggest normal conversation)
    non_disaster_keywords = [
        'good morning', 'good night', 'hello', 'hi', 'thanks', 'thank you',
        'welcome', 'please', 'happy', 'excited', 'love', 'like', 'great',
        'awesome', 'amazing', 'wonderful', 'beautiful', 'nice', 'cool',
        'fun', 'enjoy', 'party', 'celebration', 'birthday', 'weekend',
        'holiday', 'vacation', 'food', 'restaurant', 'movie', 'music'
    ]
    
    # Count disaster keywords
    for kw in disaster_keywords:
        if kw in text_lower:
            disaster_score += 1
    
    # Count non-disaster keywords
    for kw in non_disaster_keywords:
        if kw in text_lower:
            non_disaster_score += 1.5
    
    # Detect specific disaster types
    detected_disasters = []
    for disaster, keywords in DISASTER_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_disasters.append(disaster)
    
    # Calculate probability
    total = disaster_score + non_disaster_score
    if total > 0:
        disaster_prob = disaster_score / total
    else:
        # No keywords found, default to neutral
        disaster_prob = 0.5
    
    non_disaster_prob = 1 - disaster_prob
    
    # Simple confidence calculation
    confidence = abs(disaster_prob - 0.5) * 2
    
    # Text metrics
    words = text.split()
    exclamation_count = text.count('!')
    question_count = text.count('?')
    caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
    has_url = bool(re.search(r'http[s]?://', text_lower))
    numbers = re.findall(r'\d+', text)
    
    # Generate reasons
    reasons = []
    if disaster_prob > 0.5:
        reasons.append("🔴 Mock API detected potential DISASTER content")
        if disaster_score > 0:
            reasons.append(f"📊 Found {disaster_score} disaster-related keywords")
    else:
        reasons.append("✅ Mock API detected NON-DISASTER content")
        if non_disaster_score > 0:
            reasons.append(f"📊 Found {int(non_disaster_score/1.5)} non-disaster keywords")
    
    if detected_disasters:
        reasons.append(f"🌪️ Possible disaster type(s): {', '.join(detected_disasters)}")
    
    if exclamation_count > 3:
        reasons.append(f"❗ High exclamation count ({exclamation_count})")
    
    if caps_words > 2:
        reasons.append(f"📣 Multiple ALL CAPS words ({caps_words})")
    
    return {
        "is_disaster": disaster_prob > 0.5,
        "disaster_probability": disaster_prob,
        "non_disaster_probability": non_disaster_prob,
        "confidence": confidence,
        "reasons": reasons[:5],
        "detected_disasters": detected_disasters,
        "model_used": "Mock API",
        "response_time": time.time() - start_time,
        "word_count": len(words),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "caps_words": caps_words,
        "has_url": has_url,
        "number_count": len(numbers)
    }

# ================================================================
# MAIN ANALYSIS FUNCTION
# ================================================================

def analyze_tweet(text, model_choice):
    """Main analysis function routing"""
    
    if model_choice == "bert" and bert_loaded:
        result = predict_with_bert(text)
        if result is not None:
            return result
    
    # Always fall back to Mock API
    return predict_with_mock(text)

# ================================================================
# DISPLAY FUNCTIONS
# ================================================================

def display_probability_bar(disaster_prob, non_disaster_prob):
    """Display visual probability bar"""
    disaster_percent = disaster_prob * 100
    non_disaster_percent = non_disaster_prob * 100
    
    st.markdown(f"""
    <div class="probability-bar-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span style="color: #ff6b6b; font-weight: 600;">DISASTER: {disaster_percent:.1f}%</span>
            <span style="color: #10ac84; font-weight: 600;">NON-DISASTER: {non_disaster_percent:.1f}%</span>
        </div>
        <div class="probability-bar">
            <div class="disaster-bar" style="width: {disaster_percent}%;">
                {disaster_percent:.1f}%
            </div>
            <div class="non-disaster-bar" style="width: {non_disaster_percent}%;">
                {non_disaster_percent:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_comprehensive_metrics(analysis):
    """Display all analysis metrics"""
    
    st.markdown('<h3 class="section-header">📊 Analysis Results</h3>', unsafe_allow_html=True)
    
    # Probability Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Disaster Probability</h4>
            <h2 style="color: {'#ff6b6b' if analysis['disaster_probability'] > 0.5 else '#666'}">
                {analysis['disaster_probability']*100:.1f}%
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Non-Disaster Probability</h4>
            <h2 style="color: {'#10ac84' if analysis['non_disaster_probability'] > 0.5 else '#666'}">
                {analysis['non_disaster_probability']*100:.1f}%
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Decision</h4>
            <h2>{'🔴 DISASTER' if analysis['is_disaster'] else '✅ NOT DISASTER'}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Confidence and Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Confidence", f"{analysis.get('confidence', 0.5)*100:.1f}%")
    
    with col2:
        st.metric("Words", analysis.get('word_count', 0))
    
    with col3:
        st.metric("Exclamation", analysis.get('exclamation_count', 0))
    
    with col4:
        st.metric("Caps Words", analysis.get('caps_words', 0))
    
    # Additional Info
    if analysis.get('has_url'):
        st.info("🔗 Contains reference link")
    
    if analysis.get('number_count', 0) > 0:
        st.info(f"🔢 Contains {analysis['number_count']} numbers")
    
    # Disaster Types
    if analysis.get('detected_disasters'):
        st.markdown("#### 🌪️ Detected Disaster Types")
        cols = st.columns(len(analysis['detected_disasters']))
        for i, disaster in enumerate(analysis['detected_disasters']):
            with cols[i]:
                st.info(disaster.upper())
    
    # Reasons
    if analysis.get('reasons'):
        st.markdown("#### 🔍 Analysis Reasons")
        for reason in analysis['reasons']:
            if "DISASTER" in reason or "disaster" in reason:
                st.error(f"• {reason}")
            elif "NON-DISASTER" in reason or "non-disaster" in reason:
                st.success(f"• {reason}")
            else:
                st.warning(f"• {reason}")

def create_location_map(location, lat, lon, is_disaster):
    """Create a location map"""
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode='markers+text',
        marker=dict(
            size=20,
            color='red' if is_disaster else 'green',
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
        line=dict(width=2, color='red' if is_disaster else 'green'),
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

def display_live_stats():
    """Display live statistics"""
    stats = st.session_state["local_stats"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", stats.get("total_analyses", 0))
    with col2:
        st.metric("Disasters", stats.get("total_disaster", 0))
    with col3:
        st.metric("Non-Disasters", stats.get("total_non_disaster", 0))
    with col4:
        if stats.get("locations"):
            top_location = max(stats['locations'].items(), key=lambda x: x[1])[0]
            st.metric("Hotspot", top_location)
    
    if stats.get("locations"):
        st.markdown("### 📍 Top Locations")
        loc_df = pd.DataFrame([
            {"Location": loc, "Count": count}
            for loc, count in stats["locations"].items()
        ]).sort_values("Count", ascending=False).head(10)
        
        fig = px.bar(loc_df, x="Location", y="Count", 
                     title="Most Frequently Mentioned Locations",
                     color="Count", color_continuous_scale="blues")
        st.plotly_chart(fig, use_container_width=True)

def display_live_feed():
    """Display live feed of recent analyses"""
    
    analyses = st.session_state["local_analyses"][-20:]
    
    if analyses:
        st.markdown("### 📡 Live Analysis Feed")
        
        feed_data = []
        for a in analyses:
            feed_data.append({
                "Time": a.get("timestamp", "")[-8:],
                "Tweet": a.get("tweet", "")[:50] + "...",
                "Disaster %": f"{a.get('disaster_probability', 0)*100:.1f}%",
                "Location": a.get("location", "Unknown"),
                "Status": "🔴 DISASTER" if a.get("is_disaster") else "✅ NOT",
                "Model": a.get("model_used", "Unknown")
            })
        
        df = pd.DataFrame(feed_data)
        st.dataframe(df, use_container_width=True, height=300)

# ================================================================
# MAIN UI STARTS HERE
# ================================================================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================
connection_status = "Local Mode - Data stored in session"

# Determine which model is active
if bert_loaded:
    model_display = "🧠 BERT Model"
    model_badge_class = "bert-badge"
else:
    model_display = "🔍 Mock API"
    model_badge_class = "mock-badge"

st.markdown(
    f'''
    <div class="main-header">
        <h1 style="font-size: 3em; margin-bottom: 10px;">🤖 Disaster Tweet AI Detector</h1>
        <p style="font-size: 1.2em; opacity: 0.9;">Powered by BERT Model + Mock API</p>
        <div style="margin-top: 20px;">
            <span class="{model_badge_class}">{model_display}</span>
            <span class="status-badge local-badge">⚫ LOCAL</span>
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
    
    # Model Status
    st.markdown("### 🧠 Model Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if bert_loaded:
            st.success("✅ BERT Model")
            st.caption("Model loaded successfully")
        else:
            st.error("❌ BERT Model")
            st.caption("Using Mock API fallback")
    
    with col2:
        st.success("✅ Mock API")
        st.caption("Always available")
    
    st.markdown("---")
    
    # Model Selection
    st.markdown("### ⚙️ Settings")
    
    model_options = []
    if bert_loaded:
        model_options.append("bert")
    model_options.append("mock")
    
    model_labels = {
        "bert": "🧠 BERT Model (More Accurate)",
        "mock": "🔍 Mock API (Always Available)"
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
        "🔄 Auto-refresh Feed",
        value=True
    )
    
    st.markdown("---")
    
    # Live Statistics
    st.markdown("### 📊 Statistics")
    display_live_stats()
    
    # Clear data button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state["local_analyses"] = []
        st.session_state["local_stats"] = {
            "total_analyses": 0,
            "total_disaster": 0,
            "total_non_disaster": 0,
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
ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)

with ex_col1:
    if st.button("📰 Real Disaster", use_container_width=True):
        st.session_state["tweet_input"] = "Heavy rain in Kampar causing flash floods. According to JPS, water levels rising. 150 residents evacuated to relief centers."
        st.session_state["input_key_counter"] += 1
        st.rerun()

with ex_col2:
    if st.button("🚨 Fake Disaster", use_container_width=True):
        st.session_state["tweet_input"] = "URGENT! BREAKING: MASSIVE 8.0 earthquake in Kuala Lumpur! Thousands DEAD! Government hiding truth! SHARE NOW! 😱😱😱"
        st.session_state["input_key_counter"] += 1
        st.rerun()

with ex_col3:
    if st.button("🔄 Mixed", use_container_width=True):
        st.session_state["tweet_input"] = "URGENT! Flood in Johor! Water level 2 meters! Official source says evacuating. JPS confirms."
        st.session_state["input_key_counter"] += 1
        st.rerun()

with ex_col4:
    if st.button("📍 Location Test", use_container_width=True):
        st.session_state["tweet_input"] = "Landslide reported in Cameron Highlands - authorities responding, 3 people rescued"
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
    model_name = "BERT" if st.session_state["model_choice"] == "bert" else "Mock API"
    
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
            timestamp = datetime.now().strftime("%H:%M:%S")
            analysis_data = {
                "timestamp": timestamp,
                "tweet": tweet[:100] + "..." if len(tweet) > 100 else tweet,
                "location": location,
                "is_disaster": result.get("is_disaster"),
                "disaster_probability": result.get("disaster_probability"),
                "detected_disasters": result.get("detected_disasters", []),
                "model_used": result.get("model_used", model_name)
            }
            
            # Update stats
            st.session_state["local_analyses"].append(analysis_data)
            stats = st.session_state["local_stats"]
            stats["total_analyses"] += 1
            if result.get("is_disaster"):
                stats["total_disaster"] += 1
            else:
                stats["total_non_disaster"] += 1
            
            if location:
                stats["locations"][location] = stats["locations"].get(location, 0) + 1
            
            for disaster in result.get("detected_disasters", []):
                stats["disaster_types"][disaster] = stats["disaster_types"].get(disaster, 0) + 1
            
            stats["models_used"][result.get("model_used", "Unknown")] = stats["models_used"].get(result.get("model_used", "Unknown"), 0) + 1
            
            # Display results
            st.markdown("---")
            
            # Alert based on model
            alert_class = "bert-alert" if "BERT" in result.get("model_used", "") else "mock-alert"
            
            if result["is_disaster"]:
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
            display_probability_bar(result["disaster_probability"], result["non_disaster_probability"])
            
            # Display metrics
            display_comprehensive_metrics(result)
            
            # Location map
            if location:
                st.info(f"📍 Location detected: {location}")
                try:
                    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(location)}&format=json&limit=1"
                    headers = {'User-Agent': 'Disaster-Detector/1.0'}
                    response = requests.get(url, headers=headers, timeout=5)
                    data = response.json()
                    if data:
                        lat, lon = float(data[0]['lat']), float(data[0]['lon'])
                        st.markdown("### 🗺️ Location Map")
                        fig = create_location_map(location, lat, lon, result["is_disaster"])
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load map: {e}")

elif analyze_clicked and not tweet:
    st.warning("⚠️ Please enter a tweet")

# ================================================================
# LIVE FEED
# ================================================================
st.markdown("---")
display_live_feed()

if st.button("🔄 Refresh Feed", use_container_width=True):
    st.rerun()

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown(
    f'''
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea10, #764ba210); border-radius: 15px;">
        <p style="color: #666;">
            🤖 Disaster Tweet AI Detector | Powered by BERT Model + Mock API<br>
            Session: {st.session_state["session_id"]} | {connection_status}
        </p>
        <p style="color: #888; font-size: 0.8em;">
            ✅ BERT Model: {"LOADED" if bert_loaded else "NOT FOUND (using Mock API)"}
        </p>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
