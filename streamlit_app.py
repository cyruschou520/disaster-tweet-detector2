# ================================================================
# DISASTER TWEET AI DETECTOR - WITH BERT MODEL
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
import hashlib
import nltk
from nltk.corpus import stopwords
import os
import zipfile
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Disaster Tweet AI Detector - BERT Powered",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)

# ================================================================
# DOWNLOAD NLTK DATA
# ================================================================
@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))

stop_words = download_nltk_data()

# ================================================================
# SESSION STATE
# ================================================================
if "session_id" not in st.session_state:
    st.session_state["session_id"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "bert"
if "input_key_counter" not in st.session_state:
    st.session_state["input_key_counter"] = 0
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True
if "local_analyses" not in st.session_state:
    st.session_state["local_analyses"] = []

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
    "wind": ['wind', 'angin', 'gale', 'tornado', 'angin kencang']
}

# ================================================================
# LOAD BERT MODEL FROM GITHUB RELEASES
# ================================================================

@st.cache_resource(show_spinner="🔄 Loading BERT model from GitHub...")
def load_bert_model():
    """Download and load BERT model from GitHub Releases"""
    model_path = "bert_disaster_model_fine_tuned"
    
    # Check if model already exists
    if os.path.exists(model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            model.eval()
            return model, tokenizer, True
        except:
            pass
    
    # Download from GitHub Releases
    github_url = "https://github.com/cyruschou520/disaster-tweet-detector2/releases/download/v1.0.0/bert_disaster_model_fine_tuned.zip"
    
    try:
        with st.spinner("Downloading BERT model from GitHub (this may take a few minutes)..."):
            response = requests.get(github_url, stream=True)
            response.raise_for_status()
            
            # Download with progress
            zip_path = "bert_model.zip"
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Clean up
            os.remove(zip_path)
            
            # Load model
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            model.eval()
            
            return model, tokenizer, True
            
    except Exception as e:
        st.warning(f"Could not load BERT model: {e}")
        return None, None, False

# Load BERT model (will fail gracefully)
bert_model, bert_tokenizer, bert_loaded = load_bert_model()

# ================================================================
# PREPROCESSING
# ================================================================

def preprocess_tweet(text):
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
# BERT PREDICTION
# ================================================================

def predict_with_bert(text):
    if not bert_loaded:
        return None
    
    clean_text = preprocess_tweet(text)
    if not clean_text:
        return None
    
    inputs = bert_tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(outputs.logits, dim=1).item()
        confidence = probs[0][pred].item()
    
    return {
        "is_disaster": pred == 1,
        "probability": probs[0][1].item(),
        "confidence": confidence
    }

# ================================================================
# MOCK PREDICTION (FALLBACK)
# ================================================================

def predict_with_mock(text):
    text_lower = text.lower()
    
    disaster_score = 0
    non_disaster_score = 0
    
    disaster_keywords = ['flood', 'banjir', 'earthquake', 'fire', 'storm', 'urgent', 'emergency']
    non_disaster_keywords = ['good morning', 'hello', 'thanks', 'happy', 'love', 'nice']
    
    for kw in disaster_keywords:
        if kw in text_lower:
            disaster_score += 1
    
    for kw in non_disaster_keywords:
        if kw in text_lower:
            non_disaster_score += 1
    
    total = disaster_score + non_disaster_score
    disaster_prob = disaster_score / total if total > 0 else 0.5
    
    return {
        "is_disaster": disaster_prob > 0.5,
        "probability": disaster_prob,
        "confidence": abs(disaster_prob - 0.5) * 2
    }

# ================================================================
# MAIN ANALYSIS
# ================================================================

def analyze_tweet(text, use_bert=True):
    if use_bert and bert_loaded:
        result = predict_with_bert(text)
        if result:
            return result, "BERT"
    return predict_with_mock(text), "Mock AI"

# ================================================================
# DISPLAY FUNCTIONS
# ================================================================

def display_probability_bar(disaster_prob):
    st.markdown(f"""
    <div style="background: #f0f0f0; border-radius: 10px; padding: 5px; margin: 20px 0;">
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #ff4444;">DISASTER: {disaster_prob*100:.1f}%</span>
            <span style="color: #00cc66;">NON-DISASTER: {(1-disaster_prob)*100:.1f}%</span>
        </div>
        <div style="height: 30px; border-radius: 5px; overflow: hidden; display: flex;">
            <div style="width: {disaster_prob*100}%; background: #ff4444;"></div>
            <div style="width: {(1-disaster_prob)*100}%; background: #00cc66;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# MAIN UI
# ================================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px; border-radius: 20px; color: white; text-align: center; margin-bottom: 30px;
    }
    .bert-badge {
        background: linear-gradient(135deg, #667eea, #5a67d8);
        color: white; padding: 8px 20px; border-radius: 30px; display: inline-block; margin: 10px 0;
        font-weight: bold;
    }
    .mock-badge {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white; padding: 8px 20px; border-radius: 30px; display: inline-block; margin: 10px 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
model_status = "BERT" if bert_loaded else "Mock AI"
badge_class = "bert-badge" if bert_loaded else "mock-badge"

st.markdown(
    f'''
    <div class="main-header">
        <h1 style="font-size: 3em;">🤖 Disaster Tweet AI Detector</h1>
        <p style="font-size: 1.2em;">Powered by {model_status}</p>
        <div><span class="{badge_class}">Active: {model_status}</span></div>
        <p style="margin-top: 15px;">Session: {st.session_state["session_id"]}</p>
    </div>
    ''',
    unsafe_allow_html=True
)

# Model selector (only show if BERT is available)
if bert_loaded:
    use_bert = st.sidebar.radio("Select Model", ["BERT (More Accurate)", "Mock AI (Faster)"]) == "BERT (More Accurate)"
else:
    use_bert = False
    st.sidebar.info("🔍 BERT model not available - using Mock AI")

# Input
tweet = st.text_area("Enter tweet:", height=100, placeholder="Example: Heavy rain in Kampar causing flash floods...")

if st.button("Analyze", type="primary"):
    if tweet:
        result, model_used = analyze_tweet(tweet, use_bert)
        
        st.markdown("---")
        
        if result["is_disaster"]:
            st.error(f"🔴 DISASTER TWEET DETECTED")
        else:
            st.success(f"✅ NOT A DISASTER TWEET")
        
        st.info(f"Model: {model_used} | Confidence: {result['confidence']*100:.1f}%")
        display_probability_bar(result["probability"])
    else:
        st.warning("Please enter a tweet")

# Footer
st.markdown("---")
st.markdown("🚀 Disaster Tweet AI Detector | GitHub Releases + BERT")
