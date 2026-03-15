# ================================================================
# ENHANCED DISASTER TWEET AI DETECTOR - CLEAN VERSION
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
from collections import deque
import base64

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Disaster Tweet AI Detector | Fake vs Real Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎯"
)

# ================================================================
# ENHANCED CSS - MODERN DARK/LIGHT THEME
# ================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main container with glass morphism */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 30px;
        margin: 20px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    /* Animated Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #48bb78 100%);
        padding: 40px;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        animation: gradientShift 10s ease infinite;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Fake/Real Badges */
    .fake-badge {
        background: linear-gradient(135deg, #ff6b6b, #ee5253);
        color: white;
        padding: 15px 30px;
        border-radius: 50px;
        font-size: 1.5em;
        font-weight: 800;
        display: inline-block;
        margin: 10px;
        box-shadow: 0 10px 20px rgba(255, 107, 107, 0.3);
        animation: pulse 2s infinite;
    }
    
    .real-badge {
        background: linear-gradient(135deg, #10ac84, #1dd1a1);
        color: white;
        padding: 15px 30px;
        border-radius: 50px;
        font-size: 1.5em;
        font-weight: 800;
        display: inline-block;
        margin: 10px;
        box-shadow: 0 10px 20px rgba(16, 172, 132, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 10px 25px;
        border-radius: 50px;
        font-size: 1em;
        font-weight: 600;
        margin: 10px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0% { box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        50% { box-shadow: 0 5px 25px rgba(102,126,234,0.5); }
        100% { box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
    }
    
    .live-badge {
        background: linear-gradient(135deg, #ff6b6b, #ee5253);
        color: white;
    }
    
    .model-badge {
        background: linear-gradient(135deg, #667eea, #5a67d8);
        color: white;
    }
    
    /* Fake/Real Indicator Cards */
    .indicator-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid;
        transition: all 0.3s ease;
    }
    
    .fake-indicator {
        border-left-color: #ff6b6b;
        background: linear-gradient(90deg, #fff5f5, white);
    }
    
    .real-indicator {
        border-left-color: #10ac84;
        background: linear-gradient(90deg, #f0fff4, white);
    }
    
    .indicator-card:hover {
        transform: translateX(5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-item {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-item:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 30px rgba(102,126,234,0.2);
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: 800;
        color: #667eea;
        line-height: 1.2;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Input area */
    .stTextArea textarea {
        border-radius: 20px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 20px !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        background: white !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05) !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 5px 25px rgba(102,126,234,0.3) !important;
        transform: scale(1.02);
    }
    
    /* Button styles */
    .stButton button {
        border-radius: 15px !important;
        padding: 12px 25px !important;
        font-weight: 600 !important;
        font-size: 1em !important;
        transition: all 0.3s ease !important;
        border: none !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 30px rgba(102,126,234,0.4) !important;
    }
    
    /* Primary button */
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #667eea, #5a67d8) !important;
        color: white !important;
        font-size: 1.2em !important;
        padding: 15px 30px !important;
    }
    
    /* Clear button */
    .clear-button {
        background: linear-gradient(135deg, #95a5a6, #7f8c8d) !important;
        color: white !important;
    }
    
    /* Alert styles */
    .alert-box {
        padding: 25px;
        border-radius: 20px;
        color: white;
        font-size: 1.3em;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        animation: slideIn 0.5s ease-out;
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    }
    
    .fake-alert {
        background: linear-gradient(135deg, #ff6b6b, #ee5253);
    }
    
    .real-alert {
        background: linear-gradient(135deg, #10ac84, #1dd1a1);
    }
    
    @keyframes slideIn {
        from {
            transform: translateY(-20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    /* Confidence meter */
    .confidence-meter {
        width: 100%;
        height: 20px;
        background: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #5a67d8);
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea10, #764ba210);
        border-radius: 20px;
        margin-top: 30px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            padding: 15px;
            margin: 10px;
        }
        
        .fake-badge, .real-badge {
            font-size: 1.2em;
            padding: 10px 20px;
        }
        
        .metric-value {
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
if "input_key_counter" not in st.session_state:
    st.session_state["input_key_counter"] = 0
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True
if "analyses" not in st.session_state:
    st.session_state["analyses"] = deque(maxlen=50)
if "stats" not in st.session_state:
    st.session_state["stats"] = {
        "total": 0,
        "fake": 0,
        "real": 0,
        "locations": {},
        "keywords": {}
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

# ================================================================
# FAKE NEWS DETECTION PATTERNS
# ================================================================

FAKE_PATTERNS = {
    "urgency": {
        "keywords": ['urgent', 'breaking', 'alert', 'warning', '!!!', '🚨', 'segera', 'penting', 'asap'],
        "weight": 1.5,
        "icon": "🚨",
        "description": "Urgency markers - common in fake news"
    },
    "sensational": {
        "keywords": ['unbelievable', 'shocking', 'massive', 'worst ever', 'catastrophic', 'devastating', 
                    'horrific', 'terrible', 'never seen', 'unprecedented', 'historical'],
        "weight": 2.0,
        "icon": "📢",
        "description": "Sensational language - exaggerates the situation"
    },
    "conspiracy": {
        "keywords": ['government hiding', 'they don\'t want you', 'secret', 'hidden truth', 'cover up',
                    'censored', 'banned', 'suppressed', 'lying to you'],
        "weight": 2.5,
        "icon": "🔮",
        "description": "Conspiracy theories - undermines official sources"
    },
    "sharing": {
        "keywords": ['share', 'spread', 'viral', 'forward', 'retweet', 'repost', 'kongsi', 'sebarkan',
                    'tell everyone', 'pass it on', 'go viral'],
        "weight": 1.5,
        "icon": "🔄",
        "description": "Share requests - attempts to spread misinformation"
    },
    "emotional": {
        "keywords": ['pray', 'please help', 'omg', '😱', '🙏', 'tragedy', 'heartbreaking', 'crying',
                    'terrified', 'scared', 'help them', 'save them'],
        "weight": 1.0,
        "icon": "😢",
        "description": "Emotional manipulation - plays on feelings"
    },
    "vague": {
        "keywords": ['they say', 'rumors', 'allegedly', 'someone said', 'apparently', 'heard that',
                    'kata orang', 'konon', 'maybe', 'perhaps', 'could be'],
        "weight": 1.2,
        "icon": "❓",
        "description": "Vague sources - no clear attribution"
    }
}

# ================================================================
# REAL NEWS DETECTION PATTERNS
# ================================================================

REAL_PATTERNS = {
    "official_sources": {
        "keywords": ['according to', 'reported by', 'official', 'authorities', 'confirmed', 'verified',
                    'jps', 'met malaysia', 'jabatan bomba', 'bomba', 'polis', 'police', 'nadma',
                    'statement from', 'press release', 'minister', 'department'],
        "weight": 2.0,
        "icon": "📰",
        "description": "Official sources - credible information"
    },
    "specific_data": {
        "keywords": ['meter', 'km', 'mm', 'celsius', 'magnitude', 'level', 'depth', 'volume',
                    'number of', 'total', 'count', 'estimated', 'approximately'],
        "weight": 1.5,
        "icon": "📊",
        "description": "Specific measurements - adds credibility"
    },
    "details": {
        "keywords": ['at', 'on', 'date', 'time', 'location', 'coordinates', 'reported at',
                    'occurred at', 'happened at', 'situated at'],
        "weight": 1.2,
        "icon": "📍",
        "description": "Specific details - provides context"
    },
    "url": {
        "keywords": ['http', 'https', 'www.', '.com', '.my', '.gov', '.org'],
        "weight": 1.8,
        "icon": "🔗",
        "description": "Reference link - can verify information"
    }
}

# ================================================================
# DISASTER KEYWORDS
# ================================================================

DISASTER_KEYWORDS = {
    "🌊 Flood": ['flood', 'banjir', 'inundation', 'water level', 'banjir kilat', 'air naik'],
    "🌋 Earthquake": ['earthquake', 'gempa', 'tremor', 'seismic', 'gempa bumi', 'gempabumi'],
    "⛈️ Storm": ['storm', 'thunderstorm', 'ribut', 'petir', 'lightning', 'ribut petir', 'badai'],
    "🏔️ Landslide": ['landslide', 'tanah runtuh', 'mudslide', 'runtuh', 'tanah longsor'],
    "🔥 Fire": ['fire', 'kebakaran', 'burning', 'api', 'terbakar'],
    "🌊 Tsunami": ['tsunami', 'tidal wave', 'gelombang tsunami', 'ombak besar'],
    "💨 Wind": ['wind', 'angin', 'gale', 'tornado', 'angin kencang', 'puting beliung'],
    "🌫️ Haze": ['haze', 'jerebu', 'smoke', 'asap', 'kabut asap'],
    "☀️ Heatwave": ['heatwave', 'gelombang panas', 'heat stroke', 'suhu tinggi']
}

# ================================================================
# PREPROCESSING FUNCTION
# ================================================================

def preprocess_tweet(text):
    """Clean and preprocess tweet text"""
    if pd.isna(text) or text == "":
        return {
            "clean": "",
            "original": "",
            "word_count": 0,
            "char_count": 0,
            "exclamation": 0,
            "question": 0,
            "caps_words": 0,
            "has_url": False,
            "numbers": 0,
            "urls": []
        }
    
    # Save original for metrics
    original = text
    
    # Extract URLs
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', original)
    
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = [word for word in text.split() if word not in stop_words]
    
    return {
        "clean": " ".join(tokens),
        "original": original,
        "word_count": len(original.split()),
        "char_count": len(original),
        "exclamation": original.count('!'),
        "question": original.count('?'),
        "caps_words": sum(1 for word in original.split() if word.isupper() and len(word) > 2),
        "has_url": len(urls) > 0,
        "numbers": len(re.findall(r'\d+', original)),
        "urls": urls
    }

# ================================================================
# FAKE VS REAL DETECTION FUNCTION
# ================================================================

def detect_fake_vs_real(text):
    """
    Advanced detection of fake vs real disaster tweets
    Returns detailed analysis with scores and indicators
    """
    start_time = time.time()
    
    # Preprocess
    processed = preprocess_tweet(text)
    text_lower = text.lower()
    
    # Initialize scoring
    fake_score = 0
    real_score = 0
    fake_indicators = []
    real_indicators = []
    detected_disasters = []
    
    # ================================================================
    # FAKE NEWS DETECTION
    # ================================================================
    
    # Check each fake pattern
    for pattern_name, pattern in FAKE_PATTERNS.items():
        matches = [kw for kw in pattern["keywords"] if kw in text_lower]
        if matches:
            match_count = len(matches)
            points = match_count * pattern["weight"]
            fake_score += points
            
            fake_indicators.append({
                "type": pattern_name,
                "icon": pattern["icon"],
                "matches": matches[:5],
                "count": match_count,
                "points": points,
                "description": pattern["description"]
            })
    
    # Check for excessive exclamation marks
    if processed["exclamation"] > 3:
        extra_points = processed["exclamation"] * 0.5
        fake_score += extra_points
        fake_indicators.append({
            "type": "excessive_exclamation",
            "icon": "❗",
            "matches": [f"{processed['exclamation']} exclamation marks"],
            "count": processed["exclamation"],
            "points": extra_points,
            "description": "Excessive exclamation marks - emotional manipulation"
        })
    
    # Check for ALL CAPS words
    if processed["caps_words"] > 2:
        extra_points = processed["caps_words"] * 0.5
        fake_score += extra_points
        fake_indicators.append({
            "type": "all_caps",
            "icon": "📣",
            "matches": [f"{processed['caps_words']} words in ALL CAPS"],
            "count": processed["caps_words"],
            "points": extra_points,
            "description": "SHOUTING - attempts to create urgency"
        })
    
    # ================================================================
    # REAL NEWS DETECTION
    # ================================================================
    
    # Check each real pattern
    for pattern_name, pattern in REAL_PATTERNS.items():
        matches = [kw for kw in pattern["keywords"] if kw in text_lower]
        if matches:
            match_count = len(matches)
            points = match_count * pattern["weight"]
            real_score += points
            
            real_indicators.append({
                "type": pattern_name,
                "icon": pattern["icon"],
                "matches": matches[:5],
                "count": match_count,
                "points": points,
                "description": pattern["description"]
            })
    
    # Check for specific numbers (credibility boost)
    if processed["numbers"] > 2:
        extra_points = processed["numbers"] * 0.3
        real_score += extra_points
        real_indicators.append({
            "type": "specific_numbers",
            "icon": "🔢",
            "matches": [f"{processed['numbers']} numbers found"],
            "count": processed["numbers"],
            "points": extra_points,
            "description": "Specific numbers - adds credibility"
        })
    
    # Check for URLs (can verify information)
    if processed["has_url"]:
        extra_points = 2.0
        real_score += extra_points
        real_indicators.append({
            "type": "has_url",
            "icon": "🔗",
            "matches": processed["urls"][:2],
            "count": len(processed["urls"]),
            "points": extra_points,
            "description": "Contains reference links - can verify"
        })
    
    # Check for detailed information
    if processed["word_count"] > 20:
        extra_points = 1.0
        real_score += extra_points
        real_indicators.append({
            "type": "detailed",
            "icon": "📝",
            "matches": [f"{processed['word_count']} words"],
            "count": 1,
            "points": extra_points,
            "description": "Detailed information - more credible"
        })
    
    # ================================================================
    # DISASTER TYPE DETECTION
    # ================================================================
    
    for disaster, keywords in DISASTER_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if matches:
            detected_disasters.append({
                "name": disaster,
                "keywords": matches[:3],
                "count": len(matches)
            })
    
    # ================================================================
    # PROBABILITY CALCULATION
    # ================================================================
    
    total_score = fake_score + real_score
    if total_score > 0:
        fake_probability = fake_score / total_score
        real_probability = real_score / total_score
    else:
        fake_probability = 0.5
        real_probability = 0.5
    
    # Determine if fake or real
    is_fake = fake_probability > 0.5
    
    # Calculate confidence (distance from 0.5)
    confidence = abs(fake_probability - 0.5) * 2
    
    # Determine verdict
    if confidence > 0.8:
        verdict = "HIGH CONFIDENCE"
    elif confidence > 0.6:
        verdict = "MEDIUM CONFIDENCE"
    elif confidence > 0.4:
        verdict = "LOW CONFIDENCE"
    else:
        verdict = "UNCERTAIN"
    
    # Generate final reasons
    reasons = []
    
    if is_fake:
        reasons.append(f"🔴 **FAKE DISASTER TWEET** - {verdict}")
        if fake_indicators:
            reasons.append("📊 **Fake Indicators Found:**")
            for ind in fake_indicators[:3]:
                reasons.append(f"  {ind['icon']} {ind['description']} ({ind['count']} matches)")
    else:
        reasons.append(f"✅ **REAL DISASTER TWEET** - {verdict}")
        if real_indicators:
            reasons.append("📊 **Real Indicators Found:**")
            for ind in real_indicators[:3]:
                reasons.append(f"  {ind['icon']} {ind['description']} ({ind['count']} matches)")
    
    if detected_disasters:
        reasons.append("🌪️ **Disaster Types Detected:**")
        for d in detected_disasters:
            reasons.append(f"  {d['name']}")
    
    return {
        "is_fake": is_fake,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "fake_score": fake_score,
        "real_score": real_score,
        "confidence": confidence,
        "verdict": verdict,
        "fake_indicators": fake_indicators,
        "real_indicators": real_indicators,
        "detected_disasters": detected_disasters,
        "reasons": reasons,
        "metrics": processed,
        "response_time": time.time() - start_time
    }

# ================================================================
# DISPLAY FUNCTIONS
# ================================================================

def display_fake_real_banner(result):
    """Display fake/real detection banner"""
    
    if result["is_fake"]:
        st.markdown(
            f'''
            <div class="alert-box fake-alert">
                🔴 FAKE DISASTER TWEET DETECTED
                <div style="font-size: 0.7em; margin-top: 10px;">
                    Confidence: {result["confidence"]*100:.1f}% | {result["verdict"]}
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'''
            <div class="alert-box real-alert">
                ✅ REAL DISASTER TWEET - CREDIBLE SOURCE
                <div style="font-size: 0.7em; margin-top: 10px;">
                    Confidence: {result["confidence"]*100:.1f}% | {result["verdict"]}
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

def display_probability_meter(result):
    """Display fake vs real probability meter"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🔴 Fake Probability")
        st.markdown(f"<h2 style='color: #ff6b6b;'>{result['fake_probability']*100:.1f}%</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='confidence-meter'><div class='confidence-fill' style='width: {result['fake_probability']*100}%; background: #ff6b6b;'></div></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### ✅ Real Probability")
        st.markdown(f"<h2 style='color: #10ac84;'>{result['real_probability']*100:.1f}%</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='confidence-meter'><div class='confidence-fill' style='width: {result['real_probability']*100}%; background: #10ac84;'></div></div>", unsafe_allow_html=True)

def display_indicators(result):
    """Display fake and real indicators"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚨 Fake News Indicators")
        if result["fake_indicators"]:
            for ind in result["fake_indicators"]:
                st.markdown(f"""
                <div class="indicator-card fake-indicator">
                    <div style="display: flex; justify-content: space-between;">
                        <span><strong>{ind['icon']} {ind['type'].replace('_', ' ').title()}</strong></span>
                        <span style="color: #ff6b6b;">+{ind['points']:.1f} points</span>
                    </div>
                    <div style="color: #666; font-size: 0.9em;">{ind['description']}</div>
                    <div style="color: #888; font-size: 0.8em; margin-top: 5px;">
                        Found: {', '.join(ind['matches'][:3])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No fake news indicators detected")
    
    with col2:
        st.markdown("### ✅ Real News Indicators")
        if result["real_indicators"]:
            for ind in result["real_indicators"]:
                st.markdown(f"""
                <div class="indicator-card real-indicator">
                    <div style="display: flex; justify-content: space-between;">
                        <span><strong>{ind['icon']} {ind['type'].replace('_', ' ').title()}</strong></span>
                        <span style="color: #10ac84;">+{ind['points']:.1f} points</span>
                    </div>
                    <div style="color: #666; font-size: 0.9em;">{ind['description']}</div>
                    <div style="color: #888; font-size: 0.8em; margin-top: 5px;">
                        Found: {', '.join(ind['matches'][:3])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No real news indicators detected")

def display_metrics(result):
    """Display text metrics"""
    
    metrics = result["metrics"]
    
    cols = st.columns(5)
    metric_data = [
        ("Words", metrics["word_count"], "📝"),
        ("Exclamation", metrics["exclamation"], "❗"),
        ("ALL CAPS", metrics["caps_words"], "📣"),
        ("Numbers", metrics["numbers"], "🔢"),
        ("URLs", len(metrics["urls"]), "🔗")
    ]
    
    for i, (label, value, icon) in enumerate(metric_data):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{icon} {label}</div>
            </div>
            """, unsafe_allow_html=True)

def create_location_map(location):
    """Create an interactive map with location"""
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(location)}&format=json&limit=1"
        headers = {'User-Agent': 'Disaster-Detector/1.0'}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        
        if data:
            lat, lon = float(data[0]['lat']), float(data[0]['lon'])
            
            fig = go.Figure()
            
            # Add marker
            fig.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode='markers+text',
                marker=dict(
                    size=25,
                    color='red',
                    symbol='marker'
                ),
                text=[location],
                textposition="top center",
                textfont=dict(size=14, color='black', family='Arial Black')
            ))
            
            # Add 5km radius circle
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
                line=dict(width=2, color='rgba(255,0,0,0.5)', dash='dot'),
                name='5km Radius'
            ))
            
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=lat, lon=lon),
                    zoom=12
                ),
                height=400,
                margin={"r": 0, "t": 30, "l": 0, "b": 0},
                title={
                    'text': f"📍 {location} Area",
                    'y': 0.98,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=16, color='#333', family='Arial Black')
                }
            )
            
            return fig
    except Exception as e:
        return None
    
    return None

# ================================================================
# MAIN UI
# ================================================================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(
        f'''
        <div class="main-header">
            <h1 style="font-size: 3.5em; margin-bottom: 10px;">🎯 Disaster AI</h1>
            <p style="font-size: 1.3em; opacity: 0.95;">Fake vs Real Disaster Tweet Analysis</p>
            <div style="margin-top: 20px;">
                <span class="status-badge live-badge">🔴 LIVE</span>
                <span class="status-badge model-badge">🧠 Fake/Real Detector</span>
            </div>
            <p style="margin-top: 20px; font-size: 0.9em; opacity: 0.8;">
                Session: {st.session_state["session_id"]}
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )

# ================================================================
# SIDEBAR - STATS
# ================================================================
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea20, #764ba220); padding: 20px; border-radius: 20px; margin-bottom: 20px;">
        <h3 style="color: #333; margin-bottom: 15px;">📊 Detection Statistics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # SAFELY access stats with .get() to avoid KeyError
    stats = st.session_state["stats"]
    
    col1, col2 = st.columns(2)
    with col1:
        fake_count = stats.get("fake", 0)
        st.markdown(f"""
        <div style="background: white; padding: 20px; border-radius: 15px; text-align: center;">
            <div style="font-size: 2.5em; font-weight: 800; color: #ff6b6b;">{fake_count}</div>
            <div style="color: #666;">Fake Detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        real_count = stats.get("real", 0)
        st.markdown(f"""
        <div style="background: white; padding: 20px; border-radius: 15px; text-align: center;">
            <div style="font-size: 2.5em; font-weight: 800; color: #10ac84;">{real_count}</div>
            <div style="color: #666;">Real Detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    total_count = stats.get("total", 0)
    st.markdown(f"""
    <div style="background: white; padding: 20px; border-radius: 15px; text-align: center; margin-top: 10px;">
        <div style="font-size: 2.5em; font-weight: 800; color: #667eea;">{total_count}</div>
        <div style="color: #666;">Total Analyses</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    auto_refresh = st.toggle("🔄 Auto-refresh", value=st.session_state.get("auto_refresh", True))
    st.session_state["auto_refresh"] = auto_refresh
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state["analyses"] = deque(maxlen=50)
        st.session_state["stats"] = {"total": 0, "fake": 0, "real": 0, "locations": {}, "keywords": {}}
        st.rerun()

# ================================================================
# AUTO-REFRESH LOGIC
# ================================================================
if st.session_state.get("auto_refresh", True):
    if time.time() - st.session_state.get("last_refresh", time.time()) > 5:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

# ================================================================
# INPUT SECTION - Clean version with only Clear button
# ================================================================
st.markdown("### 📝 Enter Tweet for Fake/Real Analysis")

# Create two columns for input and clear button
input_col, clear_col = st.columns([6, 1])

with input_col:
    input_key = f"tweet_input_{st.session_state['input_key_counter']}"
    tweet = st.text_area(
        "",
        height=120,
        placeholder="Type or paste a tweet here to analyze...",
        key=input_key,
        label_visibility="collapsed"
    )

with clear_col:
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing for alignment
    if st.button("🗑️ Clear", use_container_width=True, help="Clear input field"):
        st.session_state["input_key_counter"] += 1
        st.rerun()

# Add a simple instruction line
st.caption("Enter any tweet above and click 'Analyze Fake/Real' to check if it's fake or real disaster news.")

# ================================================================
# ANALYZE BUTTON
# ================================================================
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_clicked = st.button("🔍 Analyze Fake/Real", type="primary", use_container_width=True)

# ================================================================
# ANALYSIS EXECUTION
# ================================================================
if analyze_clicked and tweet:
    with st.spinner("🎯 Analyzing for fake vs real indicators..."):
        result = detect_fake_vs_real(tweet)
        
        if result:
            # Extract location
            location = None
            for loc in MALAYSIA_LOCATIONS:
                if loc.lower() in tweet.lower():
                    location = loc
                    break
            
            # Update stats
            st.session_state["stats"]["total"] = st.session_state["stats"].get("total", 0) + 1
            if result["is_fake"]:
                st.session_state["stats"]["fake"] = st.session_state["stats"].get("fake", 0) + 1
            else:
                st.session_state["stats"]["real"] = st.session_state["stats"].get("real", 0) + 1
            
            if location:
                if "locations" not in st.session_state["stats"]:
                    st.session_state["stats"]["locations"] = {}
                st.session_state["stats"]["locations"][location] = st.session_state["stats"]["locations"].get(location, 0) + 1
            
            # Save analysis
            analysis_record = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "tweet": tweet[:50] + "..." if len(tweet) > 50 else tweet,
                "is_fake": result["is_fake"],
                "confidence": result["confidence"],
                "location": location if location else "Unknown",
                "fake_probability": result["fake_probability"],
                "real_probability": result["real_probability"]
            }
            st.session_state["analyses"].append(analysis_record)
            
            # Display results
            st.markdown("---")
            
            # Fake/Real Banner
            display_fake_real_banner(result)
            
            # Probability Meter
            st.markdown("### 📊 Probability Analysis")
            display_probability_meter(result)
            
            # Score Summary
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fake Score", f"{result['fake_score']:.1f} points")
            with col2:
                st.metric("Real Score", f"{result['real_score']:.1f} points")
            
            # Indicators
            display_indicators(result)
            
            # Text Metrics
            st.markdown("### 📝 Text Analysis")
            display_metrics(result)
            
            # Disaster Types
            if result["detected_disasters"]:
                st.markdown("### 🌪️ Disaster Types Detected")
                disaster_cols = st.columns(len(result["detected_disasters"]))
                for i, d in enumerate(result["detected_disasters"]):
                    with disaster_cols[i]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #ff6b6b20, #ee525320); 
                                    padding: 15px; border-radius: 15px; text-align: center;
                                    border: 2px solid #ff6b6b;">
                            <span style="font-size: 2em;">{d['name'].split()[0]}</span><br>
                            <strong>{' '.join(d['name'].split()[1:])}</strong>
                            <div style="font-size: 0.8em; color: #666;">{d['count']} keywords</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Location map
            if location:
                st.markdown(f"### 🗺️ Location: {location}")
                fig = create_location_map(location)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Reasons
            if result["reasons"]:
                st.markdown("### 🔍 Analysis Summary")
                for reason in result["reasons"]:
                    if "FAKE" in reason:
                        st.error(reason)
                    elif "REAL" in reason:
                        st.success(reason)
                    else:
                        st.info(reason)
            
            # Performance
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea10, #764ba210); 
                        padding: 15px; border-radius: 15px; margin-top: 20px;
                        display: flex; justify-content: space-between;">
                <span>⚡ Response time: {result['response_time']*1000:.0f}ms</span>
                <span>🤖 Model: Fake/Real Detector v2.0</span>
                <span>🎯 Verdict: {result['verdict']}</span>
            </div>
            """, unsafe_allow_html=True)

elif analyze_clicked and not tweet:
    st.warning("⚠️ Please enter a tweet for analysis")

# ================================================================
# LIVE FEED
# ================================================================
if st.session_state["analyses"]:
    st.markdown("---")
    st.markdown("### 📡 Recent Analyses")
    
    feed_data = []
    for a in reversed(list(st.session_state["analyses"])[-10:]):
        # Use .get() with defaults to avoid KeyError
        is_fake = a.get("is_fake", False)
        confidence = a.get("confidence", 0.5)
        location = a.get("location", "Unknown")
        tweet_text = a.get("tweet", "Unknown tweet")
        timestamp = a.get("timestamp", datetime.now().strftime("%H:%M:%S"))
        
        feed_data.append({
            "Time": timestamp,
            "Tweet": tweet_text,
            "Prediction": "🔴 FAKE" if is_fake else "✅ REAL",
            "Confidence": f"{confidence*100:.1f}%",
            "Location": location
        })
    
    df = pd.DataFrame(feed_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown(
    f'''
    <div class="footer">
        <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 20px;">
            <span style="background: #ff6b6b; color: white; padding: 8px 20px; border-radius: 30px;">🔴 Fake Detector</span>
            <span style="background: #10ac84; color: white; padding: 8px 20px; border-radius: 30px;">✅ Real Detector</span>
            <span style="background: #667eea; color: white; padding: 8px 20px; border-radius: 30px;">🎯 v2.0</span>
        </div>
        <p style="color: #666;">
            Disaster Tweet AI Detector | Advanced Fake vs Real Analysis<br>
            Session: {st.session_state["session_id"]} | Last sync: {datetime.now().strftime('%H:%M:%S')}
        </p>
        <p style="color: #888; font-size: 0.9em; margin-top: 15px;">
            ✅ Real-time fake news detection • 10+ indicators • Confidence scoring
        </p>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
