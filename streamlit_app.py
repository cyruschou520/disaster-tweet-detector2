# ================================================================
# ENHANCED DISASTER TWEET AI DETECTOR - WITH FIGURATIVE LANGUAGE DETECTION
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
    page_title="Disaster Tweet AI Detector | Smart Classification",
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
    
    /* Classification Badges */
    .normal-badge {
        background: linear-gradient(135deg, #95a5a6, #7f8c8d);
        color: white;
        padding: 15px 30px;
        border-radius: 50px;
        font-size: 1.5em;
        font-weight: 800;
        display: inline-block;
        margin: 10px;
        box-shadow: 0 10px 20px rgba(149, 165, 166, 0.3);
        animation: pulse 2s infinite;
    }
    
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
    
    /* Figurative Language Indicator */
    .figurative-indicator {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
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
    .stButton button[key="clear_button"] {
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
    
    .normal-alert {
        background: linear-gradient(135deg, #95a5a6, #7f8c8d);
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
        
        .normal-badge, .fake-badge, .real-badge {
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
        "normal": 0,
        "fake": 0,
        "real": 0,
        "figurative": 0,
        "locations": {},
        "keywords": {}
    }
if "tweet_input" not in st.session_state:
    st.session_state["tweet_input"] = ""

# ================================================================
# CALLBACK FUNCTION FOR CLEAR BUTTON
# ================================================================
def clear_input():
    """Callback function to clear the tweet input"""
    st.session_state["tweet_input"] = ""
    st.session_state["input_key_counter"] += 1

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

NORMAL_KEYWORDS = [
    'exam', 'result', 'school', 'college', 'university', 'student', 'teacher',
    'work', 'job', 'office', 'colleague', 'boss', 'meeting', 'presentation',
    'food', 'lunch', 'dinner', 'breakfast', 'restaurant', 'cafe', 'coffee',
    'movie', 'film', 'show', 'netflix', 'series', 'episode',
    'music', 'song', 'playlist', 'concert', 'band',
    'sports', 'game', 'match', 'team', 'player', 'score',
    'family', 'friend', 'party', 'celebration', 'birthday', 'wedding',
    'shopping', 'store', 'mall', 'market', 'price', 'sale',
    'weather', 'sunny', 'rainy', 'cloudy', 'temperature',
    'morning', 'afternoon', 'evening', 'night', 'weekend', 'holiday',
    'happy', 'sad', 'excited', 'tired', 'bored', 'stressed',
    'love', 'hate', 'like', 'dislike', 'feel', 'feeling'
]

# ================================================================
# FIGURATIVE LANGUAGE PATTERNS
# ================================================================

FIGURATIVE_PATTERNS = {
    "tsunami": {
        "context_words": ['exam', 'result', 'grade', 'score', 'test', 'homework', 'assignment'],
        "description": "Using 'tsunami' figuratively to describe overwhelming amount"
    },
    "flood": {
        "context_words": ['email', 'work', 'task', 'assignment', 'homework', 'message', 'notification'],
        "description": "Using 'flood' figuratively to describe overwhelming quantity"
    },
    "earthquake": {
        "context_words": ['news', 'announcement', 'result', 'change', 'surprise', 'shock'],
        "description": "Using 'earthquake' figuratively to describe shocking news"
    },
    "storm": {
        "context_words": ['argument', 'fight', 'debate', 'controversy', 'drama'],
        "description": "Using 'storm' figuratively to describe conflict"
    },
    "fire": {
        "context_words": ['energy', 'motivation', 'passion', 'excitement', 'determination'],
        "description": "Using 'fire' figuratively to describe enthusiasm"
    },
    "destroyed": {
        "context_words": ['exam', 'result', 'game', 'match', 'competition', 'feeling', 'emotion'],
        "description": "Using 'destroyed' figuratively to describe failure or strong emotion"
    },
    "devastated": {
        "context_words": ['news', 'result', 'outcome', 'feeling', 'emotion', 'relationship'],
        "description": "Using 'devastated' figuratively to describe emotional state"
    }
}

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
# FIGURATIVE LANGUAGE DETECTION
# ================================================================

def detect_figurative_language(text):
    """
    Detect if disaster keywords are being used figuratively
    Returns (is_figurative, explanation)
    """
    text_lower = text.lower()
    words = set(text_lower.split())
    
    figurative_matches = []
    
    for disaster_word, pattern in FIGURATIVE_PATTERNS.items():
        if disaster_word in text_lower:
            # Check if any context words are present
            context_matches = [ctx for ctx in pattern["context_words"] if ctx in text_lower]
            if context_matches:
                figurative_matches.append({
                    "word": disaster_word,
                    "context": context_matches,
                    "description": pattern["description"]
                })
    
    return figurative_matches

# ================================================================
# SMART CLASSIFICATION FUNCTION
# ================================================================

def classify_tweet(text):
    """
    Classify tweet as NORMAL, FAKE DISASTER, or REAL DISASTER
    with figurative language detection
    """
    start_time = time.time()
    
    # Preprocess
    processed = preprocess_tweet(text)
    text_lower = text.lower()
    
    # Check for figurative language first
    figurative_matches = detect_figurative_language(text)
    
    # Initialize scoring
    normal_score = 0
    fake_score = 0
    real_score = 0
    
    normal_indicators = []
    fake_indicators = []
    real_indicators = []
    detected_disasters = []
    
    # ================================================================
    # NORMAL TWEET DETECTION
    # ================================================================
    
    # Check for normal conversation keywords
    normal_matches = [kw for kw in NORMAL_KEYWORDS if kw in text_lower]
    if normal_matches:
        normal_score += len(normal_matches) * 1.5
        normal_indicators.append({
            "icon": "💬",
            "matches": normal_matches[:5],
            "count": len(normal_matches),
            "points": len(normal_matches) * 1.5,
            "description": "Normal conversation indicators"
        })
    
    # ================================================================
    # DISASTER KEYWORD DETECTION
    # ================================================================
    
    disaster_keyword_count = 0
    for disaster, keywords in DISASTER_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if matches:
            # Only count as disaster if NOT figurative
            is_figurative = any(fm["word"] in str(matches) for fm in figurative_matches)
            
            if not is_figurative:
                detected_disasters.append({
                    "name": disaster,
                    "keywords": matches[:3],
                    "count": len(matches)
                })
                disaster_keyword_count += len(matches)
    
    # ================================================================
    # FAKE NEWS DETECTION
    # ================================================================
    
    # Only apply fake/real scoring if there are actual disaster keywords
    if disaster_keyword_count > 0:
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
    # FINAL CLASSIFICATION
    # ================================================================
    
    # Check if all disaster keywords are figurative
    all_figurative = len(figurative_matches) > 0 and disaster_keyword_count == 0
    
    if all_figurative or disaster_keyword_count == 0:
        classification = "NORMAL"
        confidence = 0.95  # High confidence for normal tweets
        reasons = [f"✅ **NORMAL TWEET** - No actual disaster-related content detected"]
        
        if figurative_matches:
            reasons.append("📝 **Figurative Language Detected:**")
            for fm in figurative_matches:
                reasons.append(f"  • '{fm['word']}' is used figuratively with context: {', '.join(fm['context'][:3])}")
                reasons.append(f"    {fm['description']}")
        
        if normal_indicators:
            reasons.append("📊 **Normal Indicators Found:**")
            for ind in normal_indicators:
                reasons.append(f"  {ind['icon']} {ind['description']} ({ind['count']} matches)")
    else:
        # Disaster tweet - determine if fake or real
        if fake_score > real_score:
            classification = "FAKE"
            total = fake_score + real_score
            confidence = fake_score / total if total > 0 else 0.5
            reasons = [f"🔴 **FAKE DISASTER TWEET**"]
            if fake_indicators:
                reasons.append("📊 **Fake Indicators Found:**")
                for ind in fake_indicators[:3]:
                    reasons.append(f"  {ind['icon']} {ind['description']} ({ind['count']} matches)")
        else:
            classification = "REAL"
            total = fake_score + real_score
            confidence = real_score / total if total > 0 else 0.5
            reasons = [f"✅ **REAL DISASTER TWEET**"]
            if real_indicators:
                reasons.append("📊 **Real Indicators Found:**")
                for ind in real_indicators[:3]:
                    reasons.append(f"  {ind['icon']} {ind['description']} ({ind['count']} matches)")
    
    # Add disaster types if detected
    if detected_disasters:
        reasons.append("🌪️ **Disaster Types Detected:**")
        for d in detected_disasters:
            reasons.append(f"  {d['name']}")
    
    # Calculate verdict based on confidence
    if confidence > 0.8:
        verdict = "HIGH CONFIDENCE"
    elif confidence > 0.6:
        verdict = "MEDIUM CONFIDENCE"
    elif confidence > 0.4:
        verdict = "LOW CONFIDENCE"
    else:
        verdict = "UNCERTAIN"
    
    return {
        "classification": classification,
        "fake_score": fake_score,
        "real_score": real_score,
        "normal_score": normal_score,
        "confidence": confidence,
        "verdict": verdict,
        "figurative_matches": figurative_matches,
        "fake_indicators": fake_indicators,
        "real_indicators": real_indicators,
        "normal_indicators": normal_indicators,
        "detected_disasters": detected_disasters,
        "reasons": reasons,
        "metrics": processed,
        "response_time": time.time() - start_time
    }

# ================================================================
# DISPLAY FUNCTIONS
# ================================================================

def display_classification_banner(result):
    """Display classification banner"""
    
    if result["classification"] == "NORMAL":
        st.markdown(
            f'''
            <div class="alert-box normal-alert">
                💬 NORMAL TWEET - NOT DISASTER RELATED
                <div style="font-size: 0.7em; margin-top: 10px;">
                    Confidence: {result["confidence"]*100:.1f}% | {result["verdict"]}
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        st.markdown(f'<span class="normal-badge">💬 NORMAL</span>', unsafe_allow_html=True)
    elif result["classification"] == "FAKE":
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
        st.markdown(f'<span class="fake-badge">❌ FAKE</span>', unsafe_allow_html=True)
    else:  # REAL
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
        st.markdown(f'<span class="real-badge">✅ REAL</span>', unsafe_allow_html=True)

def display_indicators(result):
    """Display indicators based on classification"""
    
    # Show figurative language if detected
    if result["figurative_matches"]:
        st.markdown("### 🔤 Figurative Language Detected")
        for fm in result["figurative_matches"]:
            st.markdown(f"""
            <div class="figurative-indicator">
                <strong>✨ '{fm['word']}'</strong> is used figuratively with context words: {', '.join(fm['context'][:3])}<br>
                <small>{fm['description']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    if result["classification"] == "NORMAL":
        if result["normal_indicators"]:
            st.markdown("### 💬 Normal Conversation Indicators")
            for ind in result["normal_indicators"]:
                st.markdown(f"""
                <div class="indicator-card" style="border-left-color: #95a5a6;">
                    <div style="display: flex; justify-content: space-between;">
                        <span><strong>{ind['icon']} Normal Content</strong></span>
                        <span style="color: #7f8c8d;">+{ind['points']:.1f} points</span>
                    </div>
                    <div style="color: #666; font-size: 0.9em;">{ind['description']}</div>
                    <div style="color: #888; font-size: 0.8em; margin-top: 5px;">
                        Found: {', '.join(ind['matches'][:3])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
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
            <p style="font-size: 1.3em; opacity: 0.95;">Smart Disaster Tweet Analysis with Figurative Language Detection</p>
            <div style="margin-top: 20px;">
                <span class="status-badge live-badge">🔴 LIVE</span>
                <span class="status-badge model-badge">🧠 Smart Classifier</span>
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
        <h3 style="color: #333; margin-bottom: 15px;">📊 Classification Statistics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # SAFELY access stats with .get() to avoid KeyError
    stats = st.session_state["stats"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        normal_count = stats.get("normal", 0)
        st.markdown(f"""
        <div style="background: white; padding: 15px; border-radius: 15px; text-align: center;">
            <div style="font-size: 2em; font-weight: 800; color: #7f8c8d;">{normal_count}</div>
            <div style="color: #666;">Normal</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fake_count = stats.get("fake", 0)
        st.markdown(f"""
        <div style="background: white; padding: 15px; border-radius: 15px; text-align: center;">
            <div style="font-size: 2em; font-weight: 800; color: #ff6b6b;">{fake_count}</div>
            <div style="color: #666;">Fake</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        real_count = stats.get("real", 0)
        st.markdown(f"""
        <div style="background: white; padding: 15px; border-radius: 15px; text-align: center;">
            <div style="font-size: 2em; font-weight: 800; color: #10ac84;">{real_count}</div>
            <div style="color: #666;">Real</div>
        </div>
        """, unsafe_allow_html=True)
    
    figurative_count = stats.get("figurative", 0)
    st.markdown(f"""
    <div style="background: white; padding: 15px; border-radius: 15px; text-align: center; margin-top: 10px;">
        <div style="font-size: 1.5em; font-weight: 800; color: #f39c12;">{figurative_count}</div>
        <div style="color: #666;">Figurative Language</div>
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
        st.session_state["stats"] = {"total": 0, "normal": 0, "fake": 0, "real": 0, "figurative": 0, "locations": {}, "keywords": {}}
        st.rerun()

# ================================================================
# AUTO-REFRESH LOGIC
# ================================================================
if st.session_state.get("auto_refresh", True):
    if time.time() - st.session_state.get("last_refresh", time.time()) > 5:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

# ================================================================
# INPUT SECTION - With working Clear button
# ================================================================
st.markdown("### 📝 Enter Tweet for Analysis")

# Create a unique key for the text area that changes when clear is clicked
text_area_key = f"tweet_input_{st.session_state['input_key_counter']}"

# Input and Clear button in columns
input_col, clear_col = st.columns([6, 1])

with input_col:
    # Use the dynamic key to force refresh when cleared
    tweet = st.text_area(
        "Tweet input",
        value=st.session_state.get("tweet_input", ""),
        height=120,
        placeholder="Type or paste a tweet here to analyze...",
        key=text_area_key,
        label_visibility="collapsed"
    )

with clear_col:
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing for alignment
    # Use on_click callback to clear the input
    st.button(
        "🗑️ Clear", 
        on_click=clear_input, 
        use_container_width=True,
        key="clear_button",
        help="Clear input field"
    )

# Update session state with current tweet value
if tweet:
    st.session_state["tweet_input"] = tweet

# Add a simple instruction line
st.caption("Enter any tweet above and click 'Analyze' to classify as Normal, Fake Disaster, or Real Disaster.")

# ================================================================
# ANALYZE BUTTON
# ================================================================
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ================================================================
# ANALYSIS EXECUTION
# ================================================================
if analyze_clicked and st.session_state.get("tweet_input"):
    with st.spinner("🎯 Classifying tweet with smart detection..."):
        result = classify_tweet(st.session_state["tweet_input"])
        
        if result:
            # Extract location
            location = None
            for loc in MALAYSIA_LOCATIONS:
                if loc.lower() in st.session_state["tweet_input"].lower():
                    location = loc
                    break
            
            # Update stats
            st.session_state["stats"]["total"] = st.session_state["stats"].get("total", 0) + 1
            
            if result["classification"] == "NORMAL":
                st.session_state["stats"]["normal"] = st.session_state["stats"].get("normal", 0) + 1
            elif result["classification"] == "FAKE":
                st.session_state["stats"]["fake"] = st.session_state["stats"].get("fake", 0) + 1
            else:  # REAL
                st.session_state["stats"]["real"] = st.session_state["stats"].get("real", 0) + 1
            
            if result["figurative_matches"]:
                st.session_state["stats"]["figurative"] = st.session_state["stats"].get("figurative", 0) + 1
            
            if location:
                if "locations" not in st.session_state["stats"]:
                    st.session_state["stats"]["locations"] = {}
                st.session_state["stats"]["locations"][location] = st.session_state["stats"]["locations"].get(location, 0) + 1
            
            # Save analysis
            analysis_record = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "tweet": st.session_state["tweet_input"][:50] + "..." if len(st.session_state["tweet_input"]) > 50 else st.session_state["tweet_input"],
                "classification": result["classification"],
                "confidence": result["confidence"],
                "location": location if location else "Unknown",
                "figurative": len(result["figurative_matches"]) > 0
            }
            st.session_state["analyses"].append(analysis_record)
            
            # Display results
            st.markdown("---")
            
            # Classification Banner
            display_classification_banner(result)
            
            # Score Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Normal Score", f"{result['normal_score']:.1f} points")
            with col2:
                st.metric("Fake Score", f"{result['fake_score']:.1f} points")
            with col3:
                st.metric("Real Score", f"{result['real_score']:.1f} points")
            
            # Indicators
            display_indicators(result)
            
            # Text Metrics
            st.markdown("### 📝 Text Analysis")
            display_metrics(result)
            
            # Disaster Types (only if detected)
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
            
            # Location map (only if location detected)
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
                    elif "NORMAL" in reason:
                        st.info(reason)
                    else:
                        st.write(reason)
            
            # Performance
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea10, #764ba210); 
                        padding: 15px; border-radius: 15px; margin-top: 20px;
                        display: flex; justify-content: space-between;">
                <span>⚡ Response time: {result['response_time']*1000:.0f}ms</span>
                <span>🤖 Model: Smart Classifier v3.0</span>
                <span>🎯 Verdict: {result['verdict']}</span>
            </div>
            """, unsafe_allow_html=True)

elif analyze_clicked and not st.session_state.get("tweet_input"):
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
        classification = a.get("classification", "NORMAL")
        confidence = a.get("confidence", 0.5)
        location = a.get("location", "Unknown")
        tweet_text = a.get("tweet", "Unknown tweet")
        timestamp = a.get("timestamp", datetime.now().strftime("%H:%M:%S"))
        figurative = a.get("figurative", False)
        
        # Set emoji based on classification
        if classification == "NORMAL":
            emoji = "💬"
        elif classification == "FAKE":
            emoji = "🔴"
        else:  # REAL
            emoji = "✅"
        
        # Add figurative marker
        fig_marker = " ✨" if figurative else ""
        
        feed_data.append({
            "Time": timestamp,
            "Tweet": tweet_text,
            "Classification": f"{emoji} {classification}{fig_marker}",
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
            <span style="background: #95a5a6; color: white; padding: 8px 20px; border-radius: 30px;">💬 Normal</span>
            <span style="background: #ff6b6b; color: white; padding: 8px 20px; border-radius: 30px;">🔴 Fake</span>
            <span style="background: #10ac84; color: white; padding: 8px 20px; border-radius: 30px;">✅ Real</span>
            <span style="background: #f39c12; color: white; padding: 8px 20px; border-radius: 30px;">✨ Figurative</span>
            <span style="background: #667eea; color: white; padding: 8px 20px; border-radius: 30px;">🎯 v3.0</span>
        </div>
        <p style="color: #666;">
            Disaster Tweet AI Detector | Smart Classification with Figurative Language Detection<br>
            Session: {st.session_state["session_id"]} | Last sync: {datetime.now().strftime('%H:%M:%S')}
        </p>
        <p style="color: #888; font-size: 0.9em; margin-top: 15px;">
            ✅ Smart classifier • Figurative language detection • 3-way classification
        </p>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
