# ================================================================
# ENHANCED DISASTER TWEET AI DETECTOR - PREMIUM UI
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
    page_title="Disaster Tweet AI Detector | Real-time Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌪️"
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
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* Glowing badges */
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
    
    .stats-badge {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
    }
    
    /* Glass cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
        margin: 15px 0;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(102,126,234,0.2);
    }
    
    /* Animated probability bar */
    .probability-container {
        background: rgba(0,0,0,0.05);
        border-radius: 15px;
        padding: 8px;
        margin: 20px 0;
    }
    
    .probability-bar {
        height: 50px;
        border-radius: 12px;
        overflow: hidden;
        display: flex;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .disaster-segment {
        background: linear-gradient(90deg, #ff6b6b, #ee5253);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.1em;
        transition: width 1s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .disaster-segment::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    .non-disaster-segment {
        background: linear-gradient(90deg, #10ac84, #1dd1a1);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.1em;
        transition: width 1s ease-in-out;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 25px 0;
    }
    
    .metric-item {
        background: white;
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .metric-item:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 30px rgba(102,126,234,0.2);
    }
    
    .metric-value {
        font-size: 2.8em;
        font-weight: 800;
        color: #667eea;
        line-height: 1.2;
    }
    
    .metric-label {
        color: #666;
        font-size: 1em;
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
        padding: 15px 30px !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
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
    }
    
    /* Example buttons */
    .example-button {
        background: white !important;
        color: #667eea !important;
        border: 2px solid #667eea !important;
        padding: 10px 20px !important;
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
    
    .disaster-alert {
        background: linear-gradient(135deg, #ff6b6b, #ee5253);
    }
    
    .safe-alert {
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
    
    /* Live feed table */
    .dataframe {
        border-radius: 20px !important;
        overflow: hidden !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #667eea, #5a67d8) !important;
        color: white !important;
        padding: 15px !important;
        font-weight: 600 !important;
    }
    
    .dataframe td {
        padding: 12px !important;
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
        
        .metric-value {
            font-size: 2em;
        }
        
        .alert-box {
            font-size: 1em;
            padding: 15px;
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
        "disasters": 0,
        "safe": 0,
        "locations": {},
        "keywords": {}
    }
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

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
    "🌊 Flood": ['flood', 'banjir', 'inundation', 'water level', 'banjir kilat'],
    "🌋 Earthquake": ['earthquake', 'gempa', 'tremor', 'seismic', 'gempa bumi'],
    "⛈️ Storm": ['storm', 'thunderstorm', 'ribut', 'petir', 'lightning', 'ribut petir'],
    "🏔️ Landslide": ['landslide', 'tanah runtuh', 'mudslide', 'runtuh'],
    "🔥 Fire": ['fire', 'kebakaran', 'burning', 'api'],
    "🌊 Tsunami": ['tsunami', 'tidal wave', 'gelombang tsunami'],
    "💨 Wind": ['wind', 'angin', 'gale', 'tornado', 'angin kencang']
}

URGENCY_INDICATORS = ['urgent', 'breaking', 'alert', 'warning', '!!!', '🚨', 'segera', 'penting']
SENSATIONAL_INDICATORS = ['unbelievable', 'shocking', 'massive', 'worst ever', 'catastrophic', 'devastating']

# ================================================================
# PREPROCESSING FUNCTION
# ================================================================

def preprocess_tweet(text):
    """Clean and preprocess tweet text"""
    if pd.isna(text) or text == "":
        return ""
    
    # Save original for metrics
    original = text
    
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
        "has_url": bool(re.search(r'http[s]?://', original.lower())),
        "numbers": len(re.findall(r'\d+', original))
    }

# ================================================================
# ENHANCED PREDICTION FUNCTION
# ================================================================

def analyze_tweet(text):
    """Enhanced analysis with multiple factors"""
    start_time = time.time()
    
    # Preprocess
    processed = preprocess_tweet(text)
    text_lower = processed["clean"]
    original_lower = text.lower()
    
    # Scoring
    disaster_score = 0
    safe_score = 0
    
    # Disaster keyword scoring
    detected_disasters = []
    keyword_matches = {}
    
    for disaster, keywords in DISASTER_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in original_lower]
        if matches:
            detected_disasters.append(disaster)
            keyword_matches[disaster] = matches
            disaster_score += len(matches) * 2
    
    # Urgency scoring
    urgency_matches = [ind for ind in URGENCY_INDICATORS if ind in original_lower]
    if urgency_matches:
        disaster_score += len(urgency_matches) * 1.5
    
    # Sensational scoring
    sensational_matches = [ind for ind in SENSATIONAL_INDICATORS if ind in original_lower]
    if sensational_matches:
        disaster_score += len(sensational_matches) * 2
    
    # Safe indicators
    safe_indicators = ['good morning', 'good night', 'hello', 'thanks', 'thank you',
                       'welcome', 'please', 'happy', 'love', 'great', 'awesome']
    
    safe_matches = [ind for ind in safe_indicators if ind in original_lower]
    if safe_matches:
        safe_score += len(safe_matches) * 2
    
    # Text metrics adjustment
    if processed["exclamation"] > 3:
        disaster_score += processed["exclamation"] * 0.5
    
    if processed["caps_words"] > 2:
        disaster_score += processed["caps_words"] * 0.5
    
    if processed["has_url"]:
        safe_score += 2
    
    if processed["numbers"] > 2:
        safe_score += processed["numbers"] * 0.3
    
    # Calculate probability
    total = disaster_score + safe_score
    if total > 0:
        disaster_prob = disaster_score / total
    else:
        disaster_prob = 0.5
    
    # Calculate confidence
    confidence = abs(disaster_prob - 0.5) * 2
    
    # Generate reasons
    reasons = []
    if disaster_prob > 0.6:
        if urgency_matches:
            reasons.append(f"🚨 Urgent language detected: {', '.join(urgency_matches[:3])}")
        if sensational_matches:
            reasons.append(f"📢 Sensational language: {', '.join(sensational_matches[:3])}")
        if detected_disasters:
            reasons.append(f"🌪️ Disaster mentions: {', '.join(detected_disasters)}")
        if processed["exclamation"] > 3:
            reasons.append(f"❗ High emotion ({processed['exclamation']} exclamation marks)")
        if processed["caps_words"] > 2:
            reasons.append(f"📣 SHOUTING detected ({processed['caps_words']} words in ALL CAPS)")
    else:
        if safe_matches:
            reasons.append(f"✅ Normal conversation indicators: {', '.join(safe_matches[:3])}")
        if processed["has_url"]:
            reasons.append("🔗 Contains credible link")
        if processed["numbers"] > 2:
            reasons.append(f"📊 Specific data points ({processed['numbers']} numbers)")
        if processed["word_count"] > 20:
            reasons.append("📝 Detailed information provided")
    
    if not reasons:
        reasons.append("ℹ️ No strong indicators - neutral classification")
    
    return {
        "is_disaster": disaster_prob > 0.5,
        "probability": disaster_prob,
        "confidence": confidence,
        "reasons": reasons[:5],
        "disaster_types": detected_disasters,
        "urgency": len(urgency_matches),
        "sensational": len(sensational_matches),
        "metrics": processed,
        "response_time": time.time() - start_time,
        "keyword_matches": keyword_matches
    }

# ================================================================
# DISPLAY FUNCTIONS
# ================================================================

def display_probability_bar(prob):
    """Display animated probability bar"""
    disaster_width = prob * 100
    safe_width = (1 - prob) * 100
    
    st.markdown(f"""
    <div class="probability-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span style="color: #ff6b6b; font-weight: 600;">🔥 DISASTER: {disaster_width:.1f}%</span>
            <span style="color: #10ac84; font-weight: 600;">✅ SAFE: {safe_width:.1f}%</span>
        </div>
        <div class="probability-bar">
            <div class="disaster-segment" style="width: {disaster_width}%;">
                {disaster_width:.1f}%
            </div>
            <div class="non-disaster-segment" style="width: {safe_width}%;">
                {safe_width:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_metrics_card(analysis):
    """Display metrics in beautiful cards"""
    metrics = analysis["metrics"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{metrics['word_count']}</div>
            <div class="metric-label">Words</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{metrics['exclamation']}</div>
            <div class="metric-label">Exclamation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{metrics['caps_words']}</div>
            <div class="metric-label">ALL CAPS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-value">{metrics['numbers']}</div>
            <div class="metric-label">Numbers</div>
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
                    symbol='marker',
                    line=dict(width=2, color='white')
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
            <h1 style="font-size: 3.5em; margin-bottom: 10px;">🌪️ Disaster AI</h1>
            <p style="font-size: 1.3em; opacity: 0.95;">Real-time Disaster Tweet Analysis</p>
            <div style="margin-top: 20px;">
                <span class="status-badge live-badge">🔴 LIVE</span>
                <span class="status-badge model-badge">🧠 Mock AI</span>
                <span class="status-badge stats-badge">📊 v2.0</span>
            </div>
            <p style="margin-top: 20px; font-size: 0.9em; opacity: 0.8;">
                Session: {st.session_state["session_id"]}
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )

# ================================================================
# SIDEBAR - ENHANCED STATS
# ================================================================
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea20, #764ba220); padding: 20px; border-radius: 20px; margin-bottom: 20px;">
        <h3 style="color: #333; margin-bottom: 15px;">📊 Live Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 20px; border-radius: 15px; text-align: center;">
            <div style="font-size: 2.5em; font-weight: 800; color: #667eea;">{st.session_state['stats']['total']}</div>
            <div style="color: #666;">Total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 20px; border-radius: 15px; text-align: center;">
            <div style="font-size: 2.5em; font-weight: 800; color: #ff6b6b;">{st.session_state['stats']['disasters']}</div>
            <div style="color: #666;">Disasters</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Settings
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    auto_refresh = st.toggle("🔄 Auto-refresh", value=st.session_state["auto_refresh"])
    st.session_state["auto_refresh"] = auto_refresh
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state["analyses"] = deque(maxlen=50)
        st.session_state["stats"] = {"total": 0, "disasters": 0, "safe": 0, "locations": {}, "keywords": {}}
        st.rerun()
    
    # Recent stats
    if st.session_state["analyses"]:
        st.markdown("---")
        st.markdown("### 📈 Recent Activity")
        
        recent = list(st.session_state["analyses"])[-5:]
        for r in recent:
            emoji = "🔴" if r["is_disaster"] else "✅"
            st.markdown(f"{emoji} {r['tweet'][:30]}... ({r['probability']*100:.0f}%)")

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
        "",
        height=120,
        placeholder="Example: Heavy rain in Kampar causing flash floods - reported by local authorities...",
        key=input_key,
        label_visibility="collapsed"
    )

with input_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["input_key_counter"] += 1
        st.rerun()

# ================================================================
# QUICK EXAMPLES
# ================================================================
st.markdown("### 🎯 Quick Examples")

cols = st.columns(4)
examples = [
    ("📰 Real News", "Heavy rain in Kampar causing flash floods. According to JPS, water levels rising. 150 residents evacuated."),
    ("🚨 Fake News", "URGENT! BREAKING: MASSIVE 8.0 earthquake in Kuala Lumpur! Thousands DEAD! SHARE NOW! 😱😱😱"),
    ("🔄 Mixed", "URGENT! Flood in Johor! Water level 2 meters! Official source says evacuating. JPS confirms."),
    ("📍 Location", "Landslide reported in Cameron Highlands - authorities responding, 3 people rescued")
]

for i, (label, text) in enumerate(examples):
    with cols[i]:
        if st.button(label, use_container_width=True, key=f"ex_{i}"):
            st.session_state["tweet_input"] = text
            st.session_state["input_key_counter"] += 1
            st.rerun()

# ================================================================
# ANALYZE BUTTON
# ================================================================
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ================================================================
# ANALYSIS EXECUTION
# ================================================================
if analyze_clicked and tweet:
    with st.spinner("🤖 AI analyzing..."):
        result = analyze_tweet(tweet)
        
        if result:
            # Extract location
            location = None
            for loc in MALAYSIA_LOCATIONS:
                if loc.lower() in tweet.lower():
                    location = loc
                    break
            
            # Update stats
            st.session_state["stats"]["total"] += 1
            if result["is_disaster"]:
                st.session_state["stats"]["disasters"] += 1
            else:
                st.session_state["stats"]["safe"] += 1
            
            if location:
                st.session_state["stats"]["locations"][location] = st.session_state["stats"]["locations"].get(location, 0) + 1
            
            # Save analysis
            analysis_record = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "tweet": tweet[:50] + "...",
                "is_disaster": result["is_disaster"],
                "probability": result["probability"],
                "location": location
            }
            st.session_state["analyses"].append(analysis_record)
            
            # Display results
            st.markdown("---")
            
            # Alert
            if result["is_disaster"]:
                st.markdown(
                    f'<div class="alert-box disaster-alert">'
                    f'🔴 DISASTER TWEET DETECTED<br>'
                    f'<span style="font-size: 0.8em;">Confidence: {result["confidence"]*100:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="alert-box safe-alert">'
                    f'✅ SAFE TWEET - NO DISASTER<br>'
                    f'<span style="font-size: 0.8em;">Confidence: {result["confidence"]*100:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Probability bar
            display_probability_bar(result["probability"])
            
            # Metrics
            display_metrics_card(result)
            
            # Reasons
            if result["reasons"]:
                st.markdown("### 🔍 Analysis Reasons")
                for reason in result["reasons"]:
                    if "disaster" in reason.lower() or "urgent" in reason.lower():
                        st.error(f"• {reason}")
                    elif "safe" in reason.lower() or "normal" in reason.lower():
                        st.success(f"• {reason}")
                    else:
                        st.info(f"• {reason}")
            
            # Disaster types
            if result["disaster_types"]:
                st.markdown("### 🌪️ Detected Disaster Types")
                cols = st.columns(len(result["disaster_types"]))
                for i, d_type in enumerate(result["disaster_types"]):
                    with cols[i]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #ff6b6b20, #ee525320); 
                                    padding: 15px; border-radius: 15px; text-align: center;
                                    border: 2px solid #ff6b6b;">
                            <span style="font-size: 2em;">{d_type.split()[0]}</span><br>
                            <strong>{d_type.split()[1]}</strong>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Location map
            if location:
                st.markdown(f"### 🗺️ Location: {location}")
                fig = create_location_map(location)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Performance
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea10, #764ba210); 
                        padding: 15px; border-radius: 15px; margin-top: 20px;
                        display: flex; justify-content: space-between;">
                <span>⚡ Response time: {result['response_time']*1000:.0f}ms</span>
                <span>🤖 Model: Mock AI v2.0</span>
                <span>📊 Confidence: {result['confidence']*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

elif analyze_clicked and not tweet:
    st.warning("⚠️ Please enter a tweet")

# ================================================================
# LIVE FEED
# ================================================================
if st.session_state["analyses"]:
    st.markdown("---")
    st.markdown("### 📡 Live Analysis Feed")
    
    feed_data = []
    for a in reversed(list(st.session_state["analyses"])[-10:]):
        feed_data.append({
            "Time": a["timestamp"],
            "Tweet": a["tweet"],
            "Prediction": "🔴 DISASTER" if a["is_disaster"] else "✅ SAFE",
            "Confidence": f"{a['probability']*100:.1f}%",
            "Location": a["location"] if a["location"] else "Unknown"
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
            <span style="background: #667eea; color: white; padding: 8px 20px; border-radius: 30px;">🚀 v2.0</span>
            <span style="background: #10ac84; color: white; padding: 8px 20px; border-radius: 30px;">⚡ Real-time</span>
            <span style="background: #f39c12; color: white; padding: 8px 20px; border-radius: 30px;">🔬 AI-Powered</span>
        </div>
        <p style="color: #666;">
            Disaster Tweet AI Detector | Advanced Real-time Analysis<br>
            Session: {st.session_state["session_id"]} | Last sync: {datetime.now().strftime('%H:%M:%S')}
        </p>
        <p style="color: #888; font-size: 0.9em; margin-top: 15px;">
            ✅ Mock AI • Real-time Processing • Location Detection • Live Feed
        </p>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
