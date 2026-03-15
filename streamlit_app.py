# ================================================================
# DISASTER TWEET AI DETECTOR - COMPLETE WORKING VERSION
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

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Disaster Tweet AI Detector",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
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
if "local_analyses" not in st.session_state:
    st.session_state["local_analyses"] = []
if "analysis_count" not in st.session_state:
    st.session_state["analysis_count"] = 0

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
# PREPROCESSING FUNCTION
# ================================================================

def preprocess_tweet(text):
    """Clean and preprocess tweet text"""
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
# MOCK API PREDICTION FUNCTION
# ================================================================

def predict_with_mock(text):
    """Mock API prediction - always available"""
    start_time = time.time()
    
    # Simple keyword-based detection
    text_lower = text.lower()
    
    disaster_score = 0
    non_disaster_score = 0
    
    # Disaster indicators
    disaster_keywords = [
        'flood', 'banjir', 'earthquake', 'gempa', 'fire', 'kebakaran', 
        'storm', 'ribut', 'tsunami', 'landslide', 'runtuh', 'urgent', 
        'breaking', 'emergency', 'warning', 'alert', 'disaster', 'bencana',
        'evacuate', 'pindah', 'rescue', 'casualty', 'victim', 'damage',
        'destroyed', 'collapsed', 'injured'
    ]
    
    # Non-disaster indicators
    non_disaster_keywords = [
        'good morning', 'good night', 'hello', 'hi', 'thanks', 'thank you',
        'welcome', 'please', 'happy', 'excited', 'love', 'like', 'great',
        'awesome', 'amazing', 'fun', 'enjoy', 'party', 'birthday', 'weekend',
        'holiday', 'vacation', 'food', 'restaurant', 'movie', 'music',
        'beautiful', 'nice', 'cool', 'wonderful'
    ]
    
    # Count keywords
    for kw in disaster_keywords:
        if kw in text_lower:
            disaster_score += 1
    
    for kw in non_disaster_keywords:
        if kw in text_lower:
            non_disaster_score += 1.5
    
    # Detect disaster types
    detected_disasters = []
    for disaster, keywords in DISASTER_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_disasters.append(disaster)
    
    # Calculate probability
    total = disaster_score + non_disaster_score
    if total > 0:
        disaster_prob = disaster_score / total
    else:
        disaster_prob = 0.5
    
    non_disaster_prob = 1 - disaster_prob
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
        reasons.append("🔴 Detected potential DISASTER content")
        if disaster_score > 0:
            reasons.append(f"📊 Found {disaster_score} disaster-related keywords")
    else:
        reasons.append("✅ Detected NON-DISASTER content")
        if non_disaster_score > 0:
            reasons.append(f"📊 Found {int(non_disaster_score/1.5)} non-disaster keywords")
    
    if detected_disasters:
        reasons.append(f"🌪️ Possible disaster type(s): {', '.join(detected_disasters)}")
    
    if exclamation_count > 3:
        reasons.append(f"❗ High exclamation count ({exclamation_count})")
    
    if caps_words > 2:
        reasons.append(f"📣 Multiple ALL CAPS words ({caps_words})")
    
    if has_url:
        reasons.append("🔗 Contains URL - may provide evidence")
    
    if numbers:
        reasons.append(f"🔢 Contains {len(numbers)} numbers/data points")
    
    return {
        "is_disaster": disaster_prob > 0.5,
        "disaster_probability": disaster_prob,
        "non_disaster_probability": non_disaster_prob,
        "confidence": confidence,
        "reasons": reasons[:5],
        "detected_disasters": detected_disasters,
        "model_used": "Mock AI",
        "response_time": time.time() - start_time,
        "word_count": len(words),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "caps_words": caps_words,
        "has_url": has_url,
        "number_count": len(numbers)
    }

# ================================================================
# DISPLAY FUNCTIONS
# ================================================================

def display_probability_bar(disaster_prob, non_disaster_prob):
    disaster_percent = disaster_prob * 100
    non_disaster_percent = non_disaster_prob * 100
    
    st.markdown(f"""
    <div style="background: #f0f0f0; border-radius: 10px; padding: 5px; margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="color: #ff4444; font-weight: bold;">DISASTER: {disaster_percent:.1f}%</span>
            <span style="color: #00cc66; font-weight: bold;">NON-DISASTER: {non_disaster_percent:.1f}%</span>
        </div>
        <div style="height: 40px; border-radius: 5px; overflow: hidden; display: flex;">
            <div style="width: {disaster_percent}%; background: linear-gradient(90deg, #ff4444, #cc0000); color: white; text-align: center; line-height: 40px; font-weight: bold;">
                {disaster_percent:.1f}%
            </div>
            <div style="width: {non_disaster_percent}%; background: linear-gradient(90deg, #00cc66, #008844); color: white; text-align: center; line-height: 40px; font-weight: bold;">
                {non_disaster_percent:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_metrics(analysis):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Confidence", f"{analysis.get('confidence', 0.5)*100:.1f}%")
    with col2:
        st.metric("Words", analysis.get('word_count', 0))
    with col3:
        st.metric("Exclamation", analysis.get('exclamation_count', 0))
    with col4:
        st.metric("Caps Words", analysis.get('caps_words', 0))
    
    if analysis.get('has_url'):
        st.info("🔗 Contains reference link")
    
    if analysis.get('number_count', 0) > 0:
        st.info(f"🔢 Contains {analysis['number_count']} numbers")
    
    if analysis.get('detected_disasters'):
        st.markdown("#### 🌪️ Detected Disaster Types")
        cols = st.columns(len(analysis['detected_disasters']))
        for i, disaster in enumerate(analysis['detected_disasters']):
            with cols[i]:
                st.info(disaster.upper())
    
    if analysis.get('reasons'):
        st.markdown("#### 🔍 Analysis Reasons")
        for reason in analysis['reasons']:
            st.warning(f"• {reason}")

def create_location_map(location, lat, lon, is_disaster):
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

def display_live_feed():
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
                "Status": "🔴 DISASTER" if a.get("is_disaster") else "✅ NOT"
            })
        df = pd.DataFrame(feed_data)
        st.dataframe(df, use_container_width=True, height=300)

# ================================================================
# AUTO-REFRESH LOGIC
# ================================================================
if st.session_state["auto_refresh"]:
    if time.time() - st.session_state["last_refresh"] > 5:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

# ================================================================
# MAIN UI
# ================================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px; border-radius: 20px; color: white; text-align: center; margin-bottom: 30px;
    }
    .mock-badge {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white; padding: 8px 20px; border-radius: 30px; display: inline-block; margin: 10px 0;
        font-weight: bold;
    }
    .stButton button {
        border-radius: 12px !important;
        font-weight: bold !important;
    }
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #667eea, #5a67d8) !important;
        color: white !important;
    }
    .stat-card {
        background: white; padding: 20px; border-radius: 15px; text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1); margin: 10px 0;
    }
    .stat-value {
        font-size: 2.5em; font-weight: 700; color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    f'''
    <div class="main-header">
        <h1 style="font-size: 3em; margin-bottom: 10px;">🚀 Disaster Tweet AI Detector</h1>
        <p style="font-size: 1.2em; opacity: 0.9;">Quick Deploy Version - Works Immediately</p>
        <div><span class="mock-badge">✅ Mock AI - No Downloads Required</span></div>
        <p style="margin-top: 15px;">Session: {st.session_state["session_id"]}</p>
    </div>
    ''',
    unsafe_allow_html=True
)

# Sidebar stats
with st.sidebar:
    st.markdown("### 📊 Statistics")
    st.metric("Total Analyses", st.session_state["analysis_count"])
    st.metric("Session ID", st.session_state["session_id"][:8])
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.session_state["auto_refresh"] = st.toggle("🔄 Auto-refresh Feed", value=True)
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state["local_analyses"] = []
        st.session_state["analysis_count"] = 0
        st.rerun()

# Input Section
st.markdown("### 📝 Enter Tweet")

input_col1, input_col2 = st.columns([6, 1])
with input_col1:
    input_key = f"tweet_input_{st.session_state['input_key_counter']}"
    tweet = st.text_area(
        "",
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
st.markdown("#### 📋 Try These Examples")
cols = st.columns(4)
examples = [
    ("📰 Real Disaster", "Heavy rain in Kampar causing flash floods. According to JPS, water levels rising. 150 residents evacuated."),
    ("🚨 Fake Disaster", "URGENT! BREAKING: MASSIVE 8.0 earthquake in Kuala Lumpur! Thousands DEAD! SHARE NOW! 😱😱😱"),
    ("🔄 Mixed Signals", "URGENT! Flood in Johor! Water level 2 meters! Official source says evacuating. JPS confirms."),
    ("📍 Location Test", "Landslide reported in Cameron Highlands - authorities responding, 3 people rescued")
]

for i, (label, text) in enumerate(examples):
    with cols[i]:
        if st.button(label, use_container_width=True):
            st.session_state["tweet_input"] = text
            st.session_state["input_key_counter"] += 1
            st.rerun()

# Analyze Button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

# Analysis Execution
if analyze_clicked and tweet:
    with st.spinner("🤖 Analyzing..."):
        result = predict_with_mock(tweet)
        
        if result:
            # Extract location
            location = None
            for loc in MALAYSIA_LOCATIONS:
                if loc.lower() in tweet.lower():
                    location = loc
                    break
            
            # Save to history
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state["local_analyses"].append({
                "timestamp": timestamp,
                "tweet": tweet[:50] + "...",
                "location": location,
                "is_disaster": result["is_disaster"],
                "disaster_probability": result["disaster_probability"]
            })
            st.session_state["analysis_count"] += 1
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            # Alert
            if result["is_disaster"]:
                st.error(f"🔴 **DISASTER TWEET DETECTED**\n\nConfidence: {result['confidence']*100:.1f}%")
            else:
                st.success(f"✅ **NOT A DISASTER TWEET**\n\nConfidence: {result['confidence']*100:.1f}%")
            
            # Probability bar
            display_probability_bar(result["disaster_probability"], result["non_disaster_probability"])
            
            # Metrics
            display_metrics(result)
            
            # Model info
            st.info(f"🤖 Model: {result['model_used']} | Response time: {result['response_time']*1000:.0f}ms")
            
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

# Live Feed
st.markdown("---")
display_live_feed()

# Footer
st.markdown("---")
st.markdown(
    f'''
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea10, #764ba210); border-radius: 15px;">
        <p style="color: #666; font-size: 1.1em;">
            🚀 Disaster Tweet AI Detector | Mock AI Version
        </p>
        <p style="color: #888; font-size: 0.9em;">
            Session: {st.session_state["session_id"]} | Last sync: {datetime.now().strftime('%H:%M:%S')}
        </p>
        <p style="color: #28a745; font-size: 0.9em; font-weight: bold;">
            ✅ DEPLOYS IMMEDIATELY - No model downloads required
        </p>
    </div>
    ''',
    unsafe_allow_html=True
)
