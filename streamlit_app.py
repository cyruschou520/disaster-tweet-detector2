# ================================================================
# DISASTER TWEET DETECTION SYSTEM - BERT + SOURCE + TIME + FIREBASE (NO API)
# WITH INTEGRATED TRAINING, FEEDBACK, AND THREE‑CLASS SUPPORT
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
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import nltk
from nltk.corpus import stopwords
from collections import deque
import base64
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW                         # <-- CORRECT IMPORT
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib
import sys

# ================================================================
# CONFIGURATION
# ================================================================
NUM_CLASSES = 3  # 0 = NORMAL, 1 = REAL, 2 = FAKE (fallback to 2 for binary)
MODEL_PATH = "bert_disaster_model_fine_tuned"

# ================================================================
# FIREBASE IMPORTS (optional)
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
    page_title="Disaster Tweet Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌪️"
)

# ================================================================
# PREMIUM CSS (unchanged)
# ================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Space Grotesk', sans-serif; margin: 0; padding: 0; box-sizing: border-box; }
    
    .stApp { background: radial-gradient(circle at 10% 20%, rgba(102,126,234,0.3) 0%, rgba(118,75,162,0.3) 90%); }
    
    .glass-container {
        background: rgba(255,255,255,0.85); backdrop-filter: blur(20px);
        border-radius: 40px; padding: 30px; margin: 20px;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.3);
        animation: fadeIn 1s ease-out;
    }
    
    @keyframes fadeIn { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
    
    .premium-header {
        background: linear-gradient(-45deg, #667eea, #764ba2, #48bb78, #fbbf24);
        background-size: 400% 400%;
        padding: 50px; border-radius: 30px; color: white; text-align: center; margin-bottom: 30px;
        box-shadow: 0 30px 60px -15px rgba(102,126,234,0.5);
        animation: gradientBG 8s ease infinite, float 6s ease-in-out infinite;
        position: relative; overflow: hidden;
    }
    
    @keyframes gradientBG { 0% { background-position:0% 50%; } 50% { background-position:100% 50%; } 100% { background-position:0% 50%; } }
    @keyframes float { 0% { transform:translateY(0px); } 50% { transform:translateY(-10px); } 100% { transform:translateY(0px); } }
    
    .premium-header::before {
        content: ''; position: absolute; top:-50%; left:-50%; width:200%; height:200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
    
    .badge {
        display: inline-block; padding: 12px 28px; border-radius: 50px; font-weight: 600; margin: 10px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1); transition: all 0.3s ease;
        backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.3);
    }
    .badge:hover { transform: translateY(-3px) scale(1.05); box-shadow: 0 20px 30px rgba(0,0,0,0.2); }
    
    .live-badge { background: linear-gradient(135deg, #ff6b6b, #ee5253); color: white; }
    .bert-badge { background: linear-gradient(135deg, #9b59b6, #8e44ad); color: white; }
    .time-badge { background: linear-gradient(135deg, #3498db, #2980b9); color: white; }
    .news-badge { background: linear-gradient(135deg, #95a5a6, #7f8c8d); color: white; }
    .firebase-badge { background: linear-gradient(135deg, #ffca28, #ffb300); color: #333; }
    
    .result-card {
        background: white; border-radius: 25px; padding: 30px; margin: 20px 0;
        box-shadow: 0 20px 40px -12px rgba(0,0,0,0.2); border-left: 8px solid;
        transition: all 0.3s ease; animation: slideIn 0.5s ease-out;
    }
    .result-card:hover { transform: translateX(10px); box-shadow: 0 30px 50px -12px rgba(102,126,234,0.3); }
    .real-card { border-left-color: #10ac84; background: linear-gradient(135deg, #f0fff4, white); }
    .fake-card { border-left-color: #ff6b6b; background: linear-gradient(135deg, #fff5f5, white); }
    .uncertain-card { border-left-color: #f39c12; background: linear-gradient(135deg, #fff9e6, white); }
    .normal-card { border-left-color: #95a5a6; background: linear-gradient(135deg, #f8f9fa, white); }
    
    @keyframes slideIn { from { transform:translateX(-20px); opacity:0; } to { transform:translateX(0); opacity:1; } }
    
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap:20px; margin:20px 0; }
    
    .metric-card {
        background: white; padding: 25px; border-radius: 20px; text-align: center;
        box-shadow: 0 15px 30px -10px rgba(0,0,0,0.1); transition: all 0.3s ease;
        border: 1px solid rgba(102,126,234,0.1); position: relative; overflow: hidden;
    }
    .metric-card::before {
        content: ''; position: absolute; top:-50%; left:-50%; width:200%; height:200%;
        background: radial-gradient(circle, rgba(102,126,234,0.1) 0%, transparent 70%);
        transition: all 0.5s ease;
    }
    .metric-card:hover { transform: translateY(-5px) scale(1.02); box-shadow: 0 25px 40px -10px rgba(102,126,234,0.3); }
    .metric-card:hover::before { transform: rotate(45deg); }
    
    .metric-value { font-size: 3em; font-weight: 700; color: #667eea; line-height: 1.2; position: relative; z-index:1; }
    .metric-label { color: #666; font-size: 1em; text-transform: uppercase; letter-spacing: 1px; font-weight: 500; position: relative; z-index:1; }
    
    .source-card {
        background: linear-gradient(135deg, #f8f9fa, white); padding: 20px; border-radius: 15px; margin: 10px 0;
        border-left: 5px solid #10ac84; transition: all 0.3s ease; box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    .source-card:hover { transform: translateX(5px); box-shadow: 0 10px 25px rgba(16,172,132,0.2); }
    
    .progress-container {
        width: 100%; height: 30px; background: #f0f0f0; border-radius: 15px; overflow: hidden;
        margin: 15px 0; box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
    }
    .progress-fill {
        height: 100%; background: linear-gradient(90deg, #667eea, #5a67d8); border-radius: 15px;
        transition: width 1s cubic-bezier(0.4,0,0.2,1); display: flex; align-items: center;
        justify-content: center; color: white; font-weight: 600; font-size: 0.9em;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    
    .stTextArea textarea {
        border-radius: 20px !important; border: 2px solid #e0e0e0 !important; padding: 20px !important;
        font-size: 16px !important; transition: all 0.3s ease !important; background: white !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05) !important;
    }
    .stTextArea textarea:focus {
        border-color: #667eea !important; box-shadow: 0 5px 25px rgba(102,126,234,0.3) !important;
        transform: scale(1.02);
    }
    
    .stButton button {
        border-radius: 15px !important; padding: 15px 35px !important; font-weight: 600 !important;
        font-size: 1.1em !important; transition: all 0.3s ease !important; border: none !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important;
        background: linear-gradient(135deg, #667eea, #5a67d8) !important; color: white !important;
    }
    .stButton button:hover { transform: translateY(-3px) scale(1.02); box-shadow: 0 20px 30px rgba(102,126,234,0.4) !important; }
    
    /* Clear button style (gray) */
    .clear-button button {
        background: linear-gradient(135deg, #95a5a6, #7f8c8d) !important;
    }
    
    .sidebar-content {
        background: rgba(255,255,255,0.95); backdrop-filter: blur(10px); border-radius: 25px;
        padding: 25px; margin: 20px; box-shadow: 0 15px 35px -10px rgba(0,0,0,0.2);
    }
    
    .stat-card {
        background: white; border-radius: 20px; padding: 20px; text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05); transition: all 0.3s ease;
        border: 1px solid rgba(102,126,234,0.1);
    }
    .stat-card:hover { transform: translateY(-5px); box-shadow: 0 20px 30px rgba(102,126,234,0.2); }
    .stat-value { font-size: 2.5em; font-weight: 700; color: #667eea; line-height:1.2; }
    .stat-label { color: #666; font-size: 0.9em; text-transform: uppercase; letter-spacing:1px; }
    
    .dataframe {
        border-radius: 20px !important; overflow: hidden !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important; border: none !important;
    }
    .dataframe th {
        background: linear-gradient(135deg, #667eea, #5a67d8) !important; color: white !important;
        font-weight: 600 !important; padding: 15px !important; font-size: 1em !important;
    }
    .dataframe td { padding: 12px !important; font-size: 0.95em !important; }
    
    .footer {
        text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea10, #764ba210);
        border-radius: 25px; margin-top: 30px; backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    @media (max-width: 768px) {
        .premium-header { padding:30px; }
        .premium-header h1 { font-size:2em !important; }
        .metric-value { font-size:2em; }
        .badge { padding:8px 16px; font-size:0.9em; }
    }
    
    .loading-pulse { animation: pulse 1.5s ease-in-out infinite; }
    @keyframes pulse { 0% { opacity:1; } 50% { opacity:0.5; } 100% { opacity:1; } }
    
    .tooltip { position:relative; display:inline-block; cursor:help; }
    .tooltip .tooltiptext {
        visibility: hidden; width: 200px; background-color: #333; color: #fff; text-align: center;
        border-radius: 10px; padding: 10px; position: absolute; z-index: 1; bottom:125%; left:50%;
        margin-left: -100px; opacity:0; transition: opacity 0.3s; font-size:0.9em;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .tooltip:hover .tooltiptext { visibility: visible; opacity:1; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# FIREBASE INITIALIZATION (safe secrets check)
# ================================================================
def initialize_firebase():
    if not FIREBASE_AVAILABLE:
        return None, False
    try:
        if not firebase_admin._apps:
            try:
                has_firebase_secrets = "firebase" in st.secrets
            except:
                has_firebase_secrets = False

            if has_firebase_secrets:
                firebase_config = {
                    "type": st.secrets["firebase"]["type"],
                    "project_id": st.secrets["firebase"]["project_id"],
                    "private_key_id": st.secrets["firebase"]["private_key_id"],
                    "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
                    "client_email": st.secrets["firebase"]["client_email"],
                    "client_id": st.secrets["firebase"]["client_id"],
                    "auth_uri": st.secrets["firebase"]["auth_uri"],
                    "token_uri": st.secrets["firebase"]["token_uri"],
                    "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
                    "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
                }
                cred = credentials.Certificate(firebase_config)
                firebase_admin.initialize_app(cred)
                db = firestore.client()
                return db, True
            else:
                st.sidebar.warning("⚠️ Firebase credentials not found in secrets")
                return None, False
        else:
            return firestore.client(), True
    except Exception as e:
        st.sidebar.error(f"❌ Firebase connection error: {e}")
        return None, False

db, FIREBASE_ACTIVE = initialize_firebase()

# ================================================================
# FIREBASE DATA MANAGER (unchanged)
# ================================================================
class FirebaseDataManager:
    def __init__(self, db, active):
        self.db = db
        self.active = active
        self.collection_name = "tweet_analyses"
        self.stats_collection = "global_stats"

    def save_analysis(self, analysis_data):
        if not self.active or self.db is None:
            return None
        try:
            analysis_data["timestamp"] = firestore.SERVER_TIMESTAMP
            analysis_data["session_id"] = st.session_state["session_id"]
            doc_ref = self.db.collection(self.collection_name).document()
            doc_ref.set(analysis_data)
            self.update_global_stats(analysis_data)
            return doc_ref.id
        except Exception as e:
            st.error(f"❌ Failed to save to Firebase: {e}")
            return None

    def update_global_stats(self, analysis_data):
        try:
            stats_ref = self.db.collection(self.stats_collection).document("live_stats")
            @firestore.transactional
            def update_in_transaction(transaction, stats_ref):
                snapshot = stats_ref.get(transaction=transaction)
                if snapshot.exists:
                    stats = snapshot.to_dict()
                else:
                    stats = {
                        "total": 0,
                        "normal": 0,
                        "real": 0,
                        "fake": 0,
                        "uncertain": 0,
                        "locations": {},
                        "keywords": {},
                        "last_updated": firestore.SERVER_TIMESTAMP
                    }
                stats["total"] = stats.get("total", 0) + 1
                classification = analysis_data.get("classification", "NORMAL").lower()
                stats[classification] = stats.get(classification, 0) + 1
                location = analysis_data.get("location")
                if location and location != "Unknown":
                    if "locations" not in stats:
                        stats["locations"] = {}
                    stats["locations"][location] = stats["locations"].get(location, 0) + 1
                transaction.set(stats_ref, stats)
                return stats
            transaction = self.db.transaction()
            update_in_transaction(transaction, stats_ref)
        except Exception as e:
            print(f"Error updating global stats: {e}")

    def get_global_analyses(self, limit=50):
        if not self.active or self.db is None:
            return []
        try:
            analyses = self.db.collection(self.collection_name)\
                .order_by("timestamp", direction=firestore.Query.DESCENDING)\
                .limit(limit)\
                .stream()
            return [doc.to_dict() for doc in analyses]
        except Exception as e:
            st.warning(f"Could not fetch global analyses: {e}")
            return []

    def get_global_stats(self):
        if not self.active or self.db is None:
            return {}
        try:
            stats_ref = self.db.collection(self.stats_collection).document("live_stats")
            stats = stats_ref.get()
            return stats.to_dict() if stats.exists else {}
        except Exception as e:
            return {}

firebase_manager = FirebaseDataManager(db, FIREBASE_ACTIVE)

# ================================================================
# DOWNLOAD NLTK DATA
# ================================================================
@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))

stop_words = download_nltk_data()

# ================================================================
# DATASET CLASS FOR TRAINING (supports 2 or 3 classes)
# ================================================================
class DisasterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, num_classes=NUM_CLASSES):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        # Ensure label is in range
        if self.num_classes == 2 and label >= 2:
            # If binary but we have a 3-class label, map REAL/FAKE both to 1 (disaster)
            label = 1 if label in [1,2] else 0
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ================================================================
# PREPROCESSING FUNCTION
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
# TRAINING FUNCTION (can accept combined datasets)
# ================================================================
def train_model(df, epochs=3, batch_size=16, lr=2e-5, num_classes=NUM_CLASSES):
    # Preprocess
    with st.spinner("Preprocessing tweets..."):
        df["clean_text"] = df["text"].apply(preprocess_tweet)
    
    # Ensure labels are integers
    df["target"] = df["target"].astype(int)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['clean_text'].values,
        df['target'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['target'].values
    )
    
    # Load tokenizer and model with correct number of labels
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_classes
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create datasets
    train_dataset = DisasterDataset(train_texts, train_labels, tokenizer, num_classes=num_classes)
    val_dataset = DisasterDataset(val_texts, val_labels, tokenizer, num_classes=num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    save_path = MODEL_PATH
    
    # Progress bar placeholders
    epoch_progress = st.progress(0, text="Starting training...")
    status_text = st.empty()
    log_area = st.empty()
    logs = []
    
    for epoch in range(epochs):
        logs.append(f"\n📚 Epoch {epoch + 1}/{epochs}")
        log_area.text("\n".join(logs))
        
        # Training
        model.train()
        total_train_loss = 0
        train_steps = len(train_loader)
        
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress
            progress = (epoch * train_steps + i + 1) / (epochs * train_steps)
            epoch_progress.progress(progress, text=f"Epoch {epoch+1} - Batch {i+1}/{train_steps}")
        
        avg_train_loss = total_train_loss / train_steps
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct / total
        
        logs.append(f"  Training Loss: {avg_train_loss:.4f}")
        logs.append(f"  Validation Loss: {avg_val_loss:.4f}")
        logs.append(f"  Validation Accuracy: {accuracy*100:.2f}%")
        log_area.text("\n".join(logs))
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logs.append(f"  🏆 New best model! Saving...")
            log_area.text("\n".join(logs))
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
    
    logs.append("\n✅ Training complete!")
    log_area.text("\n".join(logs))
    epoch_progress.empty()
    status_text.success("Model trained and saved successfully!")
    return save_path

# ================================================================
# FEEDBACK HANDLING
# ================================================================
FEEDBACK_FILE = "feedback_data.csv"

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    else:
        return pd.DataFrame(columns=["text", "target"])

def save_feedback(tweet, correct_label):
    df = load_feedback()
    # Map labels to integers: NORMAL=0, REAL=1, FAKE=2
    label_map = {"NORMAL": 0, "REAL": 1, "FAKE": 2}
    target = label_map.get(correct_label, 0)
    new_row = pd.DataFrame({"text": [tweet], "target": [target]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(FEEDBACK_FILE, index=False)
    st.success("Feedback saved! It will be used in future training.")

# ================================================================
# LOAD BERT MODEL (supports 2 or 3 classes)
# ================================================================
@st.cache_resource(show_spinner="🔄 Initializing BERT engine...")
def load_bert_model():
    local_model_path = MODEL_PATH
    if os.path.exists(local_model_path):
        try:
            with st.spinner("🧠 Loading your fine-tuned BERT model..."):
                tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
                model.eval()
            # Determine number of labels from model config
            num_labels = model.config.num_labels
            st.sidebar.success(f"✅ Using your fine-tuned BERT model ({num_labels}-class)")
            return model, tokenizer, True, "local", num_labels
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load your local model: {e}. Falling back to default.")
    try:
        with st.spinner("🧠 Loading default BERT model (may take a minute)..."):
            model_name = "Jinyan/bert-base-uncased-fake-news-detection"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
        st.sidebar.info("ℹ️ Using HuggingFace fake news model (fallback) - binary")
        return model, tokenizer, True, "huggingface_fake", 2
    except:
        try:
            with st.spinner("🧠 Loading sentiment model (final fallback)..."):
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model.eval()
            st.sidebar.info("ℹ️ Using HuggingFace sentiment model (final fallback) - binary")
            return model, tokenizer, True, "huggingface_sentiment", 2
        except Exception as e:
            st.sidebar.error(f"❌ BERT unavailable: {e}")
            return None, None, False, None, 2

bert_model, bert_tokenizer, bert_loaded, bert_source, bert_num_labels = load_bert_model()

# ================================================================
# SESSION STATE
# ================================================================
if "session_id" not in st.session_state:
    st.session_state["session_id"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "input_key_counter" not in st.session_state:
    st.session_state["input_key_counter"] = 0
if "analyses" not in st.session_state:
    st.session_state["analyses"] = deque(maxlen=20)
if "global_analyses" not in st.session_state:
    st.session_state["global_analyses"] = []
if "stats" not in st.session_state:
    st.session_state["stats"] = {
        "total": 0,
        "normal": 0,
        "real": 0,
        "fake": 0,
        "uncertain": 0
    }
if "tweet_input" not in st.session_state:
    st.session_state["tweet_input"] = ""
if "last_analysis" not in st.session_state:
    st.session_state["last_analysis"] = None
if "show_correction" not in st.session_state:
    st.session_state["show_correction"] = False

# ================================================================
# CLEAR INPUT CALLBACK
# ================================================================
def clear_input():
    st.session_state["tweet_input"] = ""
    st.session_state["input_key_counter"] += 1

# ================================================================
# SOURCE VERIFICATION (unchanged)
# ================================================================
OFFICIAL_AGENCIES = {
    'jps': 'Jabatan Pengairan dan Saliran (JPS)',
    'met malaysia': 'Malaysian Meteorological Department (MET Malaysia)',
    'jabatan bomba': 'Fire and Rescue Department (BOMBA)',
    'bomba': 'Fire and Rescue Department (BOMBA)',
    'polis': 'Royal Malaysia Police (PDRM)',
    'nadma': 'National Disaster Management Agency (NADMA)',
    'kkmm': 'Ministry of Communications and Multimedia (KKMM)',
    'mkn': 'National Security Council (MKN)',
    'apm': 'Civil Defence Force (APM)',
    'jkm': 'Social Welfare Department (JKM)',
    'jabatan meteorologi': 'Malaysian Meteorological Department'
}

def verify_sources(text):
    text_lower = text.lower()
    sources_found = []
    score = 0

    for kw, name in OFFICIAL_AGENCIES.items():
        if kw in text_lower:
            sources_found.append(name)
            score += 2

    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', text)
    if urls:
        for url in urls[:2]:
            if '.gov' in url or '.my' in url:
                score += 3
                sources_found.append(f"Official domain: {url[:50]}...")
            else:
                score += 1
                sources_found.append(f"Link: {url[:50]}...")

    official_phrases = [
        'according to', 'reported by', 'official statement', 'press release',
        'confirmed by', 'announced by', 'said in a statement'
    ]
    for phrase in official_phrases:
        if phrase in text_lower:
            score += 1
            sources_found.append(f"Phrase: '{phrase}'")

    return {
        "sources": list(set(sources_found)),
        "score": score,
        "has_official": score >= 2,
        "has_url": len(urls) > 0
    }

# ================================================================
# TEMPORAL ANALYSIS (unchanged)
# ================================================================
def parse_dates_from_text(text):
    text_lower = text.lower()
    now = datetime.now()
    detected_dates = []
    recency_score = 0.5  # default
    temporal_phrases = []

    iso_dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
    for date_str in iso_dates:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            detected_dates.append(date_obj)
            temporal_phrases.append(date_str)
        except:
            pass

    time_indicators = {
        r'\bnow\b': (1.0, "now"),
        r'\bjust now\b': (1.0, "just now"),
        r'\bmoments ago\b': (1.0, "moments ago"),
        r'\btoday\b': (0.9, "today"),
        r'\bthis morning\b': (0.9, "this morning"),
        r'\bthis afternoon\b': (0.9, "this afternoon"),
        r'\bthis evening\b': (0.9, "this evening"),
        r'\btonight\b': (0.9, "tonight"),
        r'\byesterday\b': (0.5, "yesterday"),
        r'\blast night\b': (0.5, "last night"),
        r'\blast week\b': (0.2, "last week"),
        r'\blast month\b': (0.1, "last month"),
        r'\blast year\b': (0.0, "last year")
    }
    for pattern, (score, phrase) in time_indicators.items():
        if re.search(pattern, text_lower):
            recency_score = max(recency_score, score)
            temporal_phrases.append(phrase)

    if detected_dates:
        latest_date = max(detected_dates)
        days_ago = (now - latest_date).days
        if days_ago <= 1:
            recency_score = max(recency_score, 1.0)
        elif days_ago <= 7:
            recency_score = max(recency_score, 0.8)
        elif days_ago <= 30:
            recency_score = max(recency_score, 0.5)
        else:
            recency_score = max(recency_score, 0.2)

    if not temporal_phrases:
        temporal_phrases.append("no timestamp")

    return {
        "recency_score": recency_score,
        "phrases": temporal_phrases[:3],
        "is_recent": recency_score > 0.7
    }

# ================================================================
# HISTORICAL EVENT DETECTION
# ================================================================
def is_historical_tweet(text):
    current_year = datetime.now().year
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    if years:
        for y in years:
            if int(y) <= current_year - 2:
                return True
    return False

# ================================================================
# NUMBER DETECTION
# ================================================================
def has_numerical_data(text):
    return bool(re.search(r'\d', text))

# ================================================================
# FIGURATIVE LANGUAGE DETECTION
# ================================================================
FIGURATIVE_PATTERNS = {
    "tsunami": {
        "context_words": ['exam', 'result', 'grade', 'score', 'test', 'homework', 'assignment', 'work', 'email'],
        "description": "Using 'tsunami' figuratively to describe overwhelming amount"
    },
    "flood": {
        "context_words": ['email', 'work', 'task', 'assignment', 'homework', 'message', 'notification', 'application'],
        "description": "Using 'flood' figuratively to describe overwhelming quantity"
    },
    "earthquake": {
        "context_words": ['news', 'announcement', 'result', 'change', 'surprise', 'shock', 'revelation'],
        "description": "Using 'earthquake' figuratively to describe shocking news"
    },
    "storm": {
        "context_words": ['argument', 'fight', 'debate', 'controversy', 'drama', 'criticism'],
        "description": "Using 'storm' figuratively to describe conflict"
    },
    "fire": {
        "context_words": ['energy', 'motivation', 'passion', 'excitement', 'determination', 'drive'],
        "description": "Using 'fire' figuratively to describe enthusiasm"
    },
    "destroyed": {
        "context_words": ['exam', 'result', 'game', 'match', 'competition', 'feeling', 'emotion', 'confidence'],
        "description": "Using 'destroyed' figuratively to describe failure or strong emotion"
    },
    "devastated": {
        "context_words": ['news', 'result', 'outcome', 'feeling', 'emotion', 'relationship', 'heart'],
        "description": "Using 'devastated' figuratively to describe emotional state"
    }
}

def detect_figurative_language(text):
    text_lower = text.lower()
    figurative_matches = []
    for disaster_word, pattern in FIGURATIVE_PATTERNS.items():
        if disaster_word in text_lower:
            context_matches = [ctx for ctx in pattern["context_words"] if ctx in text_lower]
            if context_matches:
                figurative_matches.append({
                    "word": disaster_word,
                    "context": context_matches[:5],
                    "description": pattern["description"]
                })
    return figurative_matches

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
# NORMAL KEYWORDS
# ================================================================
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
# FAKE/REAL KEYWORDS
# ================================================================
FAKE_KEYWORDS = ['urgent', 'breaking', 'alert', 'warning', '!!!', '🚨', 'share', 'viral', 'unbelievable', 'shocking', 'massive', 'worst ever', 'catastrophic', 'devastating', 'horrific', 'never seen', 'unprecedented', 'conspiracy', 'government hiding', 'secret', 'cover up']
REAL_KEYWORDS = ['according to', 'reported by', 'official', 'authorities', 'confirmed', 'verified', 'jps', 'met malaysia', 'bomba', 'polis', 'nadma', 'statement from', 'press release', 'minister', 'department', 'meter', 'km', 'mm', 'magnitude', 'level', 'depth', 'number of', 'total', 'estimated']

def score_indicators(text):
    text_lower = text.lower()
    fake_score = sum(1 for kw in FAKE_KEYWORDS if kw in text_lower)
    real_score = sum(2 for kw in REAL_KEYWORDS if kw in text_lower)
    return fake_score, real_score

# ================================================================
# BERT PREDICTION (supports 2 or 3 classes)
# ================================================================
def predict_with_bert(text):
    if not bert_loaded or bert_model is None:
        return None
    try:
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            if bert_num_labels == 2:
                return probs[0][1].item()
            else:
                return probs[0].tolist()
    except Exception as e:
        st.warning(f"BERT inference error: {e}")
        return None

# ================================================================
# LOCATION EXTRACTION
# ================================================================
MALAYSIA_LOCATIONS = [
    'Kampar', 'Ipoh', 'Kuala Lumpur', 'KL', 'Penang', 'Pulau Pinang',
    'Johor', 'Johor Bahru', 'Shah Alam', 'Selangor', 'Perak', 'Kedah',
    'Kelantan', 'Terengganu', 'Pahang', 'Negeri Sembilan', 'Melaka',
    'Sabah', 'Sarawak', 'Langkawi', 'Kuantan', 'Kota Bharu', 'Alor Setar',
    'George Town', 'Butterworth', 'Taiping', 'Petaling Jaya', 'Subang Jaya',
    'Klang', 'Putrajaya', 'Cameron Highlands', 'Kota Kinabalu', 'Kuching'
]

def extract_location(text):
    text_lower = text.lower()
    for loc in MALAYSIA_LOCATIONS:
        if loc.lower() in text_lower:
            return loc
    return None

# ================================================================
# GEOGRAPHIC FEASIBILITY CHECK
# ================================================================
COASTAL_LOCATIONS = {
    'Penang', 'Pulau Pinang', 'George Town', 'Butterworth',
    'Kuala Lumpur', 'KL', 'Selangor', 'Klang', 'Port Klang',
    'Johor', 'Johor Bahru', 'Melaka', 'Negeri Sembilan',
    'Pahang', 'Kuantan', 'Terengganu', 'Kuala Terengganu',
    'Kelantan', 'Kota Bharu', 'Sabah', 'Kota Kinabalu',
    'Sarawak', 'Kuching', 'Miri', 'Labuan', 'Langkawi'
}

def check_geographic_feasibility(disaster_type, location):
    if not location or location == "Unknown":
        return 1.0
    disaster_lower = disaster_type.lower()
    if 'tsunami' in disaster_lower:
        is_coastal = any(loc.lower() in location.lower() for loc in COASTAL_LOCATIONS)
        if not is_coastal:
            return 2.0
    return 1.0

# ================================================================
# COMBINED CLASSIFICATION (supports 3‑class with safe type handling)
# ================================================================
def classify_tweet(text, api_key=None):
    start_time = time.time()
    figurative_matches = detect_figurative_language(text)
    text_lower = text.lower()
    
    historical = is_historical_tweet(text)
    has_numbers = has_numerical_data(text)
    
    # First, check if it's a normal tweet (no disaster keywords)
    disaster_detected = False
    for disaster, keywords in DISASTER_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                if not any(kw in fm["word"] for fm in figurative_matches):
                    disaster_detected = True
                    break
        if disaster_detected:
            break

    if not disaster_detected:
        return {
            "classification": "NORMAL",
            "combined_fake_prob": 0.0,
            "bert_fake_prob": None,
            "fake_score": 0,
            "real_score": 0,
            "source_info": {"sources": [], "score": 0, "has_official": False, "has_url": False},
            "temporal_info": parse_dates_from_text(text),
            "reasons": ["💬 Normal tweet - No disaster content detected."] + 
                      ([f"✨ Figurative language: {fm['word']} ({fm['description']})" for fm in figurative_matches] if figurative_matches else []),
            "response_time": time.time() - start_time
        }

    # Extract disaster type for geographic checks
    disaster_type = None
    for cat, keywords in DISASTER_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                disaster_type = cat.split()[1]
                break
        if disaster_type:
            break
    if not disaster_type:
        disaster_type = "disaster"

    # Get BERT prediction
    bert_result = predict_with_bert(text)
    location = extract_location(text)
    source_info = verify_sources(text)
    temporal_info = parse_dates_from_text(text)
    fake_score, real_score = score_indicators(text)
    
    rule_fake_prob = (fake_score + 1) / (fake_score + real_score + 2)

    # Geographic feasibility
    geo_multiplier = check_geographic_feasibility(disaster_type, location)

    # Numerical data boost
    num_boost_note = "🔢 Numerical data detected (reduces fake probability)" if has_numbers else None

    # --- Three-class model handling ---
    if bert_num_labels == 3 and isinstance(bert_result, list) and len(bert_result) == 3:
        probs = bert_result
        pred_class = np.argmax(probs)
        class_map = {0: "NORMAL", 1: "REAL", 2: "FAKE"}
        classification = class_map[pred_class]
        confidence = probs[pred_class] * 100

        reasons = [
            f"🤖 BERT confidence: {confidence:.1f}% ({classification})",
            f"🔍 Source credibility: {'HIGH' if source_info['has_official'] else 'MEDIUM' if source_info['has_url'] else 'LOW'}",
            f"⏰ Temporal: {', '.join(temporal_info['phrases'])} (freshness: {temporal_info['recency_score']*100:.0f}%) → {'RECENT' if temporal_info['is_recent'] else 'OLD'}",
        ]
        if geo_multiplier > 1.1:
            reasons.append(f"🗺️ Geographic: Event unlikely in {location} (×{geo_multiplier:.1f} fake boost)")
        if source_info["sources"]:
            reasons.append("📌 Sources found: " + ", ".join(source_info["sources"][:3]))
        if figurative_matches:
            reasons.append("✨ Figurative language detected: " + ", ".join([fm['word'] for fm in figurative_matches]))

        return {
            "classification": classification,
            "combined_fake_prob": 1 - confidence/100 if classification != "NORMAL" else 0,
            "bert_fake_prob": probs[2] if classification == "FAKE" else probs[1] if classification == "REAL" else None,
            "fake_score": fake_score,
            "real_score": real_score,
            "source_info": source_info,
            "temporal_info": temporal_info,
            "reasons": reasons,
            "response_time": time.time() - start_time
        }

    # --- Binary model handling (or fallback) ---
    if isinstance(bert_result, list):
        # If we got a list but we're in binary fallback, take probability of class 2 (FAKE) as fake prob
        if len(bert_result) >= 3:
            bert_fake_prob = bert_result[2]
        else:
            bert_fake_prob = 0.5
    else:
        bert_fake_prob = bert_result if bert_result is not None else 0.5

    # Dynamic weighting
    if historical or has_numbers:
        bert_weight = 0.2
        rule_weight = 0.5
    else:
        bert_weight = 0.4
        rule_weight = 0.3
    combined_fake_prob = bert_weight * bert_fake_prob + rule_weight * rule_fake_prob

    combined_fake_prob *= geo_multiplier

    if has_numbers:
        combined_fake_prob *= 0.8

    if source_info["has_official"]:
        combined_fake_prob *= 0.7
        source_cred = "HIGH (official sources)"
    elif source_info["has_url"]:
        combined_fake_prob *= 0.9
        source_cred = "MEDIUM (contains link)"
    else:
        source_cred = "LOW (no verification)"

    if not temporal_info["is_recent"]:
        trust_score = bert_fake_prob if bert_fake_prob is not None else combined_fake_prob
        if historical and trust_score < 0.4:
            time_cred = "HISTORICAL (no penalty)"
        else:
            combined_fake_prob *= 1.1
            time_cred = "OLD (may be outdated)"
    else:
        time_cred = "RECENT"

    combined_fake_prob = max(0.0, min(1.0, combined_fake_prob))

    if combined_fake_prob < 0.33:
        classification = "REAL"
    elif combined_fake_prob > 0.66:
        classification = "FAKE"
    else:
        classification = "UNCERTAIN"

    reasons = []
    if bert_fake_prob is not None:
        reasons.append(f"🤖 BERT confidence: {bert_fake_prob*100:.1f}% fake (weighted {bert_weight})")
    reasons.append(f"🔍 Source credibility: {source_cred}")
    reasons.append(f"⏰ Temporal: {', '.join(temporal_info['phrases'])} (freshness: {temporal_info['recency_score']*100:.0f}%) → {time_cred}")
    if geo_multiplier > 1.1:
        reasons.append(f"🗺️ Geographic: Event unlikely in {location} (×{geo_multiplier:.1f} fake boost)")
    if num_boost_note:
        reasons.append(num_boost_note)
    if source_info["sources"]:
        reasons.append("📌 Sources found: " + ", ".join(source_info["sources"][:3]))
    if figurative_matches:
        reasons.append("✨ Figurative language detected: " + ", ".join([fm['word'] for fm in figurative_matches]))

    return {
        "classification": classification,
        "combined_fake_prob": combined_fake_prob,
        "bert_fake_prob": bert_fake_prob,
        "fake_score": fake_score,
        "real_score": real_score,
        "source_info": source_info,
        "temporal_info": temporal_info,
        "reasons": reasons,
        "response_time": time.time() - start_time
    }

# ================================================================
# MAIN APP LAYOUT
# ================================================================
st.markdown('<div class="glass-container">', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Detection", "🧠 Train Model"])

# ================================================================
# TAB 1: DETECTION
# ================================================================
with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if bert_loaded:
            if bert_source == "local":
                bert_status = f"🧠 Your BERT ({bert_num_labels}-class)"
            elif bert_source == "huggingface_fake":
                bert_status = "🧠 HF Fake News (2-class)"
            else:
                bert_status = "🧠 HF Sentiment (2-class)"
        else:
            bert_status = "⚡ Rule-based only"

        firebase_status = "🔥 Firebase Live" if FIREBASE_ACTIVE else "💻 Local Mode"
        st.markdown(
            f'''
            <div class="premium-header">
                <h1 style="font-size: 4em; margin-bottom: 10px; font-weight: 700;">🌪️ Disaster Tweet Detection System</h1>
                <p style="font-size: 1.4em; opacity: 0.95; margin-bottom: 20px;">BERT + Source/Time Analysis + Geographic Checks (No API)</p>
                <div style="display: flex; justify-content: center; flex-wrap: wrap;">
                    <span class="badge live-badge">🔴 LIVE</span>
                    <span class="badge bert-badge">{bert_status}</span>
                    <span class="badge time-badge">⏰ Time-Aware</span>
                    <span class="badge news-badge">📰 No API</span>
                    <span class="badge firebase-badge">{firebase_status}</span>
                </div>
                <p style="margin-top: 25px; font-size: 1em; opacity: 0.8;">
                    Session: {st.session_state["session_id"]}
                </p>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("### 📊 Live Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{st.session_state['stats']['total']}</div>
                <div class="stat-label">Total</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="color: #95a5a6;">{st.session_state['stats']['normal']}</div>
                <div class="stat-label">Normal</div>
            </div>
            """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="color: #10ac84;">{st.session_state['stats']['real']}</div>
                <div class="stat-label">Real</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="color: #ff6b6b;">{st.session_state['stats']['fake']}</div>
                <div class="stat-label">Fake</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: #f39c12;">{st.session_state['stats']['uncertain']}</div>
            <div class="stat-label">Uncertain</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### ℹ️ Info")
        st.info("✅ Running completely offline – no external APIs used.")
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - 💬 **Normal** – everyday conversation, no disaster
        - ✅ **Real** – matches official sources & plausible geography
        - ❌ **Fake** – sensational, impossible geography, no official sources
        - ⏰ **Time-aware** – checks freshness; historical events get reduced penalties
        - 🗺️ **Geographic check** – flags impossible events (e.g., tsunami inland)
        - 📝 **After analysis, you can provide feedback** to improve the model.
        """)
        if FIREBASE_ACTIVE:
            st.markdown("---")
            st.markdown("### 🌍 Global Stats")
            global_stats = firebase_manager.get_global_stats()
            if global_stats:
                st.metric("Global Total", global_stats.get("total", 0))
                st.metric("Global Normal", global_stats.get("normal", 0))
                st.metric("Global Real", global_stats.get("real", 0))
                st.metric("Global Fake", global_stats.get("fake", 0))
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["analyses"] = deque(maxlen=20)
            st.session_state["stats"] = {"total": 0, "normal": 0, "real": 0, "fake": 0, "uncertain": 0}
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📝 Enter Tweet")
    input_col, clear_col = st.columns([6, 1])
    with input_col:
        text_area_key = f"tweet_input_{st.session_state['input_key_counter']}"
        tweet = st.text_area(
            "Tweet input",
            value=st.session_state.get("tweet_input", ""),
            height=120,
            placeholder="Paste a tweet here to analyze... Example: 'URGENT! BREAKING: Massive earthquake in KL!'",
            key=text_area_key,
            label_visibility="collapsed"
        )
    with clear_col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button(
            "🗑️ Clear",
            on_click=clear_input,
            use_container_width=True,
            key="clear_button"
        )
    if tweet:
        st.session_state["tweet_input"] = tweet
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze_clicked and st.session_state.get("tweet_input"):
        with st.spinner("🎯 Analyzing with BERT + source + time + geography..."):
            result = classify_tweet(st.session_state["tweet_input"])
            location = extract_location(st.session_state["tweet_input"])

            if result:
                st.session_state["last_analysis"] = {
                    "tweet": st.session_state["tweet_input"],
                    "prediction": result["classification"]
                }

                st.session_state["stats"]["total"] += 1
                st.session_state["stats"][result["classification"].lower()] += 1

                analysis_record = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "tweet": st.session_state["tweet_input"][:50] + "...",
                    "classification": result["classification"],
                    "confidence": f"{100 - result.get('combined_fake_prob',0)*100:.1f}%" if result["classification"] != "NORMAL" else "100%",
                    "location": location
                }
                st.session_state["analyses"].append(analysis_record)

                if FIREBASE_ACTIVE:
                    firebase_record = {
                        "tweet": st.session_state["tweet_input"],
                        "classification": result["classification"],
                        "confidence": 1 - result.get('combined_fake_prob', 0) if result["classification"] != "NORMAL" else 1.0,
                        "location": location if location else "Unknown",
                        "figurative": len(detect_figurative_language(st.session_state["tweet_input"])) > 0,
                        "session_id": st.session_state["session_id"]
                    }
                    firebase_manager.save_analysis(firebase_record)

                st.markdown("---")

                if result["classification"] == "NORMAL":
                    card_class = "normal-card"
                    icon = "💬"
                    title = "NORMAL TWEET"
                    confidence_text = "Not a disaster"
                elif result["classification"] == "REAL":
                    card_class = "real-card"
                    icon = "✅"
                    title = "REAL DISASTER TWEET"
                    confidence_text = f"Confidence: {(100 - result['combined_fake_prob']*100):.1f}%"
                elif result["classification"] == "FAKE":
                    card_class = "fake-card"
                    icon = "❌"
                    title = "FAKE DISASTER TWEET"
                    confidence_text = f"Confidence: {(100 - result['combined_fake_prob']*100):.1f}%"
                else:
                    card_class = "uncertain-card"
                    icon = "⚠️"
                    title = "UNCERTAIN - NEEDS VERIFICATION"
                    confidence_text = f"Confidence: {(100 - result['combined_fake_prob']*100):.1f}%"

                st.markdown(f"""
                <div class="result-card {card_class}">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <span style="font-size: 3em;">{icon}</span>
                        <div>
                            <h2 style="margin: 0; font-size: 2em;">{title}</h2>
                            <p style="margin: 5px 0 0 0; opacity: 0.7;">{confidence_text}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if result["classification"] != "NORMAL":
                    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{result['fake_score']}</div>
                            <div class="metric-label">🚨 Fake Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{result['real_score']}</div>
                            <div class="metric-label">✅ Real Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{result['source_info']['score']}</div>
                            <div class="metric-label">📰 Sources</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        freshness = result['temporal_info']['recency_score'] * 100
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{freshness:.0f}%</div>
                            <div class="metric-label">⏰ Freshness</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    if result['source_info']['sources']:
                        st.markdown("### 📌 Sources Detected")
                        for src in result['source_info']['sources'][:5]:
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>📰 {src}</strong>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("### 📊 Fake Probability Analysis")
                    if bert_num_labels == 3:
                        st.info("Three-class model used – see BERT confidence above.")
                    else:
                        fake_percent = result['combined_fake_prob'] * 100
                        real_percent = 100 - fake_percent
                        st.markdown(f"""
                        <div class="progress-container">
                            <div class="progress-fill" style="width: {fake_percent}%; background: linear-gradient(90deg, #ff6b6b, #ee5253);">
                                FAKE {fake_percent:.1f}%
                            </div>
                        </div>
                        <div class="progress-container">
                            <div class="progress-fill" style="width: {real_percent}%; background: linear-gradient(90deg, #10ac84, #1dd1a1);">
                                REAL {real_percent:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                if result["reasons"]:
                    st.markdown("### 🔍 Analysis Summary")
                    for reason in result["reasons"]:
                        st.info(reason)

                st.caption(f"⚡ Response time: {result['response_time']*1000:.0f}ms")

                # Feedback section
                st.markdown("---")
                st.markdown("### 📝 Was this classification correct?")
                col_fb1, col_fb2, col_fb3, col_fb4 = st.columns(4)
                with col_fb1:
                    if st.button("✅ Correct", key="fb_correct"):
                        st.success("Thank you for your feedback!")
                with col_fb2:
                    if st.button("❌ Incorrect", key="fb_incorrect"):
                        st.session_state["show_correction"] = True

                if st.session_state.get("show_correction", False):
                    st.markdown("#### What should the correct label be?")
                    correct_label = st.radio(
                        "Select correct classification",
                        ["NORMAL", "REAL", "FAKE"],
                        key="fb_label"
                    )
                    if st.button("Submit Correction", key="fb_submit"):
                        save_feedback(st.session_state["tweet_input"], correct_label)
                        st.session_state["show_correction"] = False
                        st.rerun()

    elif analyze_clicked and not st.session_state.get("tweet_input"):
        st.warning("⚠️ Please enter a tweet to analyze")

    # History feed
    if st.session_state["analyses"] or FIREBASE_ACTIVE:
        st.markdown("---")
        st.markdown("### 📡 Recent Analyses")
        if FIREBASE_ACTIVE:
            tabA, tabB = st.tabs(["📱 My Session", "🌍 Global Feed"])
        else:
            tabA, tabB = st.tabs(["📱 My Session", "ℹ️ Firebase Offline"])
        with tabA:
            if st.session_state["analyses"]:
                feed_data = []
                for a in reversed(list(st.session_state["analyses"])[-10:]):
                    if a["classification"] == "NORMAL":
                        emoji = "💬"
                    elif a["classification"] == "REAL":
                        emoji = "✅"
                    elif a["classification"] == "FAKE":
                        emoji = "❌"
                    else:
                        emoji = "⚠️"
                    feed_data.append({
                        "Time": a["timestamp"],
                        "Tweet": a["tweet"],
                        "Result": f"{emoji} {a['classification']}",
                        "Confidence": a["confidence"],
                        "Location": a.get("location", "Unknown")
                    })
                df = pd.DataFrame(feed_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No analyses yet. Start typing to see results!")
        with tabB:
            if FIREBASE_ACTIVE:
                global_analyses = firebase_manager.get_global_analyses(limit=20)
                if global_analyses:
                    global_feed = []
                    for a in global_analyses:
                        ts = a.get("timestamp")
                        if hasattr(ts, "strftime"):
                            timestamp = ts.strftime("%H:%M:%S")
                        else:
                            timestamp = "N/A"
                        classification = a.get("classification", "NORMAL")
                        if classification == "NORMAL":
                            emoji = "💬"
                        elif classification == "REAL":
                            emoji = "✅"
                        elif classification == "FAKE":
                            emoji = "❌"
                        else:
                            emoji = "⚠️"
                        global_feed.append({
                            "Time": timestamp,
                            "Session": a.get("session_id", "???")[:8],
                            "Tweet": a.get("tweet", "Unknown")[:50] + "...",
                            "Result": f"{emoji} {classification}",
                            "Location": a.get("location", "Unknown")
                        })
                    df_global = pd.DataFrame(global_feed)
                    st.dataframe(df_global, use_container_width=True, hide_index=True)
                else:
                    st.info("No global data yet. Be the first to analyze!")
            else:
                st.warning("Firebase is not connected. Global feed unavailable.")

# ================================================================
# TAB 2: TRAINING
# ================================================================
with tab2:
    st.markdown("## 🧠 Train Your Own BERT Model")
    st.markdown("""
    Upload a CSV file with `text` and `target` columns, or use the feedback data collected from corrections.
    - For binary classification, `target` should be 0 (non-disaster) or 1 (disaster).
    - For three‑class classification, `target` should be 0 (NORMAL), 1 (REAL), or 2 (FAKE).
    """)

    include_feedback = st.checkbox("Include feedback data from corrections (saved in feedback_data.csv)")
    uploaded_file = st.file_uploader("Choose a CSV file (optional)", type="csv")

    num_classes = st.radio("Number of output classes", [2, 3], index=1 if NUM_CLASSES==3 else 0)

    if st.button("Start Training"):
        dfs = []
        if uploaded_file is not None:
            df_main = pd.read_csv(uploaded_file)
            if 'text' not in df_main.columns or 'target' not in df_main.columns:
                st.error("Uploaded CSV must contain 'text' and 'target' columns.")
                st.stop()
            dfs.append(df_main)
        if include_feedback and os.path.exists(FEEDBACK_FILE):
            df_fb = pd.read_csv(FEEDBACK_FILE)
            st.write(f"Feedback data contains {len(df_fb)} samples.")
            dfs.append(df_fb)
        if not dfs:
            st.error("No data provided. Please upload a CSV or check 'Include feedback data'.")
            st.stop()

        combined_df = pd.concat(dfs, ignore_index=True)
        st.write(f"Combined dataset: {combined_df.shape[0]} tweets")
        st.write("Class distribution:")
        st.bar_chart(combined_df['target'].value_counts())

        epochs = st.number_input("Epochs", min_value=1, max_value=10, value=3, key="train_epochs")
        batch_size = st.number_input("Batch size", min_value=8, max_value=64, value=16, step=8, key="train_batch")
        lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-3, value=2e-5, format="%.5f", key="train_lr")

        st.cache_resource.clear()
        with st.container():
            train_model(combined_df, epochs=epochs, batch_size=batch_size, lr=lr, num_classes=num_classes)
        st.success("Training finished! Switching to detection tab now uses your new model.")

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown(
    '''
    <div class="footer">
        <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 20px; flex-wrap: wrap;">
            <span class="badge" style="background: linear-gradient(135deg, #667eea, #5a67d8); color: white;">🚀 v6.0</span>
            <span class="badge" style="background: #95a5a6; color: white;">💬 Normal</span>
            <span class="badge" style="background: #10ac84; color: white;">✅ Real</span>
            <span class="badge" style="background: #ff6b6b; color: white;">❌ Fake</span>
            <span class="badge" style="background: #f39c12; color: white;">⚠️ Uncertain</span>
        </div>
        <p style="color: #666; font-size: 0.95em;">
            Disaster Tweet Detection System | Powered by BERT + Source/Time Analysis + Geographic Feasibility + Numerical Boost
        </p>
        <p style="color: #888; font-size: 0.85em; margin-top: 10px;">
            Completely offline – no external APIs required. Train with CSV or feedback corrections. Supports 2‑class or 3‑class models.
        </p>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
