# ================================================================
# DISASTER TWEET AI DETECTOR - WITH AUTO-DOWNLOAD
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
# DOWNLOAD NLTK DATA
# ================================================================
@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))

stop_words = download_nltk_data()

# ================================================================
# DOWNLOAD MODEL FUNCTION - ADD THIS SECTION
# ================================================================

# You need to upload your model to Google Drive and get a direct download link
# Follow instructions at the bottom of this file to get your link

@st.cache_resource(show_spinner="🔄 Downloading BERT model...")
def download_and_load_model():
    """
    Download BERT model from cloud storage if not present locally
    """
    model_path = "bert_disaster_model_fine_tuned"
    
    # Check if model already exists
    if os.path.exists(model_path):
        try:
            with st.spinner("Loading existing BERT model..."):
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
                model.eval()
            return model, tokenizer, True
        except Exception as e:
            st.warning(f"Existing model corrupted: {e}. Will re-download.")
    
    # Download the model
    st.info("📥 BERT model not found locally. Downloading from cloud storage...")
    
    # TODO: REPLACE THIS WITH YOUR ACTUAL DOWNLOAD LINK
    # How to get a direct download link from Google Drive:
    # 1. Upload your zipped model to Google Drive
    # 2. Share the file and get a sharing link
    # 3. Use this tool: https://sites.google.com/site/gdocs2direct/
    # 4. Or manually convert: https://drive.google.com/uc?export=download&id=YOUR_FILE_ID
    
    download_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_HERE"
    
    try:
        # Download the zip file with progress bar
        with st.spinner("Downloading model (this may take a few minutes)..."):
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Download with progress
            with open("bert_model.zip", "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                            status_text.text(f"Downloaded: {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB")
            
            status_text.text("Download complete! Extracting...")
            
            # Extract the zip file
            with zipfile.ZipFile("bert_model.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Clean up
            os.remove("bert_model.zip")
            progress_bar.empty()
            status_text.empty()
        
        # Load the model
        with st.spinner("Loading BERT model..."):
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            model.eval()
        
        st.success("✅ Model downloaded and loaded successfully!")
        return model, tokenizer, True
        
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.info("Falling back to Mock API...")
        return None, None, False

# ================================================================
# LOAD THE MODEL (REPLACE YOUR EXISTING LOAD FUNCTION)
# ================================================================

# Replace your old load_local_bert_model() with this:
bert_model, bert_tokenizer, bert_loaded = download_and_load_model()

# ================================================================
# SESSION STATE INITIALIZATION
# ================================================================
if "session_id" not in st.session_state:
    st.session_state["session_id"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "bert" if bert_loaded else "mock"
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
# [REST OF YOUR CODE CONTINUES HERE...]
# ================================================================
# Keep all your existing code below this point:
# - CONSTANTS
# - Preprocessing functions
# - Prediction functions
# - Display functions
# - UI components
