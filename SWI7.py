# -*- coding: utf-8 -*-
# Streamlit deployment of SWI Wedge Length Ratio – Smart Predictor
# Layout:
#   Left  = Inputs + large prediction readout
#   Right = Reference sketch (auto-fit, with upload fallback)
#   Bottom = Predict, Clear, Recall, Save Inputs (JSON), Load Inputs
# Tabs:
#   Predict, Explain (SHAP), Batch, History, Article Info (Scientific Reports 15)
# Deterministic CPU predictions; cached model; no unhashable-arg caching.

import json
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib
import shap
import matplotlib.pyplot as plt

# ------------------------------
# Page / theme config
# ------------------------------
st.set_page_config(
    page_title="SWI Wedge Length Ratio – Smart Predictor",
    page_icon="🌊",
    layout="wide",
)

# ------------------------------
# Light Theme CSS (dark cards with white text + accents)
# ------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Source+Serif+4:wght@500;700&display=swap');

      :root{
        --ui-bg: #f5f7fa;          /* light background */
        --ui-card: #1e293b;        /* dark panels/cards */
        --ui-border: #3b475a;      /* borders */
        --ui-text: #ffffff;        /* white text for dark panels */
        --ui-accent-blue: #3b82f6; /* blue */
        --ui-accent-green: #10b981;/* green */
        --ui-accent-red: #ef4444;  /* red */
        --ui-accent-black: #0f172a;/* deep black */
      }

      .stApp { background: var(--ui-bg); color: var(--ui-accent-black); }
      .block-container {padding-top: 1rem; padding-bottom: 2rem;}
      body, .stApp, p, div, span, label, input, select, textarea {
        font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 16px;
        color: var(--ui-accent-black);
      }
      h1, h2, h3, h4, h5, h6 {
        font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
        color: var(--ui-accent-black);
        font-weight: 700;
      }

      /* Cards */
      .card {
        padding: 1rem 1.25rem; border-radius: 12px;
        border: 1px solid var(--ui-border); background: var(--ui-card);
        box-shadow: 0 2px 8px rgba(0,0,0,.3);
        color: var(--ui-text);
      }
      .big-number {font-size: 44px; font-weight: 800; color: var(--ui-accent-blue); margin: .2rem 0 .8rem;}

      /* Inputs */
      input, textarea, select {
        background: var(--ui-card) !important; color: var(--ui-text) !important;
        border: 1px solid var(--ui-border) !important; border-radius: 10px !important;
      }
      .stNumberInput input {
        background: var(--ui-card) !important; color: var(--ui-text) !important;
        border: 1px solid var(--ui-border) !important; border-radius: 10px !important;
      }

      /* Buttons */
      .stButton > button[kind="primary"]{
        background: var(--ui-accent-blue) !important; color: #ffffff !important;
        border: none; border-radius: 10px; padding: .6rem 1rem; font-weight: 800;
      }
      .stButton > button[kind="secondary"]{
        background: var(--ui-card) !important; color: var(--ui-text) !important;
        border: 1px solid var(--ui-border); border-radius: 10px; padding: .55rem 1rem; font-weight: 700;
      }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"]{ gap: 8px; }
      .stTabs [data-baseweb="tab"]{
        background: var(--ui-card); color: var(--ui-text);
        border: 1px solid var(--ui-border);
        border-radius: 10px; padding: .45rem 1rem; font-weight: 700;
      }
      .stTabs [aria-selected="true"]{
        background: var(--ui-accent-green) !important; color: #ffffff !important; border-color: var(--ui-accent-green) !important;
      }

      /* Tables */
      .stDataFrame { background: var(--ui-card); border: 1px solid var(--ui-border); border-radius: 12px; padding: .25rem; }
      [data-testid="stDataFrame"] thead th { background: var(--ui-accent-black) !important; color: var(--ui-text); font-weight: 800; }

      /* Uploader */
      [data-testid="stFileUploader"] section {
        border: 1px dashed var(--ui-border); background: var(--ui-card);
        border-radius: 12px; padding: .8rem; color: var(--ui-text);
      }

      .muted {color: var(--ui-text); font-size: 0.95rem; opacity:0.7;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------
# Config / constants (unchanged)
# ------------------------------
MODEL_PATH = "models/XGB.joblib"
IMAGE_CANDIDATES = [
    Path("assets/sketch22.png"),
    Path("assets/sketch.png"),
    Path("sketch22.png"),
]
FEATURE_KEYS = ['X1','X2','X3','X4','X5','X6','X7','X8']
LABELS = {
    'X1': "ρf/ρs   (Relative density)",
    'X2': "Hf/H    (Relative fracture aperture height)",
    'X3': "Df/H    (Relative fracture aperture diameter)",
    'X4': "Hd/H    (Relative subsurface dam height)",
    'X5': "Ld/H    (Relative subsurface dam distance)",
    'X6': "Hw/H    (Relative well height)",
    'X7': "Lw/H    (Relative well distance)",
    'X8': "Qw/KH²  (Relative well abstraction rate)",
}
# ... keep the rest of ranges, presets, helpers, caching, etc. as in your original code ...


# ------------------------------
# Header (updated title)
# ------------------------------
st.title("Assessing the Impact of Groundwater Abstraction and Concrete Dam Fractures on Saltwater Intrusion Using Numerical Modelling and Interpretable Machine Learning")
st.caption("For users, technicians, water resources engineers, and hydrogeologists – quick, reliable, and explainable.")

# ------------------------------
# Tabs
# ------------------------------
tab_predict, tab_explain, tab_batch, tab_hist, tab_article = st.tabs(
    ["Predict", "Explain", "Batch", "History", "Article Info"]
)

# (⚠️ Keep the rest of your Predict / Explain / Batch / History tabs code exactly as before)

# ==============================
# ARTICLE INFO TAB (updated details)
# ==============================
with tab_article:
    st.markdown("### Article & Authors")
    st.markdown(
        """
        <div style="font-size:28px; font-weight:800; line-height:1.25;">
        Assessing the Impact of Groundwater Abstraction and Concrete Dam Fractures on Saltwater Intrusion<br>
        Using Numerical Modelling and Interpretable Machine Learning
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size:20px; font-weight:700; margin-top:0.5rem;">
        Asaad M. Armanuos¹,*, Martina Zeleňáková², Mohamed Kamel Elshaarawy³,*
        </div>
        """, unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size:18px; margin-top:0.5rem;">
        ¹ Irrigation and Hydraulics Engineering Department, Faculty of Engineering, Tanta University, Tanta 31733, Egypt<br>
        ² Institute of Environmental Engineering, Faculty of Civil Engineering, Technical University of Košice, 04200 Košice, Slovakia<br>
        ³ Civil Engineering Department, Faculty of Engineering, Horus University-Egypt, New Damietta 34517, Egypt
        </div>
        """, unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size:18px; font-style:italic; margin-top:0.6rem;">
        Scientific Reports 15 (in press)
        </div>
        """, unsafe_allow_html=True,
    )

    citation = (
        "Armanuos, A.M., Zeleňáková, M., & Elshaarawy, M.K. "
        "(in press). Assessing the Impact of Groundwater Abstraction and Concrete Dam Fractures on Saltwater Intrusion "
        "Using Numerical Modelling and Interpretable Machine Learning. Scientific Reports, 15."
    )
    st.download_button("Download Citation (.txt)", data=citation.encode("utf-8"),
                       file_name="citation.txt", mime="text/plain")
