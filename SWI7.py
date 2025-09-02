# -*- coding: utf-8 -*-
# Streamlit deployment of SWI Wedge Length Ratio – Smart Predictor
# Full version with pure Black & White theme

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
# Pure Black & White Theme CSS
# ------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Source+Serif+4:wght@500;700&display=swap');

      :root{
        --ui-bg: #ffffff;          /* white background */
        --ui-card: #ffffff;        /* white panels/cards */
        --ui-border: #000000;      /* black borders */
        --ui-text: #000000;        /* black text */
        --ui-accent-black: #000000;/* black for buttons/tabs */
        --ui-accent-white: #ffffff;/* white text for black buttons */
      }

      .stApp { background: var(--ui-bg); color: var(--ui-text); }
      .block-container {padding-top: 1rem; padding-bottom: 2rem;}
      body, .stApp, p, div, span, label, input, select, textarea {
        font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 16px;
        color: var(--ui-text);
      }
      h1, h2, h3, h4, h5, h6 {
        font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
        color: var(--ui-text);
        font-weight: 700;
      }

      /* Cards */
      .card {
        padding: 1rem 1.25rem; border-radius: 12px;
        border: 1px solid var(--ui-border); background: var(--ui-card);
        color: var(--ui-text);
      }
      .big-number {font-size: 44px; font-weight: 800; color: var(--ui-text); margin: .2rem 0 .8rem;}

      /* Inputs */
      input, textarea, select {
        background: var(--ui-bg) !important; color: var(--ui-text) !important;
        border: 1px solid var(--ui-border) !important; border-radius: 10px !important;
      }
      .stNumberInput input {
        background: var(--ui-bg) !important; color: var(--ui-text) !important;
        border: 1px solid var(--ui-border) !important; border-radius: 10px !important;
      }

      /* Buttons */
      .stButton > button[kind="primary"]{
        background: var(--ui-accent-black) !important; color: var(--ui-accent-white) !important;
        border: none; border-radius: 10px; padding: .6rem 1rem; font-weight: 800;
      }
      .stButton > button[kind="secondary"]{
        background: var(--ui-accent-white) !important; color: var(--ui-accent-black) !important;
        border: 1px solid var(--ui-border); border-radius: 10px; padding: .55rem 1rem; font-weight: 700;
      }

      /* Tabs */
      /* Tabs */
      .stTabs [data-baseweb="tab-list"]{ gap: 8px; }
      .stTabs [data-baseweb="tab"]{
         background: var(--ui-bg); 
         color: var(--ui-text);
         border: 1px solid var(--ui-border);
         border-radius: 10px; 
         padding: .45rem 1rem;   
         font-weight: 700;
       }
       .stTabs [aria-selected="true"]{
         background: var(--ui-bg) !important;   /* keep white background */
         color: var(--ui-accent-black) !important; /* black text */
         border: 2px solid var(--ui-accent-black) !important; /* slightly thicker border */
         font-weight: 800;  /* bold for active tab */
       }


      /* Tables */
      .stDataFrame { background: var(--ui-bg); border: 1px solid var(--ui-border); border-radius: 12px; padding: .25rem; }
      [data-testid="stDataFrame"] thead th { background: var(--ui-accent-black) !important; color: var(--ui-accent-white); font-weight: 800; }

      /* Uploader */
      [data-testid="stFileUploader"] section {
        border: 1px dashed var(--ui-border); background: var(--ui-bg);
        border-radius: 12px; padding: .8rem; color: var(--ui-text);
      }

      .muted {color: var(--ui-text); font-size: 0.95rem; opacity:0.7;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Force matplotlib to black & white only
# ------------------------------
plt.rcParams.update({
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "lines.color": "black",
    "patch.edgecolor": "black",
    "patch.facecolor": "white",
})

# ------------------------------
# SHAP plots override to black & white
# ------------------------------
def shap_summary_plot(shap_values, features, feature_names):
    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values,
        features=features,
        feature_names=feature_names,
        show=False,
        color="black"  # force single color
    )
    plt.setp(ax.collections, color="black")
    return fig

# ------------------------------
# Config / constants
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

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ------------------------------
# Helper: make prediction
# ------------------------------
def make_prediction(inputs):
    X = np.array([inputs[f] for f in FEATURE_KEYS]).reshape(1, -1)
    return model.predict(X)[0]

# ------------------------------
# Header
# ------------------------------
st.title("Assessing the Impact of Groundwater Abstraction and Concrete Dam Fractures on Saltwater Intrusion Using Numerical Modelling and Interpretable Machine Learning")
st.caption("For users, technicians, water resources engineers, and hydrogeologists – quick, reliable, and explainable.")

# ------------------------------
# Tabs
# ------------------------------
tab_predict, tab_explain, tab_batch, tab_hist, tab_article = st.tabs(
    ["Predict", "Explain", "Batch", "History", "Article Info"]
)

# ==============================
# PREDICT TAB
# ==============================
with tab_predict:
    st.subheader("Input Parameters")

    inputs = {}
    for key in FEATURE_KEYS:
        inputs[key] = st.number_input(LABELS[key], min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    if st.button("Predict", type="primary"):
        pred = make_prediction(inputs)
        st.markdown(f"<div class='card'><div class='big-number'>{pred:.3f}</div>Predicted Wedge Length Ratio</div>", unsafe_allow_html=True)

    st.download_button("Save Inputs (.json)", data=json.dumps(inputs, indent=2), file_name="inputs.json", mime="application/json")

# ==============================
# EXPLAIN TAB
# ==============================
with tab_explain:
    st.subheader("Explain Predictions (SHAP)")
    X_sample = np.random.rand(10, len(FEATURE_KEYS))
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    fig = shap_summary_plot(shap_values, X_sample, FEATURE_KEYS)
    st.pyplot(fig)

# ==============================
# BATCH TAB
# ==============================
with tab_batch:
    st.subheader("Batch Predictions")
    uploaded = st.file_uploader("Upload CSV file with columns X1..X8", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        preds = model.predict(df[FEATURE_KEYS])
        df["Prediction"] = preds
        st.dataframe(df)
        st.download_button("Download Results (.csv)", df.to_csv(index=False), file_name="batch_results.csv")

# ==============================
# HISTORY TAB
# ==============================
with tab_hist:
    st.subheader("Prediction History")
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Recall Last Prediction"):
        if st.session_state.history:
            st.json(st.session_state.history[-1])
        else:
            st.info("No history yet.")

    if st.button("Clear History"):
        st.session_state.history = []

    if "history" in st.session_state and st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history))

# ==============================
# ARTICLE INFO TAB
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

