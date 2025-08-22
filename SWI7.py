# -*- coding: utf-8 -*-
# Streamlit deployment of SWI Wedge Length Ratio – Smart Predictor
# Now with:
#   • SHAP Explainability tab (Global summary + Dependence + Local waterfall)
#   • Sliders (with sensible ranges) + presets + recall/save/load
#   • Deterministic CPU predictions; cached model
#   • Article Info (Scientific Reports 15)

import json
from io import BytesIO
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

# Small CSS for cards & big number
st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 2rem;}
      .big-number {font-size: 44px; font-weight: 800; margin: 0.2rem 0 0.8rem 0;}
      .card {padding: 1rem 1.25rem; border-radius: 10px; border: 1px solid var(--secondary-bg); background: var(--background-color);}
      .muted {color: #8aa0b2; font-size: 0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Config / constants
# ------------------------------
MODEL_PATH = "models/XGB.joblib"       # put your model file here
IMAGE_PATH = "assets/sketch22.png"     # reference sketch

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

# Sensible slider ranges (tune if you have domain bounds)
FEATURE_RANGES = {
    'X1': (0.98, 1.05, 1.02),
    'X2': (0.00, 0.50, 0.10),
    'X3': (0.00, 0.50, 0.08),
    'X4': (0.00, 1.00, 0.30),
    'X5': (0.00, 2.00, 1.00),
    'X6': (0.00, 1.00, 0.40),
    'X7': (0.00, 2.50, 1.50),
    'X8': (0.00, 0.50, 0.05),
}

PRESETS = {
    "— choose a preset —": None,
    "Baseline":          [0.9756, 0.2, 0.005, 0.60, 0.50, 0.45, 0.50, 0.0001],
    "High Abstraction":  [0.9756, 0.2, 0.01, 0.60, 0.50, 0.6, 0.50, 0.0002],
    "Near Well":         [0.9756, 0.1, 0.002, 0.80, 0.60, 0.15, 0.12, 0.000011],
    "Near Dam":          [0.9756, 0.2, 0.005, 0.60, 0.30, 0.60, 0.30, 0.0002],
    "Case of Akrotiri coastal aquifer, Cyprus":          [0.9756, 0.1, 0.002, 0.80, 0.60, 0.15, 0.30, 0.0001],
}

# ------------------------------
# Helpers
# ------------------------------
def _lock_model_deterministic(xgb_model):
    """Force CPU + single thread; return expected feature names if available."""
    expected = None
    try:
        xgb_model.get_booster().set_param({"predictor": "cpu_predictor", "nthread": 1})
    except Exception:
        pass
    try:
        xgb_model.set_params(n_jobs=1, nthread=1)
    except Exception:
        pass
    try:
        if hasattr(xgb_model, "feature_names_in_"):
            expected = list(xgb_model.feature_names_in_)
        else:
            expected = list(xgb_model.get_booster().feature_names)
    except Exception:
        expected = None
    return expected

def _ordered_df(values: dict, expected_names):
    """Build 1-row DataFrame with correct column order."""
    if expected_names and set(map(str, expected_names)) == set(FEATURE_KEYS):
        cols = list(map(str, expected_names))
    else:
        cols = FEATURE_KEYS[:]
    row = [values[c] for c in cols]
    return pd.DataFrame([row], columns=cols).astype(np.float32)

@st.cache_resource(show_spinner=False)
def load_model_and_expected():
    model = joblib.load(MODEL_PATH)
    expected = _lock_model_deterministic(model)
    return model, expected

def predict_one(model, expected, values_dict):
    X = _ordered_df(values_dict, expected)
    y = model.predict(X)
    return float(np.ravel(y)[0])

def read_image(path):
    try:
        return Image.open(path)
    except Exception:
        return None

def json_download_bytes(obj, filename="inputs.json"):
    buf = BytesIO()
    buf.write(json.dumps(obj, indent=2).encode("utf-8"))
    buf.seek(0)
    return buf

def sample_background_df(ranges: dict, n: int = 256) -> pd.DataFrame:
    """Uniformly sample within slider ranges for background SHAP."""
    data = {}
    for k in FEATURE_KEYS:
        lo, hi, default = ranges[k]
        data[k] = np.random.uniform(lo, hi, size=n)
    return pd.DataFrame(data).astype(np.float32)

@st.cache_resource(show_spinner=False)
def get_explainer(model):
    # TreeExplainer under the hood (fast for XGBoost)
    return shap.Explainer(model)

@st.cache_data(show_spinner=False)
def shap_background_values(model, expected, ranges, n=256):
    """Compute SHAP values for a synthetic background once and cache."""
    explainer = get_explainer(model)
    df_bg = sample_background_df(ranges, n)
    # Align to model's expected order
    X_bg = _ordered_df({k: 0.0 for k in FEATURE_KEYS}, expected)  # just to get column order
    X_bg = df_bg[X_bg.columns]
    sv = explainer(X_bg)
    return sv, X_bg

# ------------------------------
# Session state
# ------------------------------
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts, each with time + Xs + pred
if "current_pred" not in st.session_state:
    st.session_state.current_pred = None
if "current_inputs" not in st.session_state:
    # initialize with defaults (slider midpoints)
    st.session_state.current_inputs = {k: FEATURE_RANGES[k][2] for k in FEATURE_KEYS}

# ------------------------------
# Header
# ------------------------------
st.title("Modeling Saltwater Intrusion with Interpretable ML")
st.caption("For users, technicians, and stakeholders – quick, reliable, and explainable.")

# Tabs
tab_predict, tab_explain, tab_batch, tab_hist, tab_article = st.tabs(
    ["Predict", "Explain", "Batch", "History", "Article Info"]
)

# ==============================
# PREDICT TAB
# ==============================
with tab_predict:
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("#### Prediction")
        big = "—" if st.session_state.current_pred is None else f"{st.session_state.current_pred:.6f}"
        st.markdown(
            f"<div class='card'><div class='big-number'>{big}</div>"
            f"<div>Predicted Relative SWI wedge length (L/H)</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Input Parameters (Dimensionless)")
        # Preset first
        preset = st.selectbox("Preset", PRESETS.keys(), index=0)
        if PRESETS.get(preset):
            vals = PRESETS[preset]
            st.session_state.current_inputs = {k: float(v) for k, v in zip(FEATURE_KEYS, vals)}

        # Sliders (reflect & update session state)
        for k in FEATURE_KEYS:
            lo, hi, df = FEATURE_RANGES[k]
            st.session_state.current_inputs[k] = st.slider(
                LABELS[k], min_value=float(lo), max_value=float(hi),
                value=float(st.session_state.current_inputs.get(k, df)),
                step=0.001, format="%.6f"
            )

        # Bottom buttons
        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
        with c1:
            if st.button("Predict", use_container_width=True, type="primary"):
                try:
                    model, expected = load_model_and_expected()
                    try: model.get_booster().set_param({"predictor":"cpu_predictor","nthread":1})
                    except Exception: pass
                    values = {k: float(st.session_state.current_inputs[k]) for k in FEATURE_KEYS}
                    y = predict_one(model, expected, values)
                    st.session_state.current_pred = y
                    st.session_state.last_inputs = [values[k] for k in FEATURE_KEYS]
                    # history
                    rec = {"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **values, "Prediction": round(float(y), 6)}
                    st.session_state.history.append(rec)
                    st.success("Prediction complete.")
                except FileNotFoundError:
                    st.error(f"Model not found at `{MODEL_PATH}`. Upload it to your repo.")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        with c2:
            if st.button("Clear", use_container_width=True):
                st.session_state.current_pred = None
                st.session_state.current_inputs = {k: FEATURE_RANGES[k][2] for k in FEATURE_KEYS}
                st.info("Cleared.")
        with c3:
            disabled = st.session_state.last_inputs is None
            if st.button("Recall Last", use_container_width=True, disabled=disabled):
                if st.session_state.last_inputs is not None:
                    for k, v in zip(FEATURE_KEYS, st.session_state.last_inputs):
                        st.session_state.current_inputs[k] = float(v)
                    st.success("Recalled last inputs.")
        with c4:
            # Save inputs
            if st.button("Save Inputs", use_container_width=True):
                buf = BytesIO()
                buf.write(json.dumps(st.session_state.current_inputs, indent=2).encode("utf-8"))
                buf.seek(0)
                st.download_button("Download inputs.json", data=buf, file_name="inputs.json",
                                   mime="application/json", use_container_width=True)
        with c5:
            # Load inputs
            up = st.file_uploader("Load Inputs", type=["json"], label_visibility="collapsed", key="upl_json_predict")
            if up is not None:
                try:
                    data = json.loads(up.read().decode("utf-8"))
                    for k in FEATURE_KEYS:
                        if k in data:
                            st.session_state.current_inputs[k] = float(data[k])
                    st.success("Inputs loaded.")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

    with col_right:
        st.markdown("#### Reference Sketch")
        img = read_image(IMAGE_PATH)
        if img is None:
            st.info("Image not found. Place it at `assets/sketch22.png` in the repo.")
        else:
            st.image(img, use_container_width=True)

# ==============================
# EXPLAIN TAB (SHAP)
# ==============================
with tab_explain:
    st.markdown("### Explain (SHAP)")

    try:
        model, expected = load_model_and_expected()
    except Exception as e:
        st.error(f"Load model first. {e}")
        st.stop()

    # Background SHAP (global)
    with st.expander("Global importance (summary) — computed on synthetic background", expanded=True):
        n_bg = st.slider("Background sample size", 100, 2000, 256, 50,
                         help="SHAP will be computed on uniformly sampled points within the slider ranges.")
        sv_bg, X_bg = shap_background_values(model, expected, FEATURE_RANGES, n=n_bg)

        colA, colB = st.columns(2)
        with colA:
            st.write("**Mean absolute SHAP (bar)**")
            fig = plt.figure(figsize=(7, 4))
            # NOTE: use .values & X_bg (DataFrame) to stay compatible across SHAP versions
            shap.summary_plot(sv_bg.values, X_bg, plot_type="bar", show=False)
            st.pyplot(fig, clear_figure=True, bbox_inches="tight")

        with colB:
            st.write("**Beeswarm (distribution of impacts)**")
            fig = plt.figure(figsize=(7, 4))
            shap.summary_plot(sv_bg.values, X_bg, show=False)
            st.pyplot(fig, clear_figure=True, bbox_inches="tight")

        # Top features for dependence
        mean_abs = np.mean(np.abs(sv_bg.values), axis=0)
        ordered_cols = list(X_bg.columns)
        order_idx = np.argsort(-mean_abs)
        top_feats = [ordered_cols[i] for i in order_idx[:5]]

        st.write("**Dependence plots**")
        dep1 = st.selectbox("Primary feature", top_feats, index=0)
        dep2_options = ["(auto color)"] + [c for c in ordered_cols if c != dep1]
        dep2 = st.selectbox("Color by (optional)", dep2_options, index=0)

        fig = plt.figure(figsize=(7, 4))
        if dep2 == "(auto color)":
            shap.dependence_plot(dep1, sv_bg.values, X_bg, show=False)
        else:
            shap.dependence_plot(dep1, sv_bg.values, X_bg, interaction_index=dep2, show=False)
        st.pyplot(fig, clear_figure=True, bbox_inches="tight")

    # Local SHAP (current input)
    with st.expander("Local explanation for current inputs", expanded=True):
        if st.session_state.current_pred is None:
            st.info("Make a prediction first in the Predict tab to see the local explanation.")
        else:
            values = {k: float(st.session_state.current_inputs[k]) for k in FEATURE_KEYS}
            X_one = _ordered_df(values, expected)
            explainer = get_explainer(model)
            sv_one = explainer(X_one)

            st.write("**Waterfall (feature contributions)**")
            try:
                fig = plt.figure(figsize=(7, 5))
                shap.plots.waterfall(sv_one[0], max_display=8, show=False)
                st.pyplot(fig, clear_figure=True, bbox_inches="tight")
            except Exception:
                # Fallback: local bar
                fig = plt.figure(figsize=(7, 4))
                shap.plots.bar(sv_one[0], show=False, max_display=8)
                st.pyplot(fig, clear_figure=True, bbox_inches="tight")

            st.caption("Note: SHAP is computed with a Tree-based explainer on your XGBoost model.")

# ==============================
# BATCH TAB
# ==============================
with tab_batch:
    st.markdown("### Batch Predictions (CSV → CSV)")
    st.write("Upload a CSV with columns **X1..X8** in any order; we will align the order automatically.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        try:
            df = pd.read_csv(up)
            model, expected = load_model_and_expected()
            if not set(FEATURE_KEYS).issubset(df.columns):
                missing = [c for c in FEATURE_KEYS if c not in df.columns]
                st.error(f"CSV missing columns: {missing}")
            else:
                # align order to model
                ordered_cols = _ordered_df({k: 0.0 for k in FEATURE_KEYS}, expected).columns
                X = df[ordered_cols].astype(np.float32)
                try:
                    model.get_booster().set_param({"predictor":"cpu_predictor","nthread":1})
                except Exception:
                    pass
                preds = model.predict(X)
                out = df.copy()
                out["Pred_L_over_H"] = preds
                st.success("Batch predictions complete.")
                st.dataframe(out.head(20), use_container_width=True)
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes,
                                   file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch error: {e}")

# ==============================
# HISTORY TAB
# ==============================
with tab_hist:
    st.markdown("### Session History")
    if len(st.session_state.history) == 0:
        st.info("No history yet. Make a prediction in the Predict tab.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download History CSV", data=csv_bytes,
                           file_name="history.csv", mime="text/csv")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# ==============================
# ARTICLE INFO TAB
# ==============================
with tab_article:
    st.markdown("### Article & Authors")
    st.markdown(
        """
        <div style="font-size:28px; font-weight:800; line-height:1.25;">
        Modeling the Impact of Groundwater Abstraction and Concrete Dam Fractures on
        Saltwater Intrusion Using Interpretable Machine Learning Models
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
        <div style="font-size:18px; margin-top:0.6rem;">
        Emails: asaad.matter@f-eng.tanta.edu.eg; martina.zelenakova@tuke.sk; melshaarawy@horus.edu.eg
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
        "(in press). Modeling the Impact of Groundwater Abstraction and Concrete Dam Fractures on "
        "Saltwater Intrusion Using Interpretable Machine Learning Models. Scientific Reports, 15."
    )
    st.download_button("Download Citation (.txt)", data=citation.encode("utf-8"),
                       file_name="citation.txt", mime="text/plain")

