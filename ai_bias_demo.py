# ai_bias_demo.py
# Streamlit App: Explaining AI Bias with Interactive Demo
# Run: streamlit run ai_bias_demo.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, precision_score,
    recall_score, accuracy_score
)
from sklearn.model_selection import train_test_split

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="AI Bias Demo", layout="wide", page_icon="⚖️")
st.title("⚖️ AI Bias — Interactive Demo")

st.markdown(
    """
This app shows **how bias can arise** in AI systems and what we can do about it.

**How to use**  
1. Adjust data generation sliders to simulate **group differences** and **label bias**.  
2. Pick a model and choose **mitigations** (remove proxy, reweight, group thresholds).  
3. Compare metrics, confusion matrices, score distributions and ROC curves **by group**.  
"""
)

# -------------------------------
# Controls
# -------------------------------
with st.sidebar:
    st.header("Data generation")
    seed = st.number_input("Random seed", 0, 10_000, value=123, step=1)
    n_samples = st.slider("Dataset size", 500, 20000, value=2000, step=500)

    st.markdown("**Group proportions**")
    p_group1 = st.slider("Share of Group B (protected=1)", 0.05, 0.95, value=0.5, step=0.05)

    st.markdown("**Feature disparity** (shift in mean of key feature for Group B)")
    feat_shift = st.slider("Feature shift (Group B vs A)", -2.0, 2.0, value=-0.5, step=0.1)

    st.markdown("**Label bias** (historical bias in outcomes)")
    label_bias = st.slider("Extra negative labels for Group B (prob.)", 0.0, 0.5, value=0.2, step=0.05)

    st.markdown("**Noise**")
    noise = st.slider("Label noise (both groups)", 0.0, 0.3, value=0.05, step=0.01)

    st.markdown("---")
    st.header("Model & Mitigations")
    model_choice = st.selectbox("Model", ["Logistic Regression"])
    remove_proxy = st.checkbox("Remove proxy feature (drop Group*Feature interaction)", value=False)
    use_reweight = st.checkbox("Re-weight training samples (counter-label-bias)", value=False)

    st.markdown("**Decision thresholds**")
    global_thr = st.slider("Global threshold", 0.0, 1.0, value=0.5, step=0.01)
    group_specific_thresholds = st.checkbox("Use group-specific thresholds", value=False)
    thr_g0 = st.slider("Threshold for Group A", 0.0, 1.0, value=0.5, step=0.01)
    thr_g1 = st.slider("Threshold for Group B", 0.0, 1.0, value=0.5, step=0.01)

    st.markdown("---")
    st.caption("Tip: start with moderate bias and compare metrics, then try mitigations.")

# -------------------------------
# Synthetic data generator
# -------------------------------
def generate_data(n=2000, p_g1=0.5, feat_shift=-0.5, label_bias=0.2, noise=0.05, rng=None):
    """
    Binary classification with a protected attribute 'group' ∈ {0 (A), 1 (B)}.
    Base signal: y* = w1*x1 + w2*x2 + w3*x3 + bias + epsilon
    Group B has shifted feature distribution (disparity).
    Label bias: extra chance of forcing negative labels for Group B (historical discrimination).
    """
    if rng is None:
        rng = np.random.default_rng(123)
    group = (rng.random(n) < p_g1).astype(int)  # 1 = Group B (protected)

    # Features
    # x1 is a key feature (e.g., credit score), with group disparity
    x1 = rng.normal(loc=0.0 + feat_shift*group, scale=1.0, size=n)
    x2 = rng.normal(0, 1, size=n)
    x3 = 0.5 * x1 + rng.normal(0, 1, size=n)  # proxy-ish feature (correlated with x1 & group)

    # Linear score -> probability
    lin = 1.2*x1 + 0.8*x2 + 0.6*x3 + (-0.2)  # base signal
    p = 1 / (1 + np.exp(-lin))

    # True labels (before bias & noise)
    y = (rng.random(n) < p).astype(int)

    # Historical label bias: Group B more likely to be marked negative
    flip_to_negative = (group == 1) & (rng.random(n) < label_bias)
    y_biased = np.where(flip_to_negative, 0, y)

    # Random noise on labels
    flip_noise = rng.random(n) < noise
    y_final = np.where(flip_noise, 1 - y_biased, y_biased)

    # Features matrix
    # Include a "proxy" interaction that often leaks group info: group*x1
    X = np.column_stack([x1, x2, x3, group, group * x1])
    cols = ["x1", "x2", "x3", "group", "group_x1_proxy"]
    df = pd.DataFrame(X, columns=cols)
    df["y"] = y_final
    df["group"] = group
    return df

rng = np.random.default_rng(seed)
df = generate_data(
    n=n_samples, p_g1=p_group1, feat_shift=feat_shift,
    label_bias=label_bias, noise=noise, rng=rng
)

# -------------------------------
# Train / Test split
# -------------------------------
X_cols = ["x1", "x2", "x3", "group", "group_x1_proxy"]
if remove_proxy:
    X_cols = [c for c in X_cols if c != "group_x1_proxy"]

X = df[X_cols].values
y = df["y"].value
