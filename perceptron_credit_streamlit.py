# perceptron_credit_streamlit.py 
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Perceptron: Credit Example", layout="centered")
st.title("Perceptron: Credit Approval Demo")

st.markdown(
    """
This demo shows how a **perceptron** learns a decision boundary for a simple credit example:

- **x-axis:** Credit score  
- **y-axis:** Debt-to-income ratio  
- **Labels:** +1 (paid) and -1 (default)  
- The perceptron **nudges weights** only when it finds a misclassified point.
"""
)

# -----------------------------------------------------------------------------
# Session state (for storing perceptron results and histories across reruns)
# -----------------------------------------------------------------------------
if "perc_line" not in st.session_state:       # (slope, intercept) or None
    st.session_state.perc_line = None
if "perc_acc" not in st.session_state:
    st.session_state.perc_acc = None
if "histories" not in st.session_state:       # cache keyed by (dataset, n, seed, lr, epochs, flip)
    st.session_state.histories = {}

# -----------------------------------------------------------------------------
# Data generators: same structure as your previous program
# -----------------------------------------------------------------------------
def generate_dataset_1(n, rng):
    # Credit score between 500 and 700
    credit_score = rng.random(n) * 200 + 500
    # Debt/Income inversely related to credit score + noise
    debt_income = 100 - 0.1 * credit_score + rng.normal(0, 3, size=n)
    # Labels: +1 "paid", -1 "default"
    labels = np.ones(n, dtype=int)
    labels[(credit_score < 600) & (debt_income > 40)] = -1
    X = np.column_stack([credit_score, debt_income])
    return X, labels

def generate_dataset_2(n, rng, flip_frac=0.2):
    X, y = generate_dataset_1(n, rng)
    n_flip = int(flip_frac * n)
    if n_flip > 0:
        idx = rng.choice(n, size=n_flip, replace=False)
        y[idx] = -y[idx]
    return X, y

# -----------------------------------------------------------------------------
# Perceptron utilities
# -----------------------------------------------------------------------------
def line_from_weights(w):
    # w1*x + w2*y + b = 0  => y = -(w1/w2) x - b/w2
    w1, w2, b = w
    if np.isclose(w2, 0.0):
        return None, None
    return -w1 / w2, -b / w2

def classify_with_line(X, slope, intercept):
