# ai_bias_demo.py
# Streamlit App: Explaining AI Bias with an Interactive Demo
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
This app shows **how bias can arise** in AI systems and how different **mitigations** affect outcomes.

**How to use**  
1) Tweak the **data** (group share, feature shift, label bias, noise).  
2) Train a model and view **metrics by group**.  
3) Try **mitigations**: remove proxy features, re-weight training samples, or set **group-specific thresholds**.
"""
)

# -------------------------------
# Controls
# -------------------------------
with st.sidebar:
    st.header("Data generation")
    seed = st.number_input("Random seed", 0, 10000, value=123, step=1)
    n_samples = st.slider("Dataset size", 500, 20000, value=2000, step=500)

    st.markdown("**Group proportions**")
    p_group1 = st.slider("Share of Group B (protected=1)", 0.05, 0.95, value=0.5, step=0.05)

    st.markdown("**Feature disparity** (mean shift for Group B on key feature)")
    feat_shift = st.slider("Feature shift (Group B vs A)", -2.0, 2.0, value=-0.5, step=0.1)

    st.markdown("**Label bias** (historical discrimination on outcomes)")
    label_bias = st.slider("Extra negative labels for Group B (prob.)", 0.0, 0.5, value=0.2, step=0.05)

    st.markdown("**Noise**")
    noise = st.slider("Label noise (both groups)", 0.0, 0.3, value=0.05, step=0.01)

    st.markdown("---")
    st.header("Model & Mitigations")
    model_choice = st.selectbox("Model", ["Logistic Regression"])
    remove_proxy = st.checkbox("Remove proxy feature (drop group*x1)", value=False)
    use_reweight = st.checkbox("Re-weight training samples (counter label bias)", value=False)

    st.markdown("**Decision thresholds**")
    global_thr = st.slider("Global threshold", 0.0, 1.0, value=0.5, step=0.01)
    group_specific_thresholds = st.checkbox("Use group-specific thresholds", value=False)
    thr_g0 = st.slider("Threshold for Group A (0)", 0.0, 1.0, value=0.5, step=0.01)
    thr_g1 = st.slider("Threshold for Group B (1)", 0.0, 1.0, value=0.5, step=0.01)

    st.caption("Tip: start with modest bias, compare metrics, then try mitigations.")

# -------------------------------
# Synthetic data generator
# -------------------------------
def generate_data(n=2000, p_g1=0.5, feat_shift=-0.5, label_bias=0.2, noise=0.05, rng=None):
    """
    Binary classification with protected attribute 'group' ∈ {0 (A), 1 (B)}.
    Group B can have:
      - feature shift (disparity)
      - extra flips to negative labels (historical bias)
    """
    if rng is None:
        rng = np.random.default_rng(123)

    group = (rng.random(n) < p_g1).astype(int)  # 1 = Group B (protected)

    # Features
    x1 = rng.normal(loc=0.0 + feat_shift*group, scale=1.0, size=n)  # key feature with disparity
    x2 = rng.normal(0, 1, size=n)
    x3 = 0.5 * x1 + rng.normal(0, 1, size=n)  # proxy-ish (correlated with x1 & group)

    # Latent linear score → probability
    lin = 1.2*x1 + 0.8*x2 + 0.6*x3 - 0.2
    p = 1 / (1 + np.exp(-lin))

    # True labels before bias/noise
    y = (rng.random(n) < p).astype(int)

    # Historical label bias against Group B: extra flips to 0
    flip_to_negative = (group == 1) & (rng.random(n) < label_bias)
    y_biased = np.where(flip_to_negative, 0, y)

    # Label noise (both groups)
    flip_noise = rng.random(n) < noise
    y_final = np.where(flip_noise, 1 - y_biased, y_biased)

    # Feature matrix with a proxy interaction
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
y = df["y"].values        # <-- FIX: use .values (plural), not .value
g = df["group"].values    # <-- FIX: use .values

X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X, y, g, test_size=0.35, random_state=seed, stratify=y
)

# -------------------------------
# Re-weighting (simple counter-bias)
# -------------------------------
sample_weight = None
if use_reweight:
    base_w = np.ones_like(y_train, dtype=float)
    # Up-weight positives in Group B (which suffered historical negative flips)
    idx = (g_train == 1) & (y_train == 1)
    base_w[idx] = base_w[idx] * (1.0 + 3.0*label_bias)  # heuristic factor
    sample_weight = base_w

# -------------------------------
# Train model
# -------------------------------
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=200, solver="lbfgs")
else:
    model = LogisticRegression(max_iter=200, solver="lbfgs")

model.fit(X_train, y_train, sample_weight=sample_weight)

# Scores / predictions
proba = model.predict_proba(X_test)[:, 1]

# -------------------------------
# Threshold(s) & predictions
# -------------------------------
def apply_thresholds(scores, groups, thr, thr0=None, thr1=None, use_group=False):
    if not use_group:
        return (scores >= thr).astype(int)
    t0 = thr0 if thr0 is not None else thr
    t1 = thr1 if thr1 is not None else thr
    preds = np.zeros_like(scores, dtype=int)
    preds[(groups == 0) & (scores >= t0)] = 1
    preds[(groups == 1) & (scores >= t1)] = 1
    return preds

y_pred = apply_thresholds(
    proba, g_test, global_thr,
    thr0=thr_g0, thr1=thr_g1, use_group=group_specific_thresholds
)

# -------------------------------
# Metrics by group
# -------------------------------
def metrics_by_group(y_true, y_score, y_pred, groups):
    rows = []
    for grp in [0, 1]:
        idx = (groups == grp)
        yt, yp, ys = y_true[idx], y_pred[idx], y_score[idx]
        label = "A (0)" if grp == 0 else "B (1)"
        n = int(idx.sum())
        acc = accuracy_score(yt, yp) if n > 0 else np.nan
        prec = precision_score(yt, yp, zero_division=0) if n > 0 else np.nan
        rec = recall_score(yt, yp, zero_division=0) if n > 0 else np.nan
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        sel = yp.mean() if n > 0 else np.nan
        auc = roc_auc_score(yt, ys) if len(np.unique(yt)) == 2 else np.nan
        rows.append({
            "Group": label, "n": n, "Accuracy": acc, "Precision": prec,
            "Recall (TPR)": rec, "FPR": fpr, "Selection Rate (P(ŷ=1))": sel, "AUC": auc
        })
    return pd.DataFrame(rows)

metrics_df = metrics_by_group(y_test, proba, y_pred, g_test)

def gap(a, b):
    if np.isnan(a) or np.isnan(b): return np.nan
    return a - b

gA = metrics_df.iloc[0]
gB = metrics_df.iloc[1]
dp_gap = gap(gA["Selection Rate (P(ŷ=1))"], gB["Selection Rate (P(ŷ=1))"])  # Demographic parity gap
eo_gap = gap(gA["Recall (TPR)"], gB["Recall (TPR)"])                        # Equal opportunity gap

# -------------------------------
# Layout: top metrics & confusion matrices
# -------------------------------
top_left, top_right = st.columns([1.1, 1.0])

with top_left:
    st.subheader("Performance & Fairness (by group)")
    st.dataframe(metrics_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1: st.metric("Demographic Parity Gap (SelRate A − B)", f"{dp_gap:.3f}")
    with c2: st.metric("Equal Opportunity Gap (TPR A − B)", f"{eo_gap:.3f}")

with top_right:
    st.subheader("Confusion Matrices")

    def plot_cm(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(cm, interpolation='nearest')
        ax.set_title(title)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['True 0', 'True 1'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    idx0 = (g_test == 0)
    idx1 = (g_test == 1)
    plot_cm(y_test[idx0], y_pred[idx0], "Group A (0)")
    plot_cm(y_test[idx1], y_pred[idx1], "Group B (1)")

# -------------------------------
# Plots row: distributions & ROC
# -------------------------------
st.subheader("Score Distributions and ROC Curves")

p1, p2 = st.columns(2)

with p1:
    fig1, ax1 = plt.subplots(figsize=(5.5, 3.5))
    scores0 = proba[g_test == 0]
    scores1 = proba[g_test == 1]
    ax1.hist(scores0, bins=30, alpha=0.6, label="Group A (0)")
    ax1.hist(scores1, bins=30, alpha=0.6, label="Group B (1)")
    ax1.set_title("Predicted score distributions")
    ax1.set_xlabel("Score (P(ŷ=1))")
    ax1.set_ylabel("Count")
    ax1.legend()
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)

with p2:
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
    for grp, name in [(0, "A (0)"), (1, "B (1)")]:
        mask = (g_test == grp)
        yt, ys = y_test[mask], proba[mask]
        if len(np.unique(yt)) == 2:
            fpr, tpr, _ = roc_curve(yt, ys)
            auc = roc_auc_score(yt, ys)
            ax2.plot(fpr, tpr, label=f"Group {name} (AUC={auc:.3f})")
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.set_title("ROC by group")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

# -------------------------------
# Explanations
# -------------------------------
st.subheader("What to notice")
st.markdown(
    f"""
- **Feature disparity** makes one group systematically score lower/higher → affects **selection rate**.  
- **Label bias** (historical discrimination) shifts training labels toward 0 for Group B → model learns biased patterns.  
- **Demographic Parity Gap**: difference in P(ŷ=1) across groups (**{dp_gap:.3f}** here).  
- **Equal Opportunity Gap**: difference in **TPR** across groups (**{eo_gap:.3f}** here).  
- **Mitigations** can reduce gaps but may trade off overall accuracy or other metrics.
"""
)

st.subheader("Mitigation options in this demo")
st.markdown(
    """
- **Remove proxy feature**: drops `group_x1_proxy` that leaks group info.  
- **Re-weight training samples**: compensates for label bias by up-weighting positive Group B examples.  
- **Group-specific thresholds**: post-processing to equalize outcomes (e.g., TPR or selection rate) across groups.  
"""
)

with st.expander("Technical notes (click to expand)"):
    st.markdown(
        """
- **Data**: synthetic, two groups (A/B), configurable feature shift and label bias.  
- **Model**: Logistic Regression (sklearn).  
- **Fairness metrics**: Demographic parity (selection rate gap) and Equal opportunity (TPR gap).  
- **Caveats**: Real systems need broader governance (data audits, drift monitoring, human oversight, compliance).
"""
    )

st.caption("Built for executives: adjust sliders, observe gaps, then try mitigations.")
