
# double_descent.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Double Descent Demo", layout="wide")
st.title("📉📈 Double Descent: Train/Test Error vs Model Size")

col_controls, col_plot = st.columns([1, 2], gap="large")

with col_controls:
    seed = st.number_input("Seed", value=7, step=1)
    n_train = st.slider("Training points (n)", 20, 200, 60, step=5)
    n_test = st.slider("Test points", 100, 1200, 400, step=50)
    noise_std = st.slider("Noise std", 0.0, 1.0, 0.2, step=0.05)
    max_features = st.slider("Max #features (p_max)", 40, 600, 240, step=10)
    step = st.slider("Sweep step for p", 1, 50, 5, step=1)
    freq_scale = st.slider("Feature frequency scale", 0.2, 5.0, 2.0, step=0.1)

rng = np.random.default_rng(int(seed))

def f_true(x):
    return np.sin(4 * x) + 0.5 * x**2

x_train = rng.uniform(-2.0, 2.0, size=n_train)
y_train = f_true(x_train) + rng.normal(0, noise_std, size=n_train)

x_test = rng.uniform(-2.0, 2.0, size=n_test)
y_test = f_true(x_test) + rng.normal(0, noise_std, size=n_test)

W_all = rng.normal(loc=0.0, scale=freq_scale, size=max_features)
b_all = rng.uniform(0.0, 2 * np.pi, size=max_features)

def rff_features(x, p):
    W = W_all[:p]
    b = b_all[:p]
    return np.cos(np.outer(x, W) + b)

def fit_min_norm(X, y):
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

p_grid = np.arange(max(step, 1), max_features + 1, step)

train_mse = []
test_mse = []

for p in p_grid:
    Xtr = rff_features(x_train, p)
    Xte = rff_features(x_test, p)
    w = fit_min_norm(Xtr, y_train)
    train_mse.append(mse(y_train, Xtr @ w))
    test_mse.append(mse(y_test, Xte @ w))

with col_plot:
    st.subheader("Train/Test MSE vs Model Size")
    fig = plt.figure(figsize=(8, 5))
    plt.plot(p_grid, train_mse, label="Train MSE")
    plt.plot(p_grid, test_mse, label="Test MSE")
    plt.axvline(n_train, linestyle="--", label="p = n (training points)")
    plt.xlabel("# of features (model size)")
    plt.ylabel("MSE")
    plt.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig)

st.markdown('''
---
### Fits in three regimes
''')

xs = np.linspace(-2.2, 2.2, 600)
reps = [max(5, n_train // 6), max(3, n_train - 1), max_features]
rep_names = ["Small model", "Near N (interpolation)", "Very large model"]

cols = st.columns(3)
for (name, p_sel), c in zip(zip(rep_names, reps), cols):
    Xs = rff_features(xs, p_sel)
    w_sel = fit_min_norm(rff_features(x_train, p_sel), y_train)
    ys_hat = Xs @ w_sel

    with c:
        st.caption(f"{name} (p = {p_sel})")
        fig2 = plt.figure(figsize=(5, 4))
        plt.scatter(x_train, y_train, s=18, alpha=0.8, label="Train data")
        plt.plot(xs, f_true(xs), linewidth=2, label="True function")
        plt.plot(xs, ys_hat, linewidth=2, linestyle="--", label="Fit")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="best")
        plt.tight_layout()
        st.pyplot(fig2)

st.markdown(r'''
**Teaching tip:** Move the sliders so that the training size `n` is crossed by the
model size `p`. You'll typically see test error spike near \(p\approx n\), then
decline again for \(p \gg n\) — the hallmark of *double descent*.
''')
