# double_descent_nn.py
# -------------------------------------------------------------
# Interactive demo of DOUBLE DESENT with a single-hidden-layer
# neural network (Extreme Learning Machine variant).
# -------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Page config
st.set_page_config(page_title="Double Descent (Neural Network)", layout="wide")
st.title("📉📈 Double Descent with a Neural Network (Single Hidden Layer)")

# Core helpers
def f_true(x: np.ndarray) -> np.ndarray:
    # Ground-truth smooth nonlinear function
    return np.sin(4 * x) + 0.5 * x**2

def make_data(rng: np.random.Generator, n_train: int, n_test: int, noise_std: float):
    x_tr = rng.uniform(-2.0, 2.0, size=n_train)
    y_tr = f_true(x_tr) + rng.normal(0, noise_std, size=n_train)
    x_te = rng.uniform(-2.0, 2.0, size=n_test)
    y_te = f_true(x_te) + rng.normal(0, noise_std, size=n_test)
    return x_tr, y_tr, x_te, y_te

def make_random_hidden(rng: np.random.Generator, p_max: int, scale: float):
    # Random hidden layer weights (1D input) and biases
    W = rng.normal(0.0, scale, size=(1, p_max))     # shape (1, p_max)
    b = rng.uniform(-1.0, 1.0, size=(p_max,))       # shape (p_max,)
    return W, b

def hidden_features(x: np.ndarray, p: int, W_all: np.ndarray, b_all: np.ndarray, activation: str = "tanh"):
    # Compute hidden activations for first p units + bias feature
    W = W_all[:, :p]     # (1, p)
    b = b_all[:p]        # (p,)
    z = np.outer(x, W.ravel()) + b  # (n, p)
    if activation == "tanh":
        h = np.tanh(z)
    else:  # ReLU
        h = np.maximum(0.0, z)
    # Add bias term as an additional feature (column of ones)
    return np.concatenate([h, np.ones((h.shape[0], 1))], axis=1)  # (n, p+1)

def fit_min_norm(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Minimum-norm least-squares: closed-form fit for output layer
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    seed = st.number_input("Seed", value=7, step=1)
    n_train = st.slider("Training points (n)", 20, 600, 80, step=5)
    n_test = st.slider("Test points", 100, 5000, 800, step=50)
    noise_std = st.slider("Noise std", 0.0, 1.0, 0.2, step=0.05)
    p_max = st.slider("Max hidden units (p_max)", 40, 2000, 600, step=20)
    p_step = st.slider("Sweep step for p", 1, 100, 10, step=1)
    freq_scale = st.slider("Hidden weight scale", 0.2, 5.0, 2.0, step=0.1)
    activation = st.selectbox("Activation", ["tanh", "ReLU"])
    selected_p = st.slider("Selected p (for the extra fit plot)", 1, 2000, 200, step=1)

selected_p = min(selected_p, p_max)  # clamp

# Data + random hidden layer
rng = np.random.default_rng(int(seed))
x_tr, y_tr, x_te, y_te = make_data(rng, n_train, n_test, noise_std)
W_all, b_all = make_random_hidden(rng, p_max, freq_scale)

# 1) Original function (no noise)
xs = np.linspace(-2.2, 2.2, 600)
fig1 = plt.figure(figsize=(7, 4.5))
plt.plot(xs, f_true(xs))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Original function (noiseless)")
plt.tight_layout()
st.pyplot(fig1)

# 2) Data (noised) + true function
fig2 = plt.figure(figsize=(7, 4.5))
plt.scatter(x_tr, y_tr, s=18, alpha=0.8, label="Train data")
plt.plot(xs, f_true(xs), linewidth=2, label="True function")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Data (noised) and true function")
plt.legend(loc="best")
plt.tight_layout()
st.pyplot(fig2)

# 3) Double Descent Curve: Train/Test MSE vs p
p_grid = np.arange(max(1, p_step), p_max + 1, p_step)
train_mse = []
test_mse = []

for p in p_grid:
    Htr = hidden_features(x_tr, p, W_all, b_all, activation=activation)
    Hte = hidden_features(x_te, p, W_all, b_all, activation=activation)
    w = fit_min_norm(Htr, y_tr)
    train_mse.append(mse(y_tr, Htr @ w))
    test_mse.append(mse(y_te, Hte @ w))

fig3 = plt.figure(figsize=(8, 5))
plt.plot(p_grid, train_mse, label="Train MSE")
plt.plot(p_grid, test_mse, label="Test MSE")
plt.axvline(n_train, linestyle="--", label="p = n (training points)")
plt.axvline(selected_p, linestyle=":", label=f"selected p = {selected_p}")
plt.xlabel("# of hidden units (model size p)")
plt.ylabel("MSE")
plt.title("Double Descent: Train/Test Error vs Model Size")
plt.legend(loc="best")
plt.tight_layout()
st.pyplot(fig3)

# 4) Fits at small, near p≈n, very large; plus a user-selected p
p_small = max(5, n_train // 6)
p_mid = min(max(1, n_train), p_max)  # p ≈ n (clamped to p_max)
p_large = p_max

def fit_and_plot(title: str, p_sel: int):
    Hs = hidden_features(xs, p_sel, W_all, b_all, activation=activation)
    Htr = hidden_features(x_tr, p_sel, W_all, b_all, activation=activation)
    w_sel = fit_min_norm(Htr, y_tr)
    yhat = Hs @ w_sel
    fig = plt.figure(figsize=(7, 4.5))
    plt.scatter(x_tr, y_tr, s=18, alpha=0.8, label="Train data")
    plt.plot(xs, f_true(xs), linewidth=2, label="True function")
    plt.plot(xs, yhat, linewidth=2, linestyle="--", label=f"NN fit (p = {p_sel})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    return fig

st.markdown("---")
st.subheader("Fits at three regimes")
c1, c2, c3 = st.columns(3)
with c1:
    st.caption(f"Small model (p = {p_small})")
    st.pyplot(fit_and_plot("Small model", p_small))
with c2:
    st.caption(f"Interpolation scale (p ≈ n = {n_train}; using p = {p_mid})")
    st.pyplot(fit_and_plot("Near interpolation (p ≈ n)", p_mid))
with c3:
    st.caption(f"Very large model (p = {p_large})")
    st.pyplot(fit_and_plot("Very large model", p_large))

st.markdown("—")
st.subheader("Fit at your selected p")
st.caption(f"Selected p = {selected_p}")
st.pyplot(fit_and_plot("User-selected model size", selected_p))

st.markdown(r"""
**Teaching tip:** Move the sliders so that the model size `p` crosses `n` (the number
of training points). You’ll usually see the **test error** peak near \(p \approx n\),
then **decrease** again for much larger \(p\). Because the output layer uses a
minimum-norm least-squares fit, this ELM setup makes the pattern fast and clear.
""")
