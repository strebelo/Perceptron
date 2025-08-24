
# double_descent_nn_appV2.py
# Run with: streamlit run double_descent_nn_app.py

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def seed_all(seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng

def target_function(x):
    # Non-trivial smooth teacher function with mixed frequencies
    # Encourages interpolation effects with noise
    return np.sin(2*np.pi*x) + 0.5*np.sin(6*np.pi*x)

def make_data(n, noise_std=0.1, rng=None):
    if rng is None:
        rng = seed_all(0)
    x = rng.uniform(-1, 1, size=(n, 1))
    y_clean = target_function(x)
    y = y_clean + rng.normal(0.0, noise_std, size=y_clean.shape)
    return x, y, y_clean

# ------------------------------------------------------------
# 1-hidden-layer tanh network with NumPy
# params: p = 3H + 1  (for 1D input, 1D output)
# ------------------------------------------------------------
class TanhNet1HL:
    def __init__(self, H, rng=None, scale=0.5):
        self.H = H
        self.rng = seed_all(0) if rng is None else rng
        # Weights
        self.W1 = self.rng.normal(0, scale, size=(1, H))   # input->hidden
        self.b1 = np.zeros((1, H))
        self.W2 = self.rng.normal(0, scale, size=(H, 1))   # hidden->out
        self.b2 = np.zeros((1, 1))

    @property
    def num_params(self):
        # (1*H + H) + (H*1 + 1) = 3H + 1
        return 3*self.H + 1

    def forward(self, X):
        # X: (N,1)
        Z1 = X @ self.W1 + self.b1    # (N,H)
        A1 = np.tanh(Z1)              # (N,H)
        out = A1 @ self.W2 + self.b2  # (N,1)
        cache = (X, Z1, A1)
        return out, cache

    def mse(self, Yhat, Y):
        return float(np.mean((Yhat - Y)**2))

    def train(self, X, Y, lr=0.05, epochs=3000, l2=0.0, verbose=False):
        N = X.shape[0]
        for t in range(epochs):
            # forward
            Yhat, (Xc, Z1, A1) = self.forward(X)
            # loss
            err = (Yhat - Y)  # (N,1)
            loss = (err**2).mean() + l2*(np.sum(self.W1**2)+np.sum(self.W2**2))/N

            # backward
            dY = 2*err / N  # (N,1)
            dW2 = A1.T @ dY + (2*l2/N)*self.W2
            db2 = np.sum(dY, axis=0, keepdims=True)

            dA1 = dY @ self.W2.T  # (N,H)
            dZ1 = dA1 * (1 - np.tanh(Z1)**2)  # tanh'
            dW1 = Xc.T @ dZ1 + (2*l2/N)*self.W1
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # update
            self.W1 -= lr*dW1
            self.b1 -= lr*db1
            self.W2 -= lr*dW2
            self.b2 -= lr*db2

            # tiny learning-rate decay helps stability
            if (t+1) % 1000 == 0:
                lr *= 0.8

            if verbose and (t+1) % 500 == 0:
                print(f"Epoch {t+1}: loss={loss:.6f}")

        # return final training loss
        return loss

    def predict(self, X):
        Yhat, _ = self.forward(X)
        return Yhat

# ------------------------------------------------------------
# Cached helpers (speed up sweeping H)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_dataset(n_train, n_test, noise_std, seed=0):
    rng = seed_all(seed)
    Xtr, Ytr, Ytr_clean = make_data(n_train, noise_std, rng)
    Xte, Yte, Yte_clean = make_data(n_test, noise_std, rng)
    return Xtr, Ytr, Ytr_clean, Xte, Yte, Yte_clean

@st.cache_data(show_spinner=False)
def sweep_models(H_list, Xtr, Ytr, Xte, Yte, lr, epochs, l2):
    train_mse = []
    test_mse  = []
    params    = []
    for H in H_list:
        net = TanhNet1HL(H, rng=seed_all(0))
        net.train(Xtr, Ytr, lr=lr, epochs=epochs, l2=l2, verbose=False)
        Yhat_tr = net.predict(Xtr)
        Yhat_te = net.predict(Xte)
        train_mse.append(float(np.mean((Yhat_tr - Ytr)**2)))
        test_mse.append(float(np.mean((Yhat_te - Yte)**2)))
        params.append(net.num_params)
    return np.array(params), np.array(train_mse), np.array(test_mse)

@st.cache_data(show_spinner=False)
def fit_specific(H, Xtr, Ytr, Xgrid, lr, epochs, l2):
    net = TanhNet1HL(H, rng=seed_all(0))
    net.train(Xtr, Ytr, lr=lr, epochs=epochs, l2=l2, verbose=False)
    Ygrid = net.predict(Xgrid)
    return Ygrid, net.num_params

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Neural-Network Double Descent (NumPy)", layout="wide", page_icon="📉")

st.title("📉 Neural-Network Double Descent (NumPy, 1-hidden-layer tanh)")

with st.sidebar:
    st.header("Controls")
    n_train = st.slider("Training points (n)", min_value=30, max_value=600, value=150, step=10)
    n_test  = st.slider("Test points",         min_value=100, max_value=2000, value=800, step=50)
    noise   = st.slider("Noise std (σ)",       min_value=0.0, max_value=0.5, value=0.15, step=0.01)
    max_p   = st.slider("Max parameters (p)",  min_value=10, max_value=4000, value=1200, step=10,
                        help="Upper bound for parameter-sweep curve. p = 3H+1.")

    st.subheader("Training hyperparameters")
    lr     = st.slider("Learning rate", min_value=0.001, max_value=0.2, value=0.05, step=0.001)
    epochs = st.slider("Epochs",        min_value=200, max_value=6000, value=2500, step=100)
    l2     = st.slider("L2 weight decay", min_value=0.0, max_value=1e-2, value=0.0, step=0.0001)

# Data
Xtr, Ytr, Ytr_clean, Xte, Yte, Yte_clean = generate_dataset(n_train, n_test, noise, seed=0)

# Grid for plotting true function and fitted curves
Xgrid = np.linspace(-1, 1, 800).reshape(-1, 1)
Ytrue = target_function(Xgrid)

# Show the data and the true function
col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("True function vs. noisy data")
    fig = plt.figure(figsize=(6,4))
    plt.plot(Xgrid, Ytrue, label="f(x) (true)", linewidth=2)
    plt.scatter(Xtr, Ytr, s=12, alpha=0.6, label="train (noisy)")
    plt.scatter(Xte, Yte, s=10, alpha=0.4, label="test (noisy)")
    plt.legend(loc="best")
    plt.xlabel("x")
    plt.ylabel("y")
    st.pyplot(fig)

with col2:
    st.subheader("Clean vs noisy distributions")
    fig2 = plt.figure(figsize=(6,4))
    plt.plot(Xgrid, Ytrue, label="f(x) (true)", linewidth=2)
    plt.scatter(Xtr, Ytr_clean, s=12, alpha=0.6, label="train (clean)")
    plt.scatter(Xtr, Ytr, s=12, alpha=0.6, label="train (noisy)")
    plt.legend(loc="best")
    plt.xlabel("x")
    plt.ylabel("y")
    st.pyplot(fig2)

# Parameter sweep to show double descent
# Translate parameter bound p_max to a hidden width range H
H_max = max(1, (max_p - 1)//3)
# Create a non-linear sweep densifying around p≈n
H_star = max(1, (n_train - 1)//3)  # H at which p≈n
Hs_left  = np.unique(np.clip(np.round(np.linspace(1, max(2, H_star-20), 25)).astype(int), 1, None))
Hs_right = np.unique(np.clip(np.round(np.geomspace(max(2, H_star+1), max(3, H_max), 25)).astype(int), 1, None))
H_list = np.unique(np.concatenate([Hs_left, [H_star], Hs_right]))

params, train_curve, test_curve = sweep_models(H_list, Xtr, Ytr, Xte, Yte, lr=lr, epochs=epochs, l2=l2)

st.subheader("Train/Test MSE vs. number of parameters p (p = 3H+1)")
fig3 = plt.figure(figsize=(7,4))
plt.plot(params, train_curve, marker="o", label="Train MSE")
plt.plot(params, test_curve, marker="o", label="Test MSE")
plt.axvline(n_train, linestyle="--", label="p ~ n (interpolation threshold)")
plt.xlabel("Parameters p")
plt.ylabel("MSE")
plt.legend(loc="best")
st.pyplot(fig3)

# Specific model fits: small, critical, large
p_small   = max(7, int(0.6*n_train))
p_middle  = n_train
p_large   = min(max_p, int(4.0*n_train))

def p_to_H(p):  # invert p = 3H+1
    H = max(1, (p - 1)//3)
    return H

H_small  = p_to_H(p_small)
H_middle = p_to_H(p_middle)
H_large  = p_to_H(p_large)

Y_small,  p_s = fit_specific(H_small,  Xtr, Ytr, Xgrid, lr=lr, epochs=epochs, l2=l2)
Y_mid,    p_m = fit_specific(H_middle, Xtr, Ytr, Xgrid, lr=lr, epochs=epochs, l2=l2)
Y_large,  p_l = fit_specific(H_large,  Xtr, Ytr, Xgrid, lr=lr, epochs=epochs, l2=l2)

colA, colB, colC = st.columns(3, gap="large")

def plot_fit(col, Xgrid, Ytrue, Xtr, Ytr, Yhat_grid, title):
    with col:
        st.markdown(title)
        fig = plt.figure(figsize=(5,4))
        plt.plot(Xgrid, Ytrue, linewidth=2, label="f(x) (true)")
        plt.scatter(Xtr, Ytr, s=12, alpha=0.6, label="train (noisy)")
        plt.plot(Xgrid, Yhat_grid, linewidth=2, label="model fit")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(loc="best")
        st.pyplot(fig)

plot_fit(colA, Xgrid, Ytrue, Xtr, Ytr, Y_small,  f"**Small model** (p≈{p_s})")
plot_fit(colB, Xgrid, Ytrue, Xtr, Ytr, Y_mid,    f"**Critical model** (p≈n≈{n_train}, p≈{p_m})")
plot_fit(colC, Xgrid, Ytrue, Xtr, Ytr, Y_large,  f"**Large model** (p≈{p_l})")

# Helpful text
st.markdown('''
**How to use this demo**

- Use the sliders to change the sample sizes, noise level, and the maximum parameter count used in the sweep.
- The vertical dashed line in the MSE plot marks the interpolation threshold **p≈n**.
- In many settings, you'll see the classic **double descent** shape: test MSE falls (underparameterized regime), peaks near **p≈n** (interpolation), then falls again as the model becomes highly overparameterized.
- If you don't see it immediately, try:
  - Increase **n** and set a **moderate noise** (e.g., σ≈0.1–0.2).
  - Train a bit longer (more epochs) and set **weight decay to 0**.
  - Increase **max parameters** so the sweep includes very overparameterized models.
''')
