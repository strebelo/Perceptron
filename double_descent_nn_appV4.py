# double_descent_nn_app.py
# Run with: streamlit run double_descent_nn_app.py

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def seed_all(seed: int = 0):
    return np.random.default_rng(seed)

def target_function(x):
    # Smooth teacher with mixed frequencies
    return np.sin(2*np.pi*x) + 0.5*np.sin(6*np.pi*x)

def make_data(n, noise_std=0.1, rng=None):
    rng = seed_all(0) if rng is None else rng
    x = rng.uniform(-1, 1, size=(n, 1))
    y_clean = target_function(x)
    y = y_clean + rng.normal(0.0, noise_std, size=y_clean.shape)
    return x, y, y_clean

# ------------------------------------------------------------
# Random-features 'neural network' (freeze first layer; fit only readout)
# phi(x) = tanh(x*W + b),  W in R^{1 x H}, b in R^{H}
# Trainable params are readout weights beta in R^{H}
#
# We count parameters p = H (trainable) to align with the sweep.
# ------------------------------------------------------------
class RandomFeatureNet:
    def __init__(self, H: int, rng=None, scale=2.5):
        self.H = H
        self.rng = seed_all(0) if rng is None else rng
        self.W = self.rng.normal(0, scale, size=(1, H))  # frozen
        self.b = self.rng.normal(0, scale, size=(H,))    # frozen
        self.beta = np.zeros((H, 1))                     # trainable readout

    @property
    def num_params(self):
        # Only counting trainable readout weights
        return self.H

    def features(self, X):
        # X: (N,1) -> Phi: (N,H)
        Z = X @ self.W + self.b  # (N,H)
        Phi = np.tanh(Z)
        return Phi

    def fit_min_norm(self, X, Y, l2=0.0):
        # Minimum-norm (ridge when l2>0) solution for beta: beta = (Phi^T Phi + l2 I)^-1 Phi^T Y
        Phi = self.features(X)  # (N,H)
        if l2 == 0.0:
            # Pseudoinverse for the ridgeless solution
            self.beta = np.linalg.pinv(Phi) @ Y
        else:
            H = Phi.shape[1]
            A = Phi.T @ Phi + l2*np.eye(H)
            self.beta = np.linalg.solve(A, Phi.T @ Y)

    def predict(self, X):
        Phi = self.features(X)
        return Phi @ self.beta

# ------------------------------------------------------------
# Cached helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_dataset(n_train, n_test, noise_std, seed=0):
    rng = seed_all(seed)
    Xtr, Ytr, Ytr_clean = make_data(n_train, noise_std, rng)
    Xte, Yte, Yte_clean = make_data(n_test, noise_std, rng)
    return Xtr, Ytr, Ytr_clean, Xte, Yte, Yte_clean

@st.cache_data(show_spinner=False)
def sweep_models(H_list, Xtr, Ytr, Xte, Yte, l2, rf_scale):
    train_mse, test_mse, params = [], [], []
    for H in H_list:
        net = RandomFeatureNet(H, rng=seed_all(0), scale=rf_scale)
        net.fit_min_norm(Xtr, Ytr, l2=l2)
        Yhat_tr = net.predict(Xtr)
        Yhat_te = net.predict(Xte)
        train_mse.append(float(np.mean((Yhat_tr - Ytr)**2)))
        test_mse.append(float(np.mean((Yhat_te - Yte)**2)))
        params.append(net.num_params)
    return np.array(params), np.array(train_mse), np.array(test_mse)

@st.cache_data(show_spinner=False)
def fit_specific(H, Xtr, Ytr, Xgrid, l2, rf_scale):
    net = RandomFeatureNet(H, rng=seed_all(0), scale=rf_scale)
    net.fit_min_norm(Xtr, Ytr, l2=l2)
    Ygrid = net.predict(Xgrid)
    return Ygrid, net.num_params

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.set_page_config(page_title="Neural Double Descent (Random Features)", layout="wide", page_icon="📉")
st.title("📉 Neural Double Descent with Random Features (ridgeless readout)")

with st.sidebar:
    st.header("Data & Model")
    n_train = st.slider("Training points (n)", min_value=30, max_value=1000, value=200, step=10)
    n_test  = st.slider("Test points",         min_value=200, max_value=5000, value=1500, step=50)
    noise   = st.slider("Noise std (σ)",       min_value=0.0, max_value=0.6, value=0.20, step=0.01)
    max_p   = st.slider("Max parameters (p)",  min_value=10, max_value=8000, value=4000, step=10,
                        help="We sweep p = H (number of readout weights).")

    st.subheader("Regularization & Features")
    l2      = st.slider("Ridge (λ)", min_value=0.0, max_value=1e-1, value=0.0, step=1e-4,
                        help="Set λ=0 for classic ridgeless interpolation (stronger double descent).")
    rf_scale = st.slider("Random feature scale", min_value=0.5, max_value=4.0, value=2.5, step=0.1,
                        help="Scale of random first-layer weights/biases (affects feature richness).")

# Data
Xtr, Ytr, Ytr_clean, Xte, Yte, Yte_clean = generate_dataset(n_train, n_test, noise, seed=0)

# Grid for plotting true function
Xgrid = np.linspace(-1, 1, 800).reshape(-1, 1)
Ytrue = target_function(Xgrid)

# 1) True function
col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("True function")
    fig = plt.figure(figsize=(6,4))
    plt.plot(Xgrid, Ytrue, linewidth=2)
    plt.xlabel("x"); plt.ylabel("y")
    st.pyplot(fig)

# 2) Noisy data (as requested: a separate plot with data only)
with col2:
    st.subheader("Noisy data (train & test)")
    fig2 = plt.figure(figsize=(6,4))
    plt.scatter(Xtr, Ytr, s=12, alpha=0.7, label="train")
    plt.scatter(Xte, Yte, s=10, alpha=0.4, label="test")
    plt.legend(loc="best")
    plt.xlabel("x"); plt.ylabel("y")
    st.pyplot(fig2)

# Parameter sweep: densify near p≈n
p_star = n_train
p_max  = max_p
# Build a sweep that has linear spacing up to just below p_star and a geometric tail after
left = np.unique(np.clip(np.round(np.linspace(5, max(6, p_star-30), 30)).astype(int), 5, None))
right = np.unique(np.clip(np.round(np.geomspace(max(8, p_star+1), max(10, p_max), 35)).astype(int), 5, None))
p_list = np.unique(np.concatenate([left, [p_star], right]))
H_list = p_list  # here p = H

params, train_curve, test_curve = sweep_models(H_list, Xtr, Ytr, Xte, Yte, l2=l2, rf_scale=rf_scale)

st.subheader("Train/Test MSE vs number of parameters p  (here p = H)")
fig3 = plt.figure(figsize=(7,4))
plt.plot(params, train_curve, marker="o", label="Train MSE")
plt.plot(params, test_curve, marker="o", label="Test MSE")
plt.axvline(p_star, linestyle="--", label="p ~ n (interpolation)")
plt.xlabel("Parameters p")
plt.ylabel("MSE")
plt.legend(loc="best")
st.pyplot(fig3)

# Specific models: small, critical, large
p_small  = max(5, int(0.6*n_train))
p_mid    = n_train
p_large  = min(max_p, int(4.0*n_train))

Y_small,  p_s = fit_specific(p_small,  Xtr, Ytr, Xgrid, l2=l2, rf_scale=rf_scale)
Y_mid,    p_m = fit_specific(p_mid,    Xtr, Ytr, Xgrid, l2=l2, rf_scale=rf_scale)
Y_large,  p_l = fit_specific(p_large,  Xtr, Ytr, Xgrid, l2=l2, rf_scale=rf_scale)

colA, colB, colC = st.columns(3, gap="large")

def plot_fit(col, Xgrid, Ytrue, Xtr, Ytr, Yhat_grid, title):
    with col:
        st.markdown(title)
        fig = plt.figure(figsize=(5,4))
        plt.plot(Xgrid, Ytrue, linewidth=2, label="f(x)")
        plt.scatter(Xtr, Ytr, s=12, alpha=0.6, label="train")
        plt.plot(Xgrid, Yhat_grid, linewidth=2, label="fit")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(loc="best")
        st.pyplot(fig)

plot_fit(colA, Xgrid, Ytrue, Xtr, Ytr, Y_small,  f"**Small model**  (p≈{p_s})")
plot_fit(colB, Xgrid, Ytrue, Xtr, Ytr, Y_mid,    f"**Critical model** (p≈n≈{n_train}, p≈{p_m})")
plot_fit(colC, Xgrid, Ytrue, Xtr, Ytr, Y_large,  f"**Large model**  (p≈{p_l})")

st.markdown('''
**Notes for class:**

- This demo uses a 2-layer neural network with **random first-layer features** and a **ridgeless (λ=0) readout**. In this setting, the test MSE often **peaks near p≈n** and then **decreases again for large p**, producing a clear **double-descent** curve.
- If the peak is muted, try **higher noise**, a larger sweep (**max parameters**), or **λ=0**.
- The three panels show the qualitative difference between underparameterized, interpolation, and overparameterized regimes.
''')
