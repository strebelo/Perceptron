# perceptron_credit_streamlit.py
# Streamlit demo: Credit-style dataset + Perceptron + user-drawn separating line
# Author: Sergio + ChatGPT
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="Perceptron (Credit) Demo", layout="centered")

st.title("Perceptron Demo on a Credit-Style Dataset")
st.caption(
    "Play with a synthetic credit dataset (credit score vs debt-to-income). "
    "Draw your own separating line, then train a perceptron (with optional Pocket) and compare!"
)

# ------------------------------
# Data generators
# ------------------------------
def generate_dataset_1(n, rng):
    """
    Dataset 1 (credit-style, linearly separable-ish):
      - x0: credit score in [500, 800]
      - x1: debt-to-income ratio (DTI) inversely related to credit score + noise
    Label rule (ground truth): y = +1 if DTI < line(credit), else -1
    """
    credit = rng.uniform(500, 800, size=n)
    # True boundary for labeling (unknown to the perceptron):
    true_slope = -0.08   # DTI decreases as credit increases
    true_intercept = 60  # base DTI when credit ~ 0 (just for synthetic rule)
    # Generate DTI with noise
    dti = true_slope * credit + true_intercept + rng.normal(0, 4.0, size=n)
    X = np.column_stack([credit, dti])

    # Labels based on ground-truth boundary
    y = np.where(dti < true_slope * credit + true_intercept, 1, -1)
    return X, y

def generate_dataset_2(n, rng):
    """
    Dataset 2 (messier, not perfectly separable):
      Same variables but more noise + a few flipped labels.
    """
    X, y = generate_dataset_1(n, rng)
    # Add more noise to make it harder
    X[:, 1] += rng.normal(0, 8.0, size=n)
    # Flip 10% labels at random to create label noise
    flips = rng.choice(n, size=max(1, n // 10), replace=False)
    y[flips] *= -1
    return X, y

# ------------------------------
# Helpers
# ------------------------------
def standardize_features(X):
    """Return standardized features and (mean, std) for inverse transforms if needed."""
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    Xs = (X - mu) / std
    return Xs, mu, std

def add_bias(X):
    """Add bias term as the last column (1.0)."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([X, ones])

def predict_raw(Xb, w):
    """Compute w^T x for each row (raw score)."""
    return Xb @ w

def predict_label(Xb, w):
    """Return labels in {-1, +1}."""
    raw = predict_raw(Xb, w)
    return np.where(raw >= 0, 1, -1)

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def perceptron_train(X, y, lr=0.1, epochs=50, pocket=True, rng=None):
    """
    Basic perceptron with optional Pocket algorithm.
    X: (n,2) features (unstandardized). We will standardize inside.
    y: labels in {-1,+1}
    Returns: weights in standardized feature space (w0,w1,b), plus info dict.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Standardize for stability
    Xs, mu, std = standardize_features(X)
    Xb = add_bias(Xs)  # shape (n, 3)
    n, d = Xb.shape
    w = np.zeros(d)

    best_w = w.copy()
    best_acc = 0.0

    history = []

    for ep in range(epochs):
        # Shuffle indices
        idx = rng.permutation(n)
        errors = 0
        for i in idx:
            xi = Xb[i]
            yi = y[i]
            if yi * (w @ xi) <= 0:  # misclassified
                w = w + lr * yi * xi
                errors += 1

        y_pred = predict_label(Xb, w)
        acc = accuracy(y, y_pred)
        history.append((ep + 1, errors, acc))

        if pocket:
            if acc > best_acc:
                best_acc = acc
                best_w = w.copy()
        else:
            best_w = w.copy()
            best_acc = acc

        # Early exit if perfect
        if best_acc == 1.0:
            break

    info = {
        "history": history,
        "standardize_mu": mu,
        "standardize_std": std,
    }
    return best_w, info

def line_from_weights(w):
    """
    Convert perceptron weights (for standardized features with bias) to a plotting function
    in standardized coordinates: w0 * x + w1 * y + b = 0  ->  y = -(w0/w1) x - b/w1
    Returns a function f(x_std) -> y_std, or None if vertical line (w1 ~ 0).
    """
    w0, w1, b = w
    if np.isclose(w1, 0.0):
        return None
    slope = -w0 / w1
    intercept = -b / w1

    def f(x_std):
        return slope * x_std + intercept

    return f

def classify_with_line(X, slope, intercept):
    """
    Classify each point in X based on position relative to y = slope*x + intercept
    Returns labels in {+1, -1}: +1 if below the line (i.e., y < slope*x + intercept), else -1
    (You can flip this if you prefer a different convention.)
    """
    return np.where(X[:, 1] < slope * X[:, 0] + intercept, 1, -1)

# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.header("Controls")

dataset_name = st.sidebar.selectbox(
    "Dataset",
    ["Dataset 1 (cleaner)", "Dataset 2 (noisier)"],
    index=0,
)

n_points = st.sidebar.slider("Number of points", min_value=40, max_value=500, value=200, step=10)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

lr = st.sidebar.slider("Learning rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
epochs = st.sidebar.slider("Epochs", min_value=5, max_value=500, value=100, step=5)
use_pocket = st.sidebar.checkbox("Use Pocket algorithm (keep best-so-far)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Your line")
user_slope = st.sidebar.slider("Slope (m)", min_value=-1.0, max_value=1.0, value=-0.1, step=0.01)
user_intercept = st.sidebar.slider("Intercept (b)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)

# ------------------------------
# Generate data
# ------------------------------
rng = np.random.default_rng(seed)
if dataset_name.startswith("Dataset 1"):
    X, y = generate_dataset_1(n_points, rng)
else:
    X, y = generate_dataset_2(n_points, rng)

# ------------------------------
# User line predictions + accuracy
# ------------------------------
user_pred = classify_with_line(X, user_slope, user_intercept)
user_acc = accuracy(y, user_pred)

# ------------------------------
# Train perceptron
# ------------------------------
if st.button("Run Perceptron"):
    w_std, info = perceptron_train(X, y, lr=lr, epochs=epochs, pocket=use_pocket, rng=rng)
    st.success(f"Perceptron trained. Last/Best accuracy (standardized space): "
               f"{accuracy(y, predict_label(add_bias(standardize_features(X)[0]), w_std)):.3f}")

    # Plotting with both your line (original feature space) and perceptron line (in standardized space)
    fig, ax = plt.subplots(figsize=(7, 5))
    # Scatter points, colored by true label
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label="+1 (Approved)", alpha=0.8, marker="o")
    ax.scatter(X[y == -1, 0], X[y == -1, 1], label="-1 (Rejected)", alpha=0.8, marker="x")

    # Plot user line in original coordinates
    xs = np.linspace(X[:, 0].min() - 5, X[:, 0].max() + 5, 200)
    ys_user = user_slope * xs + user_intercept
    ax.plot(xs, ys_user, linestyle="--", label=f"Your line: y = {user_slope:.2f}x + {user_intercept:.1f}")

    # Plot perceptron line: compute in standardized coords, then transform back
    Xs, mu, std = standardize_features(X)
    f_std = line_from_weights(w_std)
    if f_std is not None:
        # Make a grid in original x, convert to standardized x, compute standardized y, then map back
        xs_orig = xs
        xs_std = (xs_orig - mu[0, 0]) / std[0, 0]
        ys_std = f_std(xs_std)
        ys_orig = ys_std * std[0, 1] + mu[0, 1]
        ax.plot(xs_orig, ys_orig, label="Perceptron boundary", linewidth=2)

    ax.set_xlabel("Credit score")
    ax.set_ylabel("Debt-to-income (DTI)")
    ax.set_title("Perceptron vs. Your Line")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Metrics
    st.subheader("Metrics")
    # Perceptron accuracy in current (original) space computed by applying standardized model
    Xb_std = add_bias(standardize_features(X)[0])
    perc_pred = predict_label(Xb_std, w_std)
    perc_acc = accuracy(y, perc_pred)

    st.write(
        f"- **Your line accuracy**: `{user_acc:.3f}`  "
        f"\n- **Perceptron accuracy**: `{perc_acc:.3f}`  "
        f"\n- **Pocket**: `{use_pocket}`"
    )

    # Training history
    if info.get("history"):
        st.subheader("Training History")
        hist = np.array(info["history"], dtype=object)
        ep = hist[:, 0].astype(int)
        errs = hist[:, 1].astype(int)
        accs = hist[:, 2].astype(float)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(ep, accs, linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy over epochs")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(ep, errs, linewidth=2)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Number of updates (errors)")
        ax3.set_title("Updates per epoch")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

else:
    # Draw only your line + data until user trains
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label="+1 (Approved)", alpha=0.8, marker="o")
    ax.scatter(X[y == -1, 0], X[y == -1, 1], label="-1 (Rejected)", alpha=0.8, marker="x")

    xs = np.linspace(X[:, 0].min() - 5, X[:, 0].max() + 5, 200)
    ys_user = user_slope * xs + user_intercept
    ax.plot(xs, ys_user, linestyle="--", label=f"Your line: y = {user_slope:.2f}x + {user_intercept:.1f}")

    ax.set_xlabel("Credit score")
    ax.set_ylabel("Debt-to-income (DTI)")
    ax.set_title(f"{dataset_name} — Your line accuracy: {user_acc:.3f}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.info(
        "Adjust the slope/intercept to draw a line. "
        "Then click **Run Perceptron** to train and compare boundaries."
    )
