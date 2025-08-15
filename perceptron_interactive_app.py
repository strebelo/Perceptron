# perceptron_interactive_app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="Perceptron Demo", layout="centered")
st.title("Perceptron: Interactive Linear Classification Demo")

# ------------------------------
# Data generators
# ------------------------------
def generate_dataset_1(n, rng):
    """Roughly linearly separable: credit score vs debt-to-income."""
    credit_score = rng.random(n) * 200 + 500                      # 500–700
    debt_income = 100 - 0.1 * credit_score + rng.normal(0, 3, n)  # inverse + noise
    labels = np.ones(n, dtype=int)                                # +1 = paid
    labels[(credit_score < 600) & (debt_income > 40)] = -1         # -1 = default
    X = np.column_stack([credit_score, debt_income])
    return X, labels

def generate_dataset_2(n, rng, flip_frac=0.2):
    """Not linearly separable: start from dataset 1, flip a fraction of labels."""
    X, y = generate_dataset_1(n, rng)
    k = int(flip_frac * n)
    if k > 0:
        idx = rng.choice(n, size=k, replace=False)
        y[idx] *= -1
    return X, y

# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.header("Controls")
dataset_name = st.sidebar.selectbox("Dataset", ["Dataset 1 (separable-ish)", "Dataset 2 (noisy / not separable)"])
n_points = st.sidebar.slider("Number of points", 50, 800, 250, step=25)
seed = st.sidebar.number_input("Random seed", value=0, step=1)
lr = st.sidebar.slider("Learning rate", 0.001, 1.0, 0.1)
epochs = st.sidebar.slider("Epochs", 1, 300, 80)

flip_frac = 0.2
if "Dataset 2" in dataset_name:
    flip_frac = st.sidebar.slider("Label flip fraction (noise)", 0.0, 0.5, 0.2, step=0.05)

rng = np.random.default_rng(int(seed))

# ------------------------------
# Load data
# ------------------------------
if "Dataset 2" in dataset_name:
    X, y = generate_dataset_2(n_points, rng, flip_frac=flip_frac)
else:
    X, y = generate_dataset_1(n_points, rng)

# ------------------------------
# Initial plot (shown first)
# ------------------------------
fig, ax = plt.subplots()
ax.scatter(X[y == 1, 0], X[y == 1, 1], label="+1 (paid)", alpha=0.9)
ax.scatter(X[y == -1, 0], X[y == -1, 1], label="-1 (default)", alpha=0.9)
ax.set_xlabel("Credit score")
ax.set_ylabel("Debt-to-income")
ax.set_title("Data")
ax.legend()

plot_slot = st.empty()        # reserve a place on the page for the plot
plot_slot.pyplot(fig)         # show the data first

# ------------------------------
# Button AFTER the graph
# ------------------------------
run_clicked = st.button("Run perceptron", use_container_width=True)

# ------------------------------
# Perceptron training
# ------------------------------
def perceptron_train(X, y, lr=0.1, epochs=50, shuffle=True, rng=None):
    """
    Classic perceptron with bias term.
    Returns learned weights w (length 3): [bias, w1, w2]
    """
    Xb = np.c_[np.ones(len(X)), X]  # add bias column
    w = np.zeros(Xb.shape[1])
    idx = np.arange(len(Xb))

    for _ in range(epochs):
        if shuffle and rng is not None:
            rng.shuffle(idx)
        for i in idx:
            xi, yi = Xb[i], y[i]
            if yi * np.dot(w, xi) <= 0:  # misclassified or on boundary
                w = w + lr * yi * xi
    return w

def predict_signed(X, w):
    Xb = np.c_[np.ones(len(X)), X]
    return np.sign(Xb @ w)

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

# ------------------------------
# When clicked: train and update plot
# ------------------------------
if run_clicked:
    w = perceptron_train(X, y, lr=lr, epochs=epochs, shuffle=True, rng=rng)

    # Redraw with decision boundary
    fig2, ax2 = plt.subplots()
    ax2.scatter(X[y == 1, 0], X[y == 1, 1], label="+1 (paid)", alpha=0.9)
    ax2.scatter(X[y == -1, 0], X[y == -1, 1], label="-1 (default)", alpha=0.9)
    ax2.set_xlabel("Credit score")
    ax2.set_ylabel("Debt-to-income")
    ax2.set_title("Perceptron result")

    # Decision boundary: w0 + w1*x + w2*y = 0  -> y = -(w0 + w1*x)/w2
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 300)
    if abs(w[2]) > 1e-10:
        ys = -(w[0] + w[1] * xs) / w[2]
        ax2.plot(xs, ys, label="Decision boundary")
    else:
        # Vertical boundary if w2 ~ 0: x = -w0 / w1
        if abs(w[1]) > 1e-10:
            x_line = -w[0] / w[1]
            ax2.axvline(x_line, label="Decision boundary")

    ax2.legend()
    plot_slot.pyplot(fig2)  # update the same slot below the button press

    # Show metrics
    yhat = predict_signed(X, w)
    acc = accuracy(y, yhat)
    st.metric("Training accuracy", f"{acc*100:.1f}%")
    st.caption(
        "Note: On the noisy dataset, no single line can separate all points, "
        "so accuracy won’t reach 100%."
    )
