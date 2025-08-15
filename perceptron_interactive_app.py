
# perceptron_interactive_app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Perceptron Demo", layout="centered")

st.title("Perceptron: Interactive Linear Classification Demo")

#st.markdown(
#    """
#This app illustrates how the **perceptron** finds a separating line in 2D.  
#- **Dataset 1** is approximately linearly separable (like the MATLAB example with credit score vs. debt-to-income).  
#- **Dataset 2** is **not** linearly separable (we inject label noise so no single line gets everything right).  
#Use the sliders to pick your own **slope** and **intercept**, then click **Run perceptron** to see the algorithm's line.
#"""
#)

# ------------------------------
# Data generators
# ------------------------------

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
    # Start from dataset 1 structure for familiarity
    X, y = generate_dataset_1(n, rng)
    # Flip a fraction of labels to make it linearly non-separable
    n_flip = int(flip_frac * n)
    idx = rng.choice(n, size=n_flip, replace=False)
    y[idx] = -y[idx]
    return X, y

# ------------------------------
# Perceptron (with pocket option)
# ------------------------------

def perceptron_train(X, y, lr=0.01, epochs=50, pocket=True, rng=None):
    """
    X: (n,2), y in {-1,+1}
    Returns weights w = [w1,w2,bias]
    """
    n = X.shape[0]
    Xb = np.column_stack([X, np.ones(n)])
    if rng is None:
        rng = np.random.default_rng(0)
    w = rng.normal(0, 1, size=3)

    if pocket:
        best_w = w.copy()
        best_acc = (np.sign(Xb @ w) == y).mean()

    for _ in range(epochs):
        # Shuffle to avoid cycles
        idx = rng.permutation(n)
        for j in idx:
            pred = np.sign(Xb[j] @ w) or 1  # ensure not zero
            if pred != y[j]:
                w = w + lr * y[j] * Xb[j]
        if pocket:
            acc = (np.sign(Xb @ w) == y).mean()
            if acc > best_acc:
                best_acc = acc
                best_w = w.copy()
    return (best_w if pocket else w)

def line_from_weights(w):
    # w1*x + w2*y + b = 0 => y = -(w1/w2) x - b/w2
    w1, w2, b = w
    if np.isclose(w2, 0.0):
        return None, None  # vertical or undefined slope
    slope = -w1 / w2
    intercept = -b / w2
    return slope, intercept

def classify_with_line(X, slope, intercept):
    # sign(w1*x + w2*y + b) with w2=1 representation => y - (slope*x + intercept)
    # Equivalent to: y - (m x + b)
    y_hat = np.where(X[:,1] - (slope * X[:,0] + intercept) >= 0, 1, -1)
    return y_hat

# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.header("Controls")

dataset = st.sidebar.selectbox("Dataset", ["1: linearly separable", "2: non-separable"])
n_obs = st.sidebar.slider("Number of observations (n)", min_value=50, max_value=3000, value=500, step=50)
seed = st.sidebar.number_input("Random seed", min_value=0, value=10, step=1)

user_slope = st.sidebar.slider("Your line slope (m)", min_value=-0.15, max_value=0.20, value=0.0, step=0.005)
user_intercept = st.sidebar.slider("Your line intercept (b)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Perceptron hyperparameters")
lr = st.sidebar.slider("Learning rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
epochs = st.sidebar.slider("Epochs", min_value=10, max_value=1000, value=200, step=10)
use_pocket = st.sidebar.checkbox("Use pocket algorithm (keeps best weights)", value=True)

flip_frac = None
if dataset.startswith("2"):
    flip_frac = st.sidebar.slider("Noise: fraction of labels flipped", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

rng = np.random.default_rng(int(seed))

# ------------------------------
# Generate data
# ------------------------------
if dataset.startswith("1"):
    X, y = generate_dataset_1(n_obs, rng)
else:
    X, y = generate_dataset_2(n_obs, rng, flip_frac=flip_frac)

# ------------------------------
# Train perceptron (on button)
# ------------------------------
run_algo = st.button("Run perceptron")

if run_algo:
    w = perceptron_train(X, y, lr=lr, epochs=epochs, pocket=use_pocket, rng=rng)
    perc_slope, perc_intercept = line_from_weights(w)
else:
    w = None
    perc_slope, perc_intercept = None, None

# ------------------------------
# Compute accuracies
# ------------------------------
def accuracy_from_line(X, y, slope, intercept):
    y_pred = classify_with_line(X, slope, intercept)
    return (y_pred == y).mean()

user_acc = accuracy_from_line(X, y, user_slope, user_intercept)

if perc_slope is not None:
    perc_acc = accuracy_from_line(X, y, perc_slope, perc_intercept)
else:
    perc_acc = None

# ------------------------------
# Plot
# ------------------------------
fig, ax = plt.subplots()

# plot points
mask_pos = y == 1
mask_neg = ~mask_pos
ax.scatter(X[mask_pos, 0], X[mask_pos, 1], marker='o', label='+1 (paid)')
ax.scatter(X[mask_neg, 0], X[mask_neg, 1], marker='x', label='-1 (default)')
# Label the axes
ax.set_xlabel('Credit score')   # Replace with your actual variable name
ax.set_ylabel('Debt/income')

# x-range for lines
x_min, x_max = X[:,0].min(), X[:,0].max()
x_vals = np.linspace(x_min, x_max, 200)


# user-chosen line
y_user = user_slope * x_vals + user_intercept
ax.plot(x_vals, y_user, linestyle='--', label=f'Your line: y = {user_slope:.2f}x + {user_intercept:.2f}\n(acc={user_acc:.2%})')

# perceptron line (if available)
if perc_slope is not None:
    if perc_slope is not None and perc_intercept is not None:
        y_perc = perc_slope * x_vals + perc_intercept
        ax.plot(x_vals, y_perc, linewidth=2, label=f'Perceptron: y = {perc_slope:.2f}x + {perc_intercept:.2f}\n(acc={perc_acc:.2%})')
    else:
        st.info("Perceptron produced a vertical line (w2≈0). Not plotted.")

ax.set_xlabel("Credit score")
ax.set_ylabel("Debt/income")
ax.legend(loc="best")
ax.set_title("Perceptron Classification of Credit Applicants")

st.pyplot(fig)

# ------------------------------
# Notes
# ------------------------------
with st.expander("What to notice"):
    st.markdown(
        """
- In **Dataset 1**, the perceptron usually finds a line with **very high accuracy**.
- In **Dataset 2**, because labels are **noisy**, **no perfect separating line exists**; the perceptron tries to minimize mistakes but cannot reach 100%.
- Try adjusting your own line and compare your accuracy with the perceptron's.
- The **pocket** option keeps the best weights seen across epochs, which helps on non-separable data.
"""
    )
