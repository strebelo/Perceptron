# bandit_demo_min.py
# Streamlit demo: epsilon-greedy + step-size (alpha) in a deceptive 2-armed bandit
# No Q0 slider, no cumulative reward plot.
# Run: streamlit run bandit_demo_min.py

import random
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Exploration vs. Exploitation: Deceptive Bandit", layout="wide")
st.title("🎰 The roles of the learning rate $\alpha$, and of exploration versus exploitation")

st.markdown(
    """
At each episode you start at the **center** and choose **Left** or **Right**.

- **Right**: reward = **0.1** with probability 1  
- **Left**: reward = **10** with probability **0.1**, else **0** (EV = 1)

Because **Left** pays rarely, agents without **exploration** often settle on **Right**, a **suboptimal** policy.  
Use the sliders to see how **ε** (exploration) and **α** (learning rate) affect learning.
"""
)

# ------------------------------
# Sidebar Controls
# ------------------------------
with st.sidebar:
    st.header("Controls")
    episodes = st.slider("Episodes", min_value=50, max_value=5000, value=500, step=50)
    runs = st.slider("Independent runs (averaging)", min_value=1, max_value=50, value=10, step=1,
                     help="Average multiple runs to smooth randomness.")
    alpha = st.slider("Learning rate α", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    epsilon = st.slider("ε-greedy (exploration prob.)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    random_seed = st.number_input("Random seed (for reproducibility)", value=42, step=1)
    use_stationary_alpha = st.checkbox(
        "Use sample-average updates (α = 1/N_a) instead of fixed α",
        value=False,
        help="When checked, step-size uses 1/(action count). Useful to compare to constant α."
    )

    st.caption(
        "Tips:\n"
        "- Set ε=0 to show **suboptimal convergence** to Right after early bad luck on Left.\n"
        "- Try small ε (0.05–0.1) to see recovery toward Left.\n"
        "- Toggle sample-average updates to compare learning dynamics."
    )

# ------------------------------
# Environment
# ------------------------------
def env_reward(action: int, rng: random.Random) -> float:
    """
    action: 0=Left (jackpot), 1=Right (safe)
    """
    if action == 1:  # Right
        return 0.1
    else:            # Left
        return 10.0 if rng.random() < 0.1 else 0.0

# ------------------------------
# Simulation (epsilon-greedy bandit)
# ------------------------------
def run_bandit(episodes, alpha, epsilon, use_stationary_alpha, seed):
    rng = random.Random(seed)

    Q = np.array([0.0, 0.0], dtype=float)  # Q[0]=Left, Q[1]=Right; neutral initialization
    N = np.zeros(2, dtype=int)             # action counts

    q_left_hist = np.zeros(episodes)
    q_right_hist = np.zeros(episodes)
    act_hist = np.zeros(episodes, dtype=int)
    optimal_action_hist = np.zeros(episodes, dtype=int)  # 1 if chose Left (optimal in EV), else 0

    for t in range(episodes):
        # ε-greedy with random tie-breaking
        if rng.random() < epsilon:
            a = rng.choice([0, 1])
        else:
            if abs(Q[0] - Q[1]) < 1e-12:
                a = rng.choice([0, 1])  # break ties randomly
            else:
                a = int(np.argmax(Q))

        r = env_reward(a, rng)

        # Track optimal choice (Left has higher EV)
        optimal_action_hist[t] = 1 if a == 0 else 0

        # Incremental update
        N[a] += 1
        step = (1.0 / N[a]) if use_stationary_alpha and N[a] > 0 else alpha
        Q[a] += step * (r - Q[a])

        # Log histories
        q_left_hist[t] = Q[0]
        q_right_hist[t] = Q[1]
        act_hist[t] = a

    return {
        "q_left": q_left_hist,
        "q_right": q_right_hist,
        "actions": act_hist,
        "optimal": optimal_action_hist
    }

# ------------------------------
# Run multiple seeds for averaging
# ------------------------------
random.seed(random_seed)
all_qL, all_qR, all_opt = [], [], []
for i in range(runs):
    out = run_bandit(
        episodes=episodes,
        alpha=alpha,
        epsilon=epsilon,
        use_stationary_alpha=use_stationary_alpha,
        seed=random_seed + i
    )
    all_qL.append(out["q_left"])
    all_qR.append(out["q_right"])
    all_opt.append(out["optimal"])

qL_mean = np.mean(np.stack(all_qL), axis=0)
qR_mean = np.mean(np.stack(all_qR), axis=0)
opt_rate = np.mean(np.stack(all_opt), axis=0)  # fraction choosing Left at each episode

# ------------------------------
# Plots (matplotlib; one figure per chart)
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    fig1 = plt.figure()
    plt.plot(qL_mean, label="Q(Left)")
    plt.plot(qR_mean, label="Q(Right)")
    plt.xlabel("Episode")
    plt.ylabel("Estimated Q")
    plt.title("Q-value Estimates")
    plt.legend()
    st.pyplot(fig1)

with col2:
    fig2 = plt.figure()
    plt.plot(opt_rate)
    plt.ylim(0, 1)
    plt.xlabel("Episode")
    plt.ylabel("P(Choose Left)")
    plt.title("Exploration & Policy over Time")
    st.pyplot(fig2)

st.markdown("---")
st.subheader("What to Observe")
st.markdown(
    """
- **ε = 0** (pure greedy) with neutral initialization often locks onto **Right** after early bad luck on Left.  
- **Small ε > 0** lets the agent occasionally try Left, discover its higher EV, and switch over time.  
- **Sample-average vs constant α**: sample averages converge smoothly in stationary problems; constant α reacts faster if you later extend to non-stationary rewards.
"""
)
