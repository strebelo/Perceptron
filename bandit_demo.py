# bandit_demo.py
# Streamlit demo: epsilon-greedy + step-size (alpha) in a deceptive 2-armed bandit
# Run: streamlit run bandit_demo.py

import random
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Exploration vs. Exploitation: Deceptive Bandit", layout="wide")
st.title("🎰 Deceptive Two-Armed Bandit (ε-greedy Q-learning)")

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
    epsilon = st.slider("ε-greedy (exploration prob)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    q0 = st.slider("Initial Q-value (optimism encourages exploration)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
    random_seed = st.number_input("Random seed (for reproducibility)", value=42, step=1)
    use_stationary_alpha = st.checkbox("Use sample-average updates (α = 1/N_a) instead of fixed α", value=False,
                                       help="When checked, step-size uses 1/(action count). Useful to compare to constant α.")

    st.caption(
        "Tips:\n"
        "- Set ε=0 and Q0=0 to show **suboptimal convergence** to Right.\n"
        "- Try small ε (e.g., 0.05) and observe recovery toward Left.\n"
        "- Try **optimistic** Q0 (e.g., 2.0) with ε=0 to show directed exploration by optimism."
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
def run_bandit(episodes, alpha, epsilon, q0, use_stationary_alpha, seed):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    Q = np.array([q0, q0], dtype=float)  # Q[0]=Left, Q[1]=Right
    N = np.zeros(2, dtype=int)           # action counts

    q_left_hist = np.zeros(episodes)
    q_right_hist = np.zeros(episodes)
    act_hist = np.zeros(episodes, dtype=int)
    rew_hist = np.zeros(episodes)
    optimal_action_hist = np.zeros(episodes, dtype=int)  # 1 if chose Left (truly optimal in EV), else 0

    for t in range(episodes):
        # ε-greedy action selection
        if rng.random() < epsilon:
            a = rng.choice([0, 1])
        else:
            a = int(np.argmax(Q))  # break ties by lower index

        r = env_reward(a, rng)

        # Track optimal choice (Left has higher EV)
        optimal_action_hist[t] = 1 if a == 0 else 0

        # Incremental update
        N[a] += 1
        if use_stationary_alpha and N[a] > 0:
            step = 1.0 / N[a]
        else:
            step = alpha

        Q[a] += step * (r - Q[a])

        # Log histories
        q_left_hist[t] = Q[0]
        q_right_hist[t] = Q[1]
        act_hist[t] = a
        rew_hist[t] = r

    return {
        "q_left": q_left_hist,
        "q_right": q_right_hist,
        "actions": act_hist,
        "rewards": rew_hist,
        "optimal": optimal_action_hist
    }

# ------------------------------
# Run multiple seeds for averaging
# ------------------------------
random.seed(random_seed)
all_qL, all_qR, all_opt, all_rew = [], [], [], []
for i in range(runs):
    out = run_bandit(
        episodes=episodes,
        alpha=alpha,
        epsilon=epsilon,
        q0=q0,
        use_stationary_alpha=use_stationary_alpha,
        seed=random_seed + i
    )
    all_qL.append(out["q_left"])
    all_qR.append(out["q_right"])
    all_opt.append(out["optimal"])
    all_rew.append(out["rewards"])

qL_mean = np.mean(np.stack(all_qL), axis=0)
qR_mean = np.mean(np.stack(all_qR), axis=0)
opt_rate = np.mean(np.stack(all_opt), axis=0)  # fraction choosing Left at each episode
cum_rew = np.mean(np.cumsum(np.stack(all_rew), axis=1), axis=0)

# ------------------------------
# Plots (matplotlib; one figure per chart)
# ------------------------------
col1, col2, col3 = st.columns(3)

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

with col3:
    fig3 = plt.figure()
    plt.plot(cum_rew)
    plt.xlabel("Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.title("Learning Performance")
    st.pyplot(fig3)

st.markdown("### What to Try")
st.markdown(
    """
- **Show suboptimal convergence**: set **ε = 0**, **Q0 = 0**, small **α** (e.g., 0.05).  
  The agent often sticks to **Right** after early bad luck on **Left**.
- **Recovery via exploration**: keep **ε small but > 0** (e.g., 0.05–0.1).  
  Over time, the agent discovers Left’s higher EV and switches.
- **Optimistic initialization**: set **Q0 > 0** (e.g., **2.0**) and **ε = 0**.  
  Optimism alone can drive early exploration and avoid the trap.
- **Stationary vs. non-stationary**: toggle **sample-average** updates to compare to fixed **α**.
"""
)

st.markdown("---")
st.subheader("Why this works as a demo")
st.write(
    "Left is **rare high reward** (sparse signal). Without exploration, early unlucky samples make Left look bad, so the agent exploits Right’s small but consistent reward. "
    "Students can see how **ε-greedy** and **α** change convergence, and how **optimistic initialization** can substitute for explicit ε."
)
