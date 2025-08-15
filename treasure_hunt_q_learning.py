# treasure_hunt_q_learning.py
# Streamlit Treasure Hunt (Q-learning) — MATLAB -> Python
# Run: streamlit run treasure_hunt_q_learning.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------
# Page config & styles
# ------------------------------
st.set_page_config(page_title="Treasure Hunt: Q-Learning Demo", layout="wide", page_icon="🗺️")

CUSTOM_CSS = """
<style>
.small { font-size: 0.85rem; color: #666; }
.kpi { font-size: 1.1rem; font-weight: 600; }
.grid-note { font-size: 0.9rem; color: #555; }
div.stButton > button {
    border-radius: 14px;
    padding: 0.5rem 0.9rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------
# Constants and helpers
# ------------------------------
ACTIONS = ["up", "down", "left", "right"]
ACTION_MAP = {a: i for i, a in enumerate(ACTIONS)}

def state_index(r, c, n_cols):
    return (r - 1) * n_cols + (c - 1)

def valid_move(r, c, action, n_rows, n_cols):
    if action == "up":    return r > 1
    if action == "down":  return r < n_rows
    if action == "left":  return c > 1
    if action == "right": return c < n_cols
    return False

def next_position(r, c, action):
    if action == "up":    return r - 1, c
    if action == "down":  return r + 1, c
    if action == "left":  return r, c - 1
    if action == "right": return r, c + 1
    return r, c

def init_rewards(n_rows, n_cols):
    R = -1 * np.ones((n_rows, n_cols), dtype=float)
    R[0, 3] = 10     # Treasure
    R[1, 1] = -7     # Major obstacle
    R[1, 0] = -3     # Minor obstacle
    R[2, 2] = -5     # Obstacle
    return R

def init_q(n_rows, n_cols):
    n_states = n_rows * n_cols
    Q = np.full((n_states, 4), np.nan, dtype=float)
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            s = state_index(r, c, n_cols)
            if r > 1: Q[s, ACTION_MAP["up"]] = 0.0
            if r < n_rows: Q[s, ACTION_MAP["down"]] = 0.0
            if c > 1: Q[s, ACTION_MAP["left"]] = 0.0
            if c < n_cols: Q[s, ACTION_MAP["right"]] = 0.0
    return Q

def q_update(Q, s, a_idx, r, s_next, alpha, gamma):
    max_next = np.nanmax(Q[s_next, :])
    Q[s, a_idx] = Q[s, a_idx] + alpha * (r + gamma * max_next - Q[s, a_idx])

def draw_grid(R, visited, agent_pos, start_pos, treasure_pos):
    n_rows, n_cols = R.shape
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(0, n_cols + 1))
    ax.set_yticks(np.arange(0, n_rows + 1))
    ax.grid(True)
    ax.invert_yaxis()

    for i in range(n_rows):
        for j in range(n_cols):
            y = i
            x = j
            if visited[i, j] or (i+1, j+1) == start_pos:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, alpha=0.20))
                ax.text(x + 0.5, y + 0.6, f"{R[i, j]:.1f}", ha="center", va="center", fontsize=11)
            else:
                ax.text(x + 0.5, y + 0.6, "?", ha="center", va="center", fontsize=12, fontweight="bold")

    tr, tc = treasure_pos
    ax.text(tc - 0.5, tr - 0.5, "💎", ha="center", va="center", fontsize=22)

    sr, sc = start_pos
    ax.text(sc - 0.5, sr - 0.5, "⚑", ha="center", va="center", fontsize=16)

    ar, ac = agent_pos
    ax.text(ac - 0.5, ar - 0.5, "🤖", ha="center", va="center", fontsize=18)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout()
    return fig

def q_dataframe(Q, n_rows, n_cols):
    labels = [f"({r},{c})" for r in range(1, n_rows + 1) for c in range(1, n_cols + 1)]
    return pd.DataFrame(Q, columns=ACTIONS, index=labels)

# ------------------------------
# Session state
# ------------------------------
if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.n_rows = 4
    st.session_state.n_cols = 4
    st.session_state.alpha = 1.0
    st.session_state.gamma = 0.8
    st.session_state.R = init_rewards(st.session_state.n_rows, st.session_state.n_cols)
    st.session_state.Q = init_q(st.session_state.n_rows, st.session_state.n_cols)
    st.session_state.start_pos = (4, 1)
    st.session_state.pos = tuple(st.session_state.start_pos)
    st.session_state.visited = np.zeros_like(st.session_state.R, dtype=bool)
    st.session_state.visited[st.session_state.start_pos[0]-1, st.session_state.start_pos[1]-1] = True
    st.session_state.episode_score = 0.0
    st.session_state.log = []
    st.session_state.treasure_pos = (1, 4)

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.title("🧭 Controls")
    st.session_state.alpha = st.slider("Learning rate (α)", 0.0, 1.0, float(st.session_state.alpha), 0.05)
    st.session_state.gamma = st.slider("Discount factor (γ)", 0.0, 0.99, float(st.session_state.gamma), 0.01)

    if st.button("🔄 Reset episode", use_container_width=True):
        st.session_state.pos = tuple(st.session_state.start_pos)
        st.session_state.episode_score = 0.0
        st.session_state.log.clear()

    if st.button("🧹 Reset ALL", use_container_width=True):
        st.session_state.R = init_rewards(st.session_state.n_rows, st.session_state.n_cols)
        st.session_state.Q = init_q(st.session_state.n_rows, st.session_state.n_cols)
        st.session_state.visited = np.zeros_like(st.session_state.R, dtype=bool)
        st.session_state.visited[st.session_state.start_pos[0]-1, st.session_state.start_pos[1]-1] = True
        st.session_state.pos = tuple(st.session_state.start_pos)
        st.session_state.episode_score = 0.0
        st.session_state.log.clear()

# ------------------------------
# Step function
# ------------------------------
def do_step(action: str):
    r, c = st.session_state.pos
    if action not in ACTIONS:
        st.session_state.log.append(f"❗ Invalid action: {action}")
        return False
    if not valid_move(r, c, action, st.session_state.n_rows, st.session_state.n_cols):
        st.session_state.log.append(f"❌ Invalid move from ({r},{c}) going {action}.")
        return False

    r_next, c_next = next_position(r, c, action)
    st.session_state.visited[r_next - 1, c_next - 1] = True
    s = state_index(r, c, st.session_state.n_cols)
    s_next = state_index(r_next, c_next, st.session_state.n_cols)
    a_idx = ACTION_MAP[action]
    reward = st.session_state.R[r_next - 1, c_next - 1]
    st.session_state.episode_score += float(reward)

    q_update(st.session_state.Q, s, a_idx, reward, s_next,
             alpha=float(st.session_state.alpha), gamma=float(st.session_state.gamma))

    st.session_state.pos = (r_next, c_next)

    if (r_next, c_next) == st.session_state.treasure_pos:
        st.balloons()
        st.success(f"🎉 Treasure found! Total reward: {st.session_state.episode_score:.1f}")
        st.session_state.pos = tuple(st.session_state.start_pos)
        st.session_state.episode_score = 0.0
        st.session_state.log.append("🔁 Starting a new episode…")
        return True

    st.session_state.log.append(f"Moved **{action}** → ({r_next},{c_next}), reward {reward:.1f}")
    return False

# ------------------------------
# Main layout
# ------------------------------
st.title("🗺️ Treasure Hunt — Q-Learning Interactive")

col_left, col_right = st.columns([1.05, 1])

with col_left:
    st.subheader("🎮 Your Move")
    c_up = st.button("⬆️ Up")
    c_left = st.button("⬅️ Left")
    c_right = st.button("➡️ Right")
    c_down = st.button("⬇️ Down")
    c_rand = st.button("🎲 Random move")

    if c_up: do_step("up")
    if c_left: do_step("left")
    if c_right: do_step("right")
    if c_down: do_step("down")
    if c_rand:
        r, c = st.session_state.pos
        candidates = [a for a in ACTIONS if valid_move(r, c, a, st.session_state.n_rows, st.session_state.n_cols)]
        if candidates:
            do_step(np.random.choice(candidates))

    st.subheader("🧩 Discovered Grid")
    fig = draw_grid(st.session_state.R, st.session_state.visited,
                    st.session_state.pos, st.session_state.start_pos, st.session_state.treasure_pos)
    st.pyplot(fig, use_container_width=True)

with col_right:
    st.subheader("📊 Q-Matrix (state × actions)")
    dfQ = q_dataframe(st.session_state.Q, st.session_state.n_rows, st.session_state.n_cols)
    try:
        styled = dfQ.style.format("{:.2f}").background_gradient(axis=None, cmap="Blues")
        st.dataframe(styled, use_container_width=True, height=420)
    except Exception:
        st.dataframe(dfQ.round(2).fillna(""), use_container_width=True, height=420)

    with st.expander("Show raw Q-matrix values"):
        st.write(dfQ)

    st.subheader("📝 Activity Log")
    if st.session_state.log:
        for line in st.session_state.log[::-1][:12]:
            st.markdown(f"- {line}")
    else:
        st.markdown("_No moves yet._")
