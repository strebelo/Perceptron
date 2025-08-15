# treasure_hunt_q_learning.py
# Minimal, robust Streamlit Treasure Hunt (Q-learning)
# Run: streamlit run treasure_hunt_q_learning.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Tuple

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(page_title="Treasure Hunt: Q-Learning", layout="wide", page_icon="🗺️")

# ----------------------------------------------------
# Core constants & helpers
# ----------------------------------------------------
ACTIONS = ["up", "down", "left", "right"]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

def state_index(r: int, c: int, n_cols: int) -> int:
    # (r,c) are 1-based; convert to 0-based linear index
    return (r - 1) * n_cols + (c - 1)

def valid_move(r: int, c: int, action: str, n_rows: int, n_cols: int) -> bool:
    if action == "up":    return r > 1
    if action == "down":  return r < n_rows
    if action == "left":  return c > 1
    if action == "right": return c < n_cols
    return False

def next_position(r: int, c: int, action: str) -> Tuple[int, int]:
    if action == "up":    return r - 1, c
    if action == "down":  return r + 1, c
    if action == "left":  return r, c - 1
    if action == "right": return r, c + 1
    return r, c

def init_rewards(n_rows: int, n_cols: int) -> np.ndarray:
    R = -1 * np.ones((n_rows, n_cols), dtype=float)   # movement cost
    R[0, 3] = 10.0   # treasure at (1,4)
    R[1, 1] = -7.0   # (2,2) major obstacle
    R[1, 0] = -3.0   # (2,1) minor obstacle
    R[2, 2] = -5.0   # (3,3) obstacle
    return R

def init_q(n_rows: int, n_cols: int) -> np.ndarray:
    # NaN for invalid actions; 0 for valid moves from each state
    Q = np.full((n_rows * n_cols, 4), np.nan, dtype=float)
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            s = state_index(r, c, n_cols)
            if r > 1:      Q[s, ACTION_IDX["up"]] = 0.0
            if r < n_rows: Q[s, ACTION_IDX["down"]] = 0.0
            if c > 1:      Q[s, ACTION_IDX["left"]] = 0.0
            if c < n_cols: Q[s, ACTION_IDX["right"]] = 0.0
    return Q

def q_update(Q: np.ndarray, s: int, a: int, r: float, s_next: int, alpha: float, gamma: float) -> None:
    max_next = np.nanmax(Q[s_next, :])  # ignore NaNs
    Q[s, a] = Q[s, a] + alpha * (r + gamma * max_next - Q[s, a])

def draw_grid(R: np.ndarray, visited: np.ndarray,
              agent_pos: Tuple[int, int], start_pos: Tuple[int, int],
              treasure_pos: Tuple[int, int]):
    """Plot grid: discovered rewards, '?' unknown, markers for start (⚑), agent (🤖), treasure (💎).
       No rectangle under treasure cell."""
    n_rows, n_cols = R.shape
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(range(0, n_cols + 1))
    ax.set_yticks(range(0, n_rows + 1))
    ax.grid(True)
    ax.invert_yaxis()  # row 1 at top

    for i in range(n_rows):
        for j in range(n_cols):
            x, y = j, i
            cell = (i + 1, j + 1)
            if cell == treasure_pos:
                # skip rectangle/label to avoid the box under 💎
                continue
            if visited[i, j] or cell == start_pos:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, alpha=0.2))
                ax.text(x + 0.5, y + 0.6, f"{R[i, j]:.1f}", ha="center", va="center", fontsize=11)
            else:
                ax.text(x + 0.5, y + 0.6, "?", ha="center", va="center", fontsize=12, fontweight="bold")

    # Treasure icon
    tr, tc = treasure_pos
    ax.text(tc - 0.5, tr - 0.5, "💎", ha="center", va="center", fontsize=22)
    # Start marker
    sr, sc = start_pos
    ax.text(sc - 0.5, sr - 0.5, "⚑", ha="center", va="center", fontsize=16)
    # Agent marker
    ar, ac = agent_pos
    ax.text(ac - 0.5, ar - 0.5, "🤖", ha="center", va="center", fontsize=18)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout()
    return fig

def q_dataframe(Q: np.ndarray, n_rows: int, n_cols: int) -> pd.DataFrame:
    labels = [f"({r},{c})" for r in range(1, n_rows + 1) for c in range(1, n_cols + 1)]
    df = pd.DataFrame(Q, columns=ACTIONS, index=labels)
    # Keep it super robust: round, and show NaNs as empty
    df = df.round(2)
    return df

# ----------------------------------------------------
# Session state (one-time init)
# ----------------------------------------------------
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.n_rows = 4
    st.session_state.n_cols = 4
    st.session_state.alpha = 1.0
    st.session_state.gamma = 0.8
    st.session_state.R = init_rewards(st.session_state.n_rows, st.session_state.n_cols)
    st.session_state.Q = init_q(st.session_state.n_rows, st.session_state.n_cols)
    st.session_state.start_pos = (4, 1)
    st.session_state.pos = (4, 1)
    st.session_state.visited = np.zeros_like(st.session_state.R, dtype=bool)
    st.session_state.visited[st.session_state.start_pos[0]-1, st.session_state.start_pos[1]-1] = True
    st.session_state.episode_score = 0.0
    st.session_state.treasure_pos = (1, 4)
    st.session_state.log = []
    st.session_state.episodes_done = 0

# ----------------------------------------------------
# Sidebar controls (simple & stable)
# ----------------------------------------------------
with st.sidebar:
    st.header("Controls")
    st.session_state.alpha = st.slider("Learning rate (α)", 0.0, 1.0, float(st.session_state.alpha), 0.05)
    st.session_state.gamma = st.slider("Discount factor (γ)", 0.0, 0.99, float(st.session_state.gamma), 0.01)

    st.write("---")
    if st.button("Reset episode"):
        st.session_state.pos = st.session_state.start_pos
        st.session_state.episode_score = 0.0
        st.session_state.log.append("Episode reset.")

    if st.button("Reset ALL (Q, visits)"):
        st.session_state.R = init_rewards(st.session_state.n_rows, st.session_state.n_cols)
        st.session_state.Q = init_q(st.session_state.n_rows, st.session_state.n_cols)
        st.session_state.visited = np.zeros_like(st.session_state.R, dtype=bool)
        st.session_state.visited[st.session_state.start_pos[0]-1, st.session_state.start_pos[1]-1] = True
        st.session_state.pos = st.session_state.start_pos
        st.session_state.episode_score = 0.0
        st.session_state.log.clear()
        st.session_state.episodes_done = 0

# ----------------------------------------------------
# Step logic & auto episodes
# ----------------------------------------------------
def do_step(action: str) -> bool:
    """Do one action; return True if treasure found and episode ends."""
    r, c = st.session_state.pos
    if action not in ACTIONS:
        st.session_state.log.append(f"Invalid action: {action}")
        return False
    if not valid_move(r, c, action, st.session_state.n_rows, st.session_state.n_cols):
        st.session_state.log.append(f"Invalid move from ({r},{c}) going {action}.")
        return False

    r2, c2 = next_position(r, c, action)
    st.session_state.visited[r2 - 1, c2 - 1] = True

    s = state_index(r, c, st.session_state.n_cols)
    s_next = state_index(r2, c2, st.session_state.n_cols)
    a_idx = ACTION_IDX[action]
    reward = float(st.session_state.R[r2 - 1, c2 - 1])
    st.session_state.episode_score += reward

    q_update(st.session_state.Q, s, a_idx, reward, s_next,
             alpha=float(st.session_state.alpha), gamma=float(st.session_state.gamma))

    st.session_state.pos = (r2, c2)

    if (r2, c2) == st.session_state.treasure_pos:
        st.balloons()
        st.success(f"Treasure found! Total reward this episode: {st.session_state.episode_score:.1f}")
        st.session_state.pos = st.session_state.start_pos
        st.session_state.episode_score = 0.0
        st.session_state.log.append("New episode started.")
        st.session_state.episodes_done += 1
        return True

    st.session_state.log.append(f"Moved {action} → ({r2},{c2}), reward {reward:.1f}")
    return False

def play_random_episode(max_steps: int = 500) -> bool:
    """Random valid moves until treasure found or step cap reached."""
    steps = 0
    while steps < max_steps:
        r, c = st.session_state.pos
        candidates = [a for a in ACTIONS if valid_move(r, c, a, st.session_state.n_rows, st.session_state.n_cols)]
        if not candidates:
            st.session_state.log.append("No valid moves; episode aborted.")
            return False
        action = np.random.choice(candidates)
        if do_step(action):
            return True
        steps += 1
    st.warning(f"Reached max steps ({max_steps}) without treasure; episode aborted.")
    st.session_state.pos = st.session_state.start_pos
    st.session_state.episode_score = 0.0
    st.session_state.log.append("New episode started.")
    return False

# ----------------------------------------------------
# Layout
# ----------------------------------------------------
st.title("🗺️ Treasure Hunt — Q-Learning Interactive")
st.write(
    "Start at **(4,1)** (⚑). Find the treasure at **(1,4)** (💎). "
    "Moves cost −1; obstacles are more negative; treasure gives +10. "
    "Q updates:  \n"
    "`Q(s,a) ← Q(s,a) + α [ r + γ max_a′ Q(s′,a′) − Q(s,a) ]`"
)

# Top row: left controls, right discovered grid
left, right = st.columns([1.0, 1.2])

with left:
    st.subheader("Manual moves")
    b_up = st.button("⬆️ Up", use_container_width=True)
    cols = st.columns(3)
    b_left  = cols[0].button("⬅️ Left", use_container_width=True)
    b_right = cols[2].button("➡️ Right", use_container_width=True)
    b_down = st.button("⬇️ Down", use_container_width=True)

    if b_up:    do_step("up")
    if b_left:  do_step("left")
    if b_right: do_step("right")
    if b_down:  do_step("down")

    st.write("---")
    st.subheader("Auto-play (random episodes)")
    n_eps = st.number_input("Episodes to run", min_value=1, max_value=500, value=5, step=1)
    max_steps = st.number_input("Max steps per episode", min_value=10, max_value=5000, value=500, step=10)
    if st.button("▶️ Run episodes", use_container_width=True):
        wins = 0
        start_done = st.session_state.episodes_done
        for _ in range(int(n_eps)):
            if play_random_episode(max_steps=int(max_steps)):
                wins += 1
        st.info(
            f"Ran {int(n_eps)} episode(s). Treasure found in {wins}. "
            f"Total episodes completed: {st.session_state.episodes_done} "
            f"(+{st.session_state.episodes_done - start_done})."
        )

with right:
    st.subheader("Discovered grid")
    fig = draw_grid(
        st.session_state.R,
        st.session_state.visited,
        st.session_state.pos,
        st.session_state.start_pos,
        st.session_state.treasure_pos,
    )
    st.pyplot(fig, use_container_width=True)

    k1, k2, k3 = st.columns(3)
    with k1: st.metric("Position", f"{st.session_state.pos}")
    with k2: st.metric("Episode reward", f"{st.session_state.episode_score:.1f}")
    with k3:
        r, c = st.session_state.pos
        s = state_index(r, c, st.session_state.n_cols)
        vmax = float(np.nanmax(st.session_state.Q[s, :]))
        st.metric("Max Q at state", f"{vmax:.2f}")

# Bottom: Q-matrix then log
st.subheader("Q-matrix (state × actions)")
dfQ = q_dataframe(st.session_state.Q, st.session_state.n_rows, st.session_state.n_cols)
# Show NaNs as blank strings to avoid any render quirks
st.dataframe(dfQ.where(pd.notnull(dfQ), ""), use_container_width=True, height=420)

st.subheader("Activity log")
if st.session_state.log:
    for line in st.session_state.log[::-1][:16]:
        st.write("• " + line)
else:
    st.write("_No moves yet._")
