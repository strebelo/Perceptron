# treasure_hunt_q_learning.py
# Streamlit Treasure Hunt (Q-learning) — MATLAB -> Python
# Run: streamlit run treasure_hunt_q_learning.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Tuple

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

def state_index(r: int, c: int, n_cols: int) -> int:
    # (row, col) are 1-based for display; convert to 0-based linear index
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
    R = -1 * np.ones((n_rows, n_cols), dtype=float)
    # Match MATLAB setup
    R[0, 3] = 10     # (1,4) treasure
    R[1, 1] = -7     # (2,2) major obstacle
    R[1, 0] = -3     # (2,1) minor obstacle
    R[2, 2] = -5     # (3,3) obstacle
    return R

def init_q(n_rows: int, n_cols: int) -> np.ndarray:
    n_states = n_rows * n_cols
    Q = np.full((n_states, 4), np.nan, dtype=float)  # invalid = NaN
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            s = state_index(r, c, n_cols)
            if r > 1:  Q[s, ACTION_MAP["up"]] = 0.0
            if r < n_rows: Q[s, ACTION_MAP["down"]] = 0.0
            if c > 1:  Q[s, ACTION_MAP["left"]] = 0.0
            if c < n_cols: Q[s, ACTION_MAP["right"]] = 0.0
    return Q

def q_update(Q: np.ndarray, s: int, a_idx: int, r: float, s_next: int, alpha: float, gamma: float) -> None:
    max_next = np.nanmax(Q[s_next, :])
    Q[s, a_idx] = Q[s, a_idx] + alpha * (r + gamma * max_next - Q[s, a_idx])

def draw_grid(R: np.ndarray, visited: np.ndarray, agent_pos: Tuple[int, int],
              start_pos: Tuple[int, int], treasure_pos: Tuple[int, int]):
    """Matplotlib grid: rewards on discovered cells, '?' on unknown; NO rectangle on treasure cell."""
    n_rows, n_cols = R.shape
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(0, n_cols + 1))
    ax.set_yticks(np.arange(0, n_rows + 1))
    ax.grid(True)
    ax.invert_yaxis()  # row 1 at top (like the MATLAB printout)

    for i in range(n_rows):
        for j in range(n_cols):
            y = i
            x = j
            cell_coords = (i + 1, j + 1)
            if cell_coords == treasure_pos:
                # Skip rectangle/label here to avoid the unwanted box under the 💎
                continue
            if visited[i, j] or cell_coords == start_pos:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, alpha=0.20))
                ax.text(x + 0.5, y + 0.6, f"{R[i, j]:.1f}", ha="center", va="center", fontsize=11)
            else:
                ax.text(x + 0.5, y + 0.6, "?", ha="center", va="center", fontsize=12, fontweight="bold")

    # Treasure icon (no rectangle)
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
    st.session_state.treasure_pos = (1, 4)
    st.session_state.log = []
    st.session_state.episodes_played = 0

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.title("🧭 Controls")
    st.session_state.alpha = st.slider("Learning rate (α)", 0.0, 1.0, float(st.session_state.alpha), 0.05)
    st.session_state.gamma = st.slider("Discount factor (γ)", 0.0, 0.99, float(st.session_state.gamma), 0.01)

    st.divider()
    st.markdown("**Episodes & Resets**")
    if st.button("🔄 Reset episode", use_container_width=True):
        st.session_state.pos = tuple(st.session_state.start_pos)
        st.session_state.episode_score = 0.0
        st.session_state.log.append("Episode reset.")

    if st.button("🧹 Reset ALL (Q, visits)", use_container_width=True):
        st.session_state.R = init_rewards(st.session_state.n_rows, st.session_state.n_cols)
        st.session_state.Q = init_q(st.session_state.n_rows, st.session_state.n_cols)
        st.session_state.visited = np.zeros_like(st.session_state.R, dtype=bool)
        st.session_state.visited[st.session_state.start_pos[0]-1, st.session_state.start_pos[1]-1] = True
        st.session_state.pos = tuple(st.session_state.start_pos)
        st.session_state.episode_score = 0.0
        st.session_state.log.clear()
        st.session_state.episodes_played = 0

# ------------------------------
# Step function (manual or auto)
# ------------------------------
def do_step(action: str) -> bool:
    """Execute one action; return True if treasure found (episode ends)."""
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

    q_update(
        st.session_state.Q, s, a_idx, reward, s_next,
        alpha=float(st.session_state.alpha), gamma=float(st.session_state.gamma)
    )

    st.session_state.pos = (r_next, c_next)

    if (r_next, c_next) == st.session_state.treasure_pos:
        st.balloons()
        st.success(f"🎉 Treasure found! Total reward this episode: {st.session_state.episode_score:.1f}")
        # start a new episode automatically
        st.session_state.pos = tuple(st.session_state.start_pos)
        st.session_state.episode_score = 0.0
        st.session_state.log.append("🔁 Starting a new episode…")
        st.session_state.episodes_played += 1
        return True

    st.session_state.log.append(f"Moved **{action}** → ({r_next},{c_next}), reward {reward:.1f}")
    return False

def play_random_episode(max_steps: int = 500) -> bool:
    """Play one full episode with RANDOM valid moves until treasure found or step cap reached.
       Returns True if treasure found within cap, False otherwise."""
    steps = 0
    while steps < max_steps:
        r, c = st.session_state.pos
        candidates = [a for a in ACTIONS if valid_move(r, c, a, st.session_state.n_rows, st.session_state.n_cols)]
        if not candidates:
            # Shouldn't happen, but guard anyway
            st.session_state.log.append("No valid moves; aborting episode.")
            return False
        action = np.random.choice(candidates)
        ended = do_step(action)
        steps += 1
        if ended:
            return True
    st.warning(f"Reached max steps ({max_steps}) without finding treasure; episode aborted.")
    # Reset episode state to start for the next run
    st.session_state.pos = tuple(st.session_state.start_pos)
    st.session_state.episode_score = 0.0
    st.session_state.log.append("🔁 Starting a new episode…")
    return False

# ------------------------------
# Main layout
# ------------------------------
st.title("🗺️ Treasure Hunt — Q-Learning Interactive")
st.markdown(
    "Guide the **agent (🤖)** from **start (⚑ at (4,1))** to the **treasure (💎 at (1,4))**. "
    "Each move has a reward (movement cost −1; obstacles negative; treasure +10). "
    "Use the buttons to move or run **multiple random episodes**; the **Q-matrix** updates live."
)

left, right = st.columns([1.05, 1])

with left:
    st.subheader("🎮 Manual Moves")
    c_up = st.button("⬆️ Up", use_container_width=True)
    cols = st.columns(3)
    c_left = cols[0].button("⬅️ Left", use_container_width=True)
    c_right = cols[2].button("➡️ Right", use_container_width=True)
    c_down = st.button("⬇️ Down", use_container_width=True)

    if c_up:   do_step("up")
    if c_left: do_step("left")
    if c_right: do_step("right")
    if c_down: do_step("down")

    st.divider()
    st.subheader("🤖 Auto-play (Random Episodes)")
    n_eps = st.number_input("Number of random episodes to run", min_value=1, max_value=500, value=5, step=1)
    max_steps = st.number_input("Max steps per episode (safety cap)", min_value=10, max_value=5000, value=500, step=10)
    if st.button("▶️ Run N random episodes", use_container_width=True):
        successes = 0
        base_played = st.session_state.episodes_played
        for _ in range(int(n_eps)):
            found = play_random_episode(max_steps=int(max_steps))
            if found:
                successes += 1
        st.info(f"Ran {int(n_eps)} episode(s). Found treasure in {successes} episode(s). "
                f"Total episodes completed so far: {st.session_state.episodes_played} "
                f"(+{st.session_state.episodes_played - base_played}).")

    st.subheader("🧩 Discovered Grid")
    fig = draw_grid(
        st.session_state.R,
        st.session_state.visited,
        st.session_state.pos,
        st.session_state.start_pos,
        st.session_state.treasure_pos,
    )
    st.pyplot(fig, use_container_width=True)
    st.caption("Shown values are rewards (R). Unknown cells display “?”. The agent (🤖) and treasure (💎) are always visible.")

    # KPIs
    k1, k2, k3 = st.columns(3)
    with k1: st.metric("Current position", f"{st.session_state.pos}")
    with k2: st.metric("Episode reward", f"{st.session_state.episode_score:.1f}")
    with k3:
        r, c = st.session_state.pos
        s = state_index(r, c, st.session_state.n_cols)
        vmax = np.nanmax(st.session_state.Q[s, :])
        st.metric("Max Q at current state", f"{vmax:.2f}")

with right:
    st.subheader("📊 Q-Matrix (state × actions)")
    dfQ = q_dataframe(st.session_state.Q, st.session_state.n_rows, st.session_state.n_cols)
    # Robust display: try styling; fall back to plain if the environment lacks Styler features
    try:
        styled = dfQ.style.format("{:.2f}").background_gradient(axis=None, cmap="Blues")
        st.dataframe(styled, use_container_width=True, height=420)
    except Exception:
        st.dataframe(dfQ.round(2).fillna(""), use_container_width=True, height=420)

    with st.expander("Show raw Q-matrix values"):
        st.write(dfQ)

    st.subheader("📝 Activity Log")
    if st.session_state.log:
        for line in st.session_state.log[::-1][:14]:
            st.markdown(f"- {line}")
    else:
        st.markdown("_No moves yet._")

# ------------------------------
# Footnote
# ------------------------------
st.markdown(
    "<div class='small'>Update rule: <code>Q(s,a) ← Q(s,a) + α [ r + γ maxₐ′ Q(s′,a′) − Q(s,a) ]</code>. "
    "Random episodes use valid random moves until the treasure is found (or the step cap hits).</div>",
    unsafe_allow_html=True,
)
