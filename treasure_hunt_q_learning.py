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
ACTION_MAP = {a: i for i, a in enumerate(ACTIONS)}  # action -> index

def state_index(r, c, n_cols):
    """(row, col) -> linear index (0-based)."""
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
    # Match your MATLAB example
    # Treasure
    R[0, 3] = 10      # (1,4)
    # Obstacles
    R[1, 1] = -7      # (2,2)
    R[1, 0] = -3      # (2,1)
    R[2, 2] = -5      # (3,3)
    return R

def init_q(n_rows, n_cols):
    # NaN everywhere; 0 for valid actions from each state (like MATLAB)
    n_states = n_rows * n_cols
    Q = np.full((n_states, 4), np.nan, dtype=float)
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            s = state_index(r, c, n_cols)
            if r > 1:              Q[s, ACTION_MAP["up"]] = 0.0
            if r < n_rows:         Q[s, ACTION_MAP["down"]] = 0.0
            if c > 1:              Q[s, ACTION_MAP["left"]] = 0.0
            if c < n_cols:         Q[s, ACTION_MAP["right"]] = 0.0
    return Q

def q_update(Q, s, a_idx, r, s_next, alpha, gamma):
    # max over valid actions in next state (ignore NaNs)
    q_next = Q[s_next, :]
    max_next = np.nanmax(q_next)  # safe because invalid actions are NaN
    Q[s, a_idx] = Q[s, a_idx] + alpha * (r + gamma * max_next - Q[s, a_idx])

def draw_grid(R, visited, agent_pos, start_pos, treasure_pos):
    """Matplotlib grid view: shows rewards for discovered cells and ? for unknown."""
    n_rows, n_cols = R.shape
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(0, n_cols + 1))
    ax.set_yticks(np.arange(0, n_rows + 1))
    ax.grid(True)
    ax.invert_yaxis()  # row 1 at top like MATLAB display

    # Shade visited cells
    for i in range(n_rows):
        for j in range(n_cols):
            y = i
            x = j
            if visited[i, j] or (i + 1, j + 1) == start_pos:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, alpha=0.20))
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

def q_dataframe(Q, n_rows, n_cols):
    labels = [f"({r},{c})" for r in range(1, n_rows + 1) for c in range(1, n_cols + 1)]
    df = pd.DataFrame(Q, columns=ACTIONS, index=labels)
    return df

# ------------------------------
# Session state (persist across interactions)
# ------------------------------
if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.n_rows = 4
    st.session_state.n_cols = 4
    st.session_state.alpha = 1.0
    st.session_state.gamma = 0.8
    st.session_state.R = init_rewards(st.session_state.n_rows, st.session_state.n_cols)
    st.session_state.Q = init_q(st.session_state.n_rows, st.session_state.n_cols)
    st.session_state.start_pos = (4, 1)  # (row, col)
    st.session_state.pos = tuple(st.session_state.start_pos)
    st.session_state.visited = np.zeros_like(st.session_state.R, dtype=bool)
    st.session_state.visited[st.session_state.start_pos[0]-1, st.session_state.start_pos[1]-1] = True
    st.session_state.episode_score = 0.0
    st.session_state.log = []
    st.session_state.treasure_pos = (1, 4)
    st.session_state.autosteps = 10

# ------------------------------
# Sidebar controls
# ------------------------------
with st.sidebar:
    st.title("🧭 Controls")
    st.markdown("Tune learning and explore the grid.")
    st.session_state.alpha = st.slider("Learning rate (α)", 0.0, 1.0, float(st.session_state.alpha), 0.05)
    st.session_state.gamma = st.slider("Discount factor (γ)", 0.0, 0.99, float(st.session_state.gamma), 0.01)

    st.divider()
    st.markdown("**Episode**")
    col_reset1, col_reset2 = st.columns(2)
    with col_reset1:
        if st.button("🔄 Reset episode", use_container_width=True):
            st.session_state.pos = tuple(st.session_state.start_pos)
            st.session_state.episode_score = 0.0
            st.session_state.log.clear()
    with col_reset2:
        if st.button("🧹 Reset ALL (Q, visits)", use_container_width=True):
            st.session_state.R = init_rewards(st.session_state.n_rows, st.session_state.n_cols)
            st.session_state.Q = init_q(st.session_state.n_rows, st.session_state.n_cols)
            st.session_state.visited = np.zeros_like(st.session_state.R, dtype=bool)
            st.session_state.visited[st.session_state.start_pos[0]-1, st.session_state.start_pos[1]-1] = True
            st.session_state.pos = tuple(st.session_state.start_pos)
            st.session_state.episode_score = 0.0
            st.session_state.log.clear()

    st.divider()
    st.markdown("**Auto-play**")
    st.session_state.autosteps = st.number_input("Steps", min_value=1, max_value=200, value=st.session_state.autosteps, step=1)
    if st.button("▶️ Run Auto-play", use_container_width=True):
        for _ in range(int(st.session_state.autosteps)):
            # random valid action from current state
            r, c = st.session_state.pos
            candidates = [a for a in ACTIONS if valid_move(r, c, a, st.session_state.n_rows, st.session_state.n_cols)]
            # ε-greedy w.r.t. Q (ε=0.2)
            eps = 0.2
            if np.random.rand() < eps:
                action = np.random.choice(candidates)
            else:
                s = state_index(r, c, st.session_state.n_cols)
                qvals = st.session_state.Q[s, :]
                # among valid actions, pick argmax
                best = None
                best_q = -np.inf
                for a in candidates:
                    q = qvals[ACTION_MAP[a]]
                    if q > best_q:
                        best_q = q
                        best = a
                action = best or np.random.choice(candidates)
            # do one step
            # (use same function as manual step below)
            # If treasure hit, stop loop for a fresh episode
            ended = False
            ended = st.session_state.get("_do_step", lambda _a: False)(action)
            if ended:
                break

# ------------------------------
# Header
# ------------------------------
st.title("🗺️ Treasure Hunt — Q-Learning Interactive")
st.markdown(
    "Guide the **agent (🤖)** from **start (⚑)** to the **treasure (💎 at (1,4))**. "
    "Each move has a reward (movement cost −1; obstacles negative; treasure +10). "
    "Use the buttons or let it auto-play; the **Q-matrix** updates live."
)

# ------------------------------
# Core step function (attached into session so sidebar can call it)
# ------------------------------
def do_step(action: str):
    """Execute one student/agent action; return True if treasure found (episode reset)."""
    r, c = st.session_state.pos
    if action not in ACTIONS:
        st.session_state.log.append(f"❗ Invalid action: {action}")
        return False

    if not valid_move(r, c, action, st.session_state.n_rows, st.session_state.n_cols):
        st.session_state.log.append(f"❌ Invalid move from ({r},{c}) going {action}.")
        return False

    # compute next state
    r_next, c_next = next_position(r, c, action)
    st.session_state.visited[r_next - 1, c_next - 1] = True

    # rewards & Q update
    s = state_index(r, c, st.session_state.n_cols)
    s_next = state_index(r_next, c_next, st.session_state.n_cols)
    a_idx = ACTION_MAP[action]
    reward = st.session_state.R[r_next - 1, c_next - 1]
    st.session_state.episode_score += float(reward)

    q_update(
        st.session_state.Q, s, a_idx, reward, s_next,
        alpha=float(st.session_state.alpha),
        gamma=float(st.session_state.gamma),
    )

    # move agent
    st.session_state.pos = (r_next, c_next)

    # check treasure
    if (r_next, c_next) == st.session_state.treasure_pos:
        st.balloons()
        st.success(f"🎉 Treasure found! Total reward this episode: {st.session_state.episode_score:.1f}")
        # new episode
        st.session_state.pos = tuple(st.session_state.start_pos)
        st.session_state.episode_score = 0.0
        st.session_state.log.append("🔁 Starting a new episode…")
        return True

    # log discovered grid line
    st.session_state.log.append(f"Moved **{action}** → ({r_next},{c_next}), reward {reward:.1f}")
    return False

# store function in session for sidebar auto-play
st.session_state._do_step = do_step

# ------------------------------
# Layout: left = grid & controls, right = Q-table & logs
# ------------------------------
left, right = st.columns([1.05, 1])

with left:
    # Move controls
    st.subheader("🎮 Your Move")
    move_cols = st.columns([1, 1, 1])
    with move_cols[1]:
        up = st.button("⬆️ Up", use_container_width=True)
    with move_cols[0]:
        left_b = st.button("⬅️ Left", use_container_width=True)
    with move_cols[2]:
        right_b = st.button("➡️ Right", use_container_width=True)
    with move_cols[1]:
        down = st.button("⬇️ Down", use_container_width=True)

    rnd_col = st.columns([1, 1, 1])[1]
    with rnd_col:
        random_step = st.button("🎲 Random valid move", use_container_width=True)

    # Handle button presses
    ended = False
    if up:    ended = do_step("up")
    if left_b: ended = do_step("left") or ended
    if right_b: ended = do_step("right") or ended
    if down:  ended = do_step("down") or ended
    if random_step:
        r, c = st.session_state.pos
        candidates = [a for a in ACTIONS if valid_move(r, c, a, st.session_state.n_rows, st.session_state.n_cols)]
        if candidates:
            ended = do_step(np.random.choice(candidates)) or ended

    # Grid view
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
        st.metric("Max Q at state", f"{vmax:.2f}")

with right:
    st.subheader("📊 Q-Matrix (state × actions)")
    dfQ = q_dataframe(st.session_state.Q, st.session_state.n_rows, st.session_state.n_cols)
    # Highlight NaNs (invalid moves) for readability
    styled = dfQ.style.format("{:.2f}").background_gradient(axis=None, cmap="Blues", subset=pd.IndexSlice[:, :]).highlight_null(null_color="#eee")
    st.dataframe(styled, use_container_width=True, height=420)

    st.subheader("📝 Activity Log")
    if st.session_state.log:
        for line in st.session_state.log[::-1][:12]:
            st.markdown(f"- {line}")
    else:
        st.markdown("_Your moves and updates will appear here._")

# ------------------------------
# Footnote
# ------------------------------
st.markdown(
    "<div class='small'>Tip: try a few random steps, then manual choices. "
    "Observe how Q-values at visited states improve. "
    "The update is: <code>Q(s,a) ← Q(s,a) + α [ r + γ maxₐ′ Q(s′,a′) − Q(s,a) ]</code>.</div>",
    unsafe_allow_html=True,
)
