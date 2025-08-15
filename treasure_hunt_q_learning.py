# treasure_hunt_q_learning.py
# Streamlit Treasure Hunt (Q-learning)
# Run: streamlit run treasure_hunt_q_learning.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Tuple

# ---------------------------------------------------------------------
# Page config (keep this top-level so it always runs)
# ---------------------------------------------------------------------
st.set_page_config(page_title="Treasure Hunt: Q-Learning Demo",
                   layout="wide", page_icon="🗺️")

# ---------------------------------------------------------------------
# App code in main() so we can catch and surface exceptions nicely
# ---------------------------------------------------------------------
def main():
    # Heartbeat so you KNOW the app mounted
    st.write("✅ App loaded")

    # ------------------------------
    # Optional CSS (harmless if ignored)
    # ------------------------------
    st.markdown(
        """
        <style>
        .small { font-size: 0.85rem; color: #666; }
        div.stButton > button { border-radius: 14px; padding: 0.5rem 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------
    # Constants and helpers
    # ------------------------------
    ACTIONS = ["up", "down", "left", "right"]
    ACTION_MAP = {a: i for i, a in enumerate(ACTIONS)}

    def state_index(r: int, c: int, n_cols: int) -> int:
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
                if r > 1:      Q[s, ACTION_MAP["up"]] = 0.0
                if r < n_rows: Q[s, ACTION_MAP["down"]] = 0.0
                if c > 1:      Q[s, ACTION_MAP["left"]] = 0.0
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
        ax.invert_yaxis()  # row 1 at top

        for i in range(n_rows):
            for j in range(n_cols):
                y = i
                x = j
                cell_coords = (i + 1, j + 1)
                if cell_coords == treasure_pos:
                    # Skip rectangle/label here to avoid the box under the 💎
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
    # Session state (first run)
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
            st.session_st_
    n_states = n_rows * n_cols
    Q = np.full((n_states, 4), np.nan, dtype=float)  # invalid = NaN
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            s = state_index(r, c, n_cols)
            if r > 1:      Q[s, ACTION_MAP["up"]] = 0.0
            if r < n_rows: Q[s, ACTION_MAP["down"]] = 0.0
            if c > 1:      Q[s, ACTION_MAP["left"]] = 0.0
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
    ax.invert_yaxis()  # row 1 at top

    for i in range(n_rows):
        for j in range(n_cols):
            y = i
            x = j
            cell_coords = (i + 1, j + 1)
            if cell_coords == treasure_pos:
                # Skip rectangle/label here to avoid the box under the 💎
                continue
            if visited[i, j] or cell_coords == start_pos:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, alpha=0.20))
                ax.text(x + 0.5, y + 0.6, f"{R[i, j]:.1f}", ha="center", va="center", fontsize=11)
            else:
                ax.text(x + 0.5, y + 0.6, "?", ha="center", va="center", fontsize=12, fontweight="bold")

    # Treasure icon (no rectangle)
    tr, tc = treasure
