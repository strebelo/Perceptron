# treasure_hunt_q_learning.py
# Streamlit Treasure Hunt (Q-learning)
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
