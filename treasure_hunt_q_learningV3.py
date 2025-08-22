# treasure_hunt_q_learningV3.py
# Minimal, robust Streamlit Treasure Hunt (Q-learning) with ε-greedy autoplay
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
ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}

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

def epsilon_greedy_action(Q_row: np.ndarray, epsilon: float, valid_mask: np.ndarray, rng: np.random.Generator) -> int:
    valid_actions = np.where(valid_mask)[0]
    if rng.random() < epsilon:
        return rng.choice(valid_actions)
    # exploit: pick argmax over valid actions
    q_vals = Q_row.copy()
    q_vals[~valid_mask] = -np.inf
    return int(np.argmax(q_vals))

def best_action_symbol(Q: np.ndarray, s: int) -> str:
    row = Q[s, :]
    if np.all(np.isnan(row)):
        return "·"
    a = int(np.nanargmax(row))
    return ARROWS[ACTIONS[a]]

def make_policy_grid(Q: np.ndarray, n_rows: int, n_cols: int) -> pd.DataFrame:
    symbols = []
    for r in range(1, n_rows + 1):
        row_syms = []
        for c in range(1, n_cols + 1):
            s = state_index(r, c, n_cols)
            row_syms.append(best_action_symbol(Q, s))
        symbols.append(row_syms)
    df = pd.DataFrame(symbols, columns=[f"C{j}" for j in range(1, n_cols + 1)])
    df.index = [f"R{i}" for i in range(1, n_rows + 1)]
    return df

def make_q_dataframe(Q: np.ndarray, n_rows: int, n_cols: int) -> pd.DataFrame:
    df = pd.DataFrame(Q, columns=ACTIONS)
    # label states as (r,c)
    df.index = [f"({(i // n_cols) + 1},{(i % n_cols) + 1})" for i in range(n_rows * n_cols)]
    # prettier NaNs
    return df.round(3)

# ----------------------------------------------------
# Sidebar controls
# ----------------------------------------------------
with st.sidebar:
    st.header("Environment")
    n_rows = st.number_input("Rows", min_value=2, max_value=6, value=3, step=1)
    n_cols = st.number_input("Cols", min_value=2, max_value=6, value=4, step=1)
    st.caption("Treasure: (1,4). Obstacles: (2,2) -7, (2,1) -3, (3,3) -5; movement cost -1.")

    st.header("Learning")
    alpha = st.slider("Learning rate (α)", 0.01, 1.0, 0.3, 0.01)
    gamma = st.slider("Discount (γ)", 0.0, 0.99, 0.9, 0.01)
    epsilon = st.slider("Exploration (ε)", 0.0, 1.0, 0.2, 0.01)
    max_steps = st.slider("Max steps per episode", 1, 100, 30, 1)

    st.header("Autoplay / Train")
    episodes = st.number_input("Episodes to train now", min_value=0, max_value=5000, value=100, step=50)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1)
    do_reset = st.button("🔄 Reset Q")

# ----------------------------------------------------
# State init
# ----------------------------------------------------
if "n_rows" not in st.session_state:
    st.session_state.n_rows = n_rows
    st.session_state.n_cols = n_cols
    st.session_state.R = init_rewards(n_rows, n_cols)
    st.session_state.Q = init_q(n_rows, n_cols)
    st.session_state.episode_count = 0
    st.session_state.rng = np.random.default_rng(seed)

# Re-init if shape changed or user reset
shape_changed = (st.session_state.n_rows != n_rows) or (st.session_state.n_cols != n_cols)
if do_reset or shape_changed:
    st.session_state.n_rows = n_rows
    st.session_state.n_cols = n_cols
