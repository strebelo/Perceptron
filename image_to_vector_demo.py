# image_to_vector_demo.py
# Streamlit demo: Show how an image becomes a vector of numbers
#
# Run: streamlit run image_to_vector_demo.py

import io
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image → Vector Demo", layout="wide")

st.title("Image → Vector of Numbers: An Intuitive Walkthrough")

st.markdown(
    """
This app shows the exact steps used to turn an image into a numerical vector:
1) **Load** an image → 2) **Grayscale** → 3) **Resize** to a grid → 4) **Normalize** →  
5) **View** the 2D array → 6) **Flatten** to a 1D vector.
"""
)

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Controls")

target_size = st.sidebar.slider("Target grid size (pixels per side)", min_value=8, max_value=64, value=28, step=4)
norm_mode = st.sidebar.selectbox("Normalization", ["0–255 (uint8)", "0–1 (float)"])
invert = st.sidebar.checkbox("Invert intensities (white=0 ↔ black=255)", value=False)
flatten_order = st.sidebar.radio("Flatten order", ["Row-major (C-like)", "Column-major (Fortran-like)"], index=0)
show_grid = st.sidebar.checkbox("Show grid lines on heatmap", value=True)
annotate_numbers = st.sidebar.checkbox("Overlay pixel values (small sizes recommended)", value=False)
preview_k = st.sidebar.slider("Preview first K vector entries", 10, 400, 100, step=10)

st.sidebar.markdown("---")
download_fmt = st.sidebar.selectbox("Download vector format", ["CSV", "NumPy .npy"])

# -----------------------
# Image input
# -----------------------
uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
colA, colB = st.columns(2, gap="large")

def load_image_default(size=256):
    # Create a simple synthetic image (gradient + circle) if none uploaded
    img = Image.new("RGB", (size, size), "white")
    arr = np.linspace(0, 255, size).astype(np.uint8)
    grad = np.tile(arr, (size, 1))
    base = Image.fromarray(grad).convert("L").convert("RGB")
    # draw a darker circle
    cx, cy
