# image_to_vector_demo.py
# Run: streamlit run image_to_vector_demo.py
# Purpose: Show, step by step, how an image becomes a vector of numbers.

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# --- Optional: HEIC support (harmless if not installed) ---
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_ENABLED = True
except Exception:
    HEIC_ENABLED = False

st.set_page_config(page_title="Image → Vector Demo", layout="wide")

st.title("Image → Vector of Numbers: A Visual Walkthrough")
st.caption("Upload a JPG/PNG (HEIC supported if pillow-heif is installed). We’ll show grayscale → resize → normalize → flatten.")

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Controls")
    target_size = st.slider("Target grid size", 8, 64, 28, step=4)
    norm_mode = st.selectbox("Normalization", ["0–255 (uint8)", "0–1 (float)"])
    invert = st.checkbox("Invert intensities", value=False)
    flatten_order = st.radio("Flatten order", ["Row-major (C-like)", "Column-major (Fortran-like)"], index=0)
    show_grid = st.checkbox("Show grid lines", value=True)
    annotate_numbers = st.checkbox("Overlay pixel values (≤32 advisable)", value=False)
    preview_k = st.slider("Preview first K vector entries", 10, 400, 100, step=10)
    st.markdown("---")
    download_fmt = st.selectbox("Download vector format", ["CSV", "NumPy .npy"])
    debug = st.toggle("Debug info", value=False)

# -----------------------
# Helpers
# -----------------------
def load_image_default(size=256):
    """Synthetic fallback image (gradient + circle)."""
    img = Image.new("RGB", (size, size), "white")
    arr = np.linspace(0, 255, size).astype(np.uint8)
    grad = np.tile(arr, (size, 1))
    base = Image.fromarray(grad).convert("L").convert("RGB")
    cx, cy, r = size // 2, size // 2, size // 3
    a = np.array(base)
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    a[mask] = [50, 50, 50]
    return Image.fromarray(a)

# -----------------------
# Upload
# -----------------------
st.subheader("Upload")
uploaded = st.file_uploader("Choose an image (JPG/PNG/HEIC/etc.)", type=None)

if debug:
    st.write("Uploader object:", uploaded)

# Two columns for visuals
colA, colB = st.columns(2, gap="large")

with st.status("Pipeline status", expanded=True) as status:
    # Step 1: Read image
    if uploaded is None:
        st.write("No file uploaded → using a synthetic example image.")
        img = load_image_default()
    else:
        st.write(f"Received file: **{uploaded.name}** ({uploaded.size} bytes, type={uploaded.type})")
        try:
            uploaded.seek(0)  # ensure we're at the start of the in-memory file
            img = Image.open(uploaded)
            if img.mode != "RGB":
                img = img.convert("RGB")
            st.write(f"Opened image with Pillow ✅  (mode={img.mode}, size={img.size})")
        except Exception as e:
            st.error("Could not open the uploaded file with Pillow.")
            st.exception(e)
            if not HEIC_ENABLED:
                st.info("If this is a HEIC image, install HEIC support:\n\npip install pillow-heif")
            st.write("Falling back to a synthetic example image.")
            img = load_image_default()

    with colA:
        st.subheader("1) Original Image")
        st.image(img, use_container_width=True, caption="Uploaded / Example")

    # Step 2: Grayscale (+ optional invert)
    gray = ImageOps.grayscale(img)
    if invert:
        gray = ImageOps.invert(gray)
    st.write("Converted to grayscale (and inverted if selected) ✅")

    # Step 3: Resize to fixed grid
    gray_small = gray.resize((target_size, target_size), Image.BILINEAR)
    with colB:
        st.subheader("2) Grayscale & Resized")
        st.image(gray_small, use_container_width=True, caption=f"{target_size}×{target_size}")

    # Step 4: Normalize to numeric array
    arr_uint8 = np.array(gray_small)  # shape (H, W)
    if norm_mode == "0–255 (uint8)":
        arr = arr_uint8.astype(np.uint8)
    else:
        arr = (arr_uint8 / 255.0).astype(np.float32)
    st.write(f"Normalized array ready ✅  (dtype={arr.dtype}, shape={arr.shape})")

    # Step 5: Heatmap of 2D array
    st.subheader("3) Pixel Grid as a 2D Array (Heatmap)")
    fig, ax = plt.subplots()
    ax.imshow(arr, cmap="gray", interpolation="nearest")
    if show_grid:
        ax.set_xticks(np.arange(-0.5, target_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, target_size, 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if annotate_numbers and target_size <= 32:
        for (i, j), val in np.ndenumerate(arr):
            txt = f"{val:.2f}" if arr.dtype.kind == "f" else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6)
    st.pyplot(fig, use_container_width=True)

    # Step 6: Flatten to 1D vector
    order = "C" if flatten_order.startswith("Row") else "F"
    vec = arr.flatten(order=order)
    st.subheader("4) Flattened Vector")
    st.markdown(
        f"**Shape:** {arr.shape} → **Vector length:** {vec.size}  \n"
        f"**Order:** {'Row-major (rows left→right, top→bottom)' if order=='C' else 'Column-major (cols top→bottom, left→right)'}"
    )
    st.code(f"First {preview_k} values: {np.array2string(vec[:preview_k], precision=3, separator=', ')}")

    # Vector bar plot (to see ordering/values)
    st.caption("Vector visualization (bar chart of values by index):")
    fig2, ax2 = plt.subplots()
    ax2.bar(np.arange(vec.size), vec)
    ax2.set_xlabel("Vector index")
    ax2.set_ylabel("Value" + (" (0–1)" if arr.dtype.kind == "f" else " (0–255)"))
    st.pyplot(fig2, use_container_width=True)

    # Step 7: Download vector
    if download_fmt == "CSV":
        csv_bytes = "\n".join(str(x) for x in vec).encode("utf-8")
        st.download_button("Download Vector (CSV)", data=csv_bytes, file_name="vector.csv", mime="text/csv")
    else:
        buf = io.BytesIO()
        np.save(buf, vec)
        st.download_button("Download Vector (NumPy .npy)", data=buf.getvalue(), file_name="vector.npy", mime="application/octet-stream")

    status.update(label="Pipeline complete", state="complete")

# -----------------------
# Educational notes
# -----------------------
with st.expander("What’s happening under the hood?"):
    st.markdown(
        """
**Grayscale:** Convert RGB → single channel (brightness), so each pixel is one number.  
**Resize:** Models need a fixed input size (e.g., 28×28 like MNIST).  
**Normalize:** Keep raw 0–255 or scale to 0–1 (friendlier for many models).  
**Flatten:** Turn the H×W grid into a length H·W vector in a consistent order.  
**Use in ML:** Linear models/MLPs often flatten; CNNs keep the 2D structure.
"""
    )
