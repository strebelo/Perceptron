# adversarial_demo.py
# Streamlit Adversarial Example Demo (FGSM on MobileNetV2)
# Run: streamlit run adversarial_demo.py

import io
import numpy as np
from PIL import Image
import streamlit as st

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input, decode_predictions
)

st.set_page_config(page_title="Adversarial Examples: FGSM Demo", layout="centered")
st.title("🔒 Adversarial Examples Demo (FGSM)")

st.markdown(
    """
This interactive demo shows how a tiny, almost invisible perturbation can cause a powerful
image classifier to produce a wildly wrong prediction.

**How to use:**
1. Upload an image (a clear object works best).
2. Pick an ε (epsilon) value — the perturbation size in model input space.
3. Click **Generate adversarial** and compare predictions.
"""
)

# ------------------------------
# Utilities
# ------------------------------
@st.cache_resource
def load_model():
    # Pretrained MobileNetV2 on ImageNet
    model = mobilenet_v2.MobileNetV2(weights="imagenet")
    return model

def load_image(file_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_resized = img.resize(target_size, Image.LANCZOS)
    return img, img_resized  # original (for display), resized (for model)

def np_to_pil(x_uint8):
    return Image.fromarray(x_uint8)

def to_uint8_image(x):
    """
    x is a float image in [0, 255] or [-1,1] after preprocess. We’ll handle both.
    If x is in [-1,1] (MobileNetV2 preprocessed), convert back to [0,255].
    """
    x = np.array(x)
    if x.min() >= -1.0 and x.max() <= 1.0:
        x = (x + 1.0) * 127.5
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def topk_preds(model, x_pre, k=5):
    preds = model(x_pre, training=False).numpy()
    decoded = decode_predictions(preds, top=k)[0]
    return [(cls, label, float(prob)) for (cls, label, prob) in decoded], preds

def make_one_hot(indices, num_classes=1000):
    y = np.zeros((1, num_classes), dtype=np.float32)
    y[0, indices] = 1.0
    return tf.convert_to_tensor(y)

def fgsm_attack(model, x_pre, y_true_onehot, epsilon):
    """
    x_pre: preprocessed input (batch, 224,224,3), in MobileNetV2's [-1,1] range.
    y_true_onehot: one-hot tensor of the *original predicted class* (non-targeted)
    epsilon: perturbation size (e.g. 0.005–0.03). Operates in preprocessed space.
    """
    x_var = tf.Variable(x_pre)
    with tf.GradientTape() as tape:
        tape.watch(x_var)
        y_pred = model(x_var, training=False)
        # Use categorical cross-entropy with original (predicted) label
        loss = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred)
    grad = tape.gradient(loss, x_var)
    signed_grad = tf.sign(grad)
    # Add perturbation and clip to MobileNetV2 input range [-1, 1]
    x_adv = x_var + epsilon * signed_grad
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return x_adv.numpy()

# ------------------------------
# Sidebar controls
# ------------------------------
with st.sidebar:
    st.header("Controls")
    epsilon = st.slider(
        "Epsilon (perturbation size)",
        min_value=0.0, max_value=0.05, value=0.01, step=0.002,
        help="Higher values = stronger (more visible) attack. Typical range: 0.005–0.02"
    )
    show_diff = st.checkbox("Show perturbation (amplified for visibility)", value=True)
    amplify = st.slider("Amplification factor for diff preview", 1, 50, 15)

# ------------------------------
# Main UI
# ------------------------------
model = load_model()

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
col1, col2 = st.columns(2)

if uploaded is not None:
    original_img, resized_img = load_image(uploaded.read(), target_size=(224, 224))

    # Prepare model input (batch, H, W, C), preprocessed for MobileNetV2
    x = np.array(resized_img, dtype=np.float32)
    x_batch = np.expand_dims(x, axis=0)
    x_pre = preprocess_input(x_batch.copy())  # [-1,1]

    # --- Original prediction ---
    with st.spinner("Analyzing original image..."):
        original_top5, original_logits = topk_preds(model, x_pre, k=5)

    # Display original image + preds
    with col1:
        st.subheader("Original")
        st.image(original_img, caption="Uploaded image (displayed at original resolution)")

        st.write("**Top-5 predictions (original):**")
        st.table(
            [{"rank": i+1, "label": lbl, "probability": f"{p:.3f}"} for i, (_, lbl, p) in enumerate(original_top5)]
        )

    # Pick the top-1 predicted class as "true" for a non-targeted attack
    top1_index = int(np.argmax(original_logits, axis=1))
    y_true = make_one_hot([top1_index], num_classes=1000)

    # Attack button
    if st.button("🚀 Generate adversarial"):
        # --- Craft adversarial example ---
        with st.spinner("Crafting adversarial example..."):
            x_adv = fgsm_attack(model, x_pre, y_true, epsilon)
            adv_top5, _ = topk_preds(model, x_adv, k=5)

        # Visuals
        adv_uint8 = to_uint8_image(x_adv[0])  # convert back to [0,255]
        orig_uint8 = to_uint8_image(x_pre[0])  # convert preprocessed original back for fair side-by-side

        with col2:
            st.subheader("Adversarial")
            st.image(adv_uint8, caption="Adversarial (shown at 224×224)")

            st.write("**Top-5 predictions (adversarial):**")
            st.table(
                [{"rank": i+1, "label": lbl, "probability": f"{p:.3f}"} for i, (_, lbl, p) in enumerate(adv_top5)]
            )

        # Diff view (amplified for human visibility)
        if show_diff:
            st.subheader("Perturbation (visualized)")
            diff = (adv_uint8.astype(np.int16) - orig_uint8.astype(np.int16))  # signed
            diff_vis = np.clip((diff * amplify) + 128, 0, 255).astype(np.uint8)
            st.image(diff_vis, caption=f"Amplified difference (×{amplify}) • True perturbation is much smaller/invisible")

        # Summary
        st.markdown("---")
        orig_label, orig_prob = original_top5[0][1], original_top5[0][2]
        adv_label, adv_prob = adv_top5[0][1], adv_top5[0][2]
        st.markdown(
            f"""
**Result:** The model went from **{orig_label}** ({orig_prob:.3f})  
to **{adv_label}** ({adv_prob:.3f}) with ε = **{epsilon:.3f}**.

- The noise is added in the model’s normalized input space (range [-1, 1]).  
- We used **FGSM**: `x_adv = clip(x + ε · sign(∇_x loss))`.  
- Loss is the cross-entropy with the original top-1 class, so the attack *reduces* the model’s confidence in that class (non-targeted).
"""
        )
else:
    st.info("👆 Upload an image to get started (PNG/JPG/WebP).")

st.caption("Model: MobileNetV2 (ImageNet). Demo for educational purposes.")
