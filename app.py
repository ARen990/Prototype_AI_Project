import io
import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Processing Lab (Streamlit)", layout="wide")

# ------------ Utilities ------------
def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """PIL -> OpenCV BGR"""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR -> PIL"""
    if img_bgr.ndim == 2:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def fetch_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

def apply_brightness_contrast(img: np.ndarray, brightness: int, contrast: int) -> np.ndarray:
    # brightness [-100,100], contrast [-100,100]
    beta = brightness
    if contrast >= 0:
        alpha = 1 + (contrast / 100) * 2.0   # up to 3.0
    else:
        alpha = 1 + (contrast / 100)         # down to 0.0
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out

def ensure_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr

def synthetic_sample() -> Image.Image:
    """Create a sample image if there is no file/camera/URL."""
    return Image.open(r".\simple.jpg").convert("RGB")

# ------------ Sidebar: Source & Controls ------------
st.sidebar.title("🧰 Controls")

src = st.sidebar.radio("Image source", ["Upload", "Webcam (camera_input)", "URL", "Sample"], index=0)

uploaded = None
img_pil = None
error = None

if src == "Upload":
    uploaded = st.sidebar.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            img_pil = Image.open(uploaded).convert("RGB")
        except Exception as e:
            error = f"Failed to read file: {e}"

elif src == "Webcam (camera_input)":
    cam = st.sidebar.camera_input("Take photo with a camera")
    if cam:
        try:
            img_pil = Image.open(cam).convert("RGB")
        except Exception as e:
            error = f"Failed to read image from camera: {e}"

elif src == "URL":
    url = st.sidebar.text_input("Enter the image URL (jpg/png) such as https://...")
    if url:
        try:
            img_pil = fetch_image_from_url(url)
        except Exception as e:
            error = f"Failed to retrieve image from URL: {e}"

else:
    img_pil = synthetic_sample()

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Image Processing Parameters")

brightness = st.sidebar.slider("Brightness (β)", -100, 100, 0, 1)
contrast   = st.sidebar.slider("Contrast (α)", -100, 100, 0, 1)

to_gray = st.sidebar.checkbox("Convert to Grayscale", value=False)

blur_on = st.sidebar.checkbox("Gaussian Blur", value=False)
kernel = st.sidebar.slider("Blur kernel (odd)", 1, 25, 5, step=2)

edge_on = st.sidebar.checkbox("Canny Edge", value=False)
canny_low  = st.sidebar.slider("Canny Threshold 1", 0, 255, 100, 1)
canny_high = st.sidebar.slider("Canny Threshold 2", 0, 255, 200, 1)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Try adjusting Brightness/Contrast first, then turn on Grayscale/Blur/Edge.")

# ---- Histogram display options ----
st.sidebar.markdown("---")
st.sidebar.subheader("📊 RGB Histogram Options")
bins = st.sidebar.slider("Bins", 16, 256, 64, step=16)
norm = st.sidebar.checkbox("Normalize (density)", True)
use_logy = st.sidebar.checkbox("Log Y scale", True)
mode = st.sidebar.radio("Combined style", ["Overlay bars", "Side-by-side bars", "Smooth lines"], index=1)
smooth_win = st.sidebar.slider("Smooth window (odd)", 3, 31, 9, step=2)

# ------------ Main UI ------------
st.title("🖼️ Streamlit Image Processing Lab")
st.write("Select the image on the left, adjust the parameters, and view the results and graphs below.")

if error:
    st.error(error)

if img_pil is not None:
    img_bgr = pil_to_cv2(img_pil)
    orig_bgr = img_bgr.copy()

    # 1) Brightness/Contrast
    img_bgr = apply_brightness_contrast(img_bgr, brightness, contrast)

    # 2) Grayscale (optional)
    work = ensure_gray(img_bgr) if to_gray else img_bgr

    # 3) Blur (optional)
    if blur_on and kernel >= 1:
        k = max(1, kernel)
        if k % 2 == 0:
            k += 1
        work = cv2.GaussianBlur(work, (k, k), 0)

    # 4) Canny (optional - auto grayscale inside)
    if edge_on:
        gray_for_edge = ensure_gray(work)
        edges = cv2.Canny(gray_for_edge, threshold1=canny_low, threshold2=canny_high)
        processed_display = edges  # single-channel
    else:
        processed_display = work

    # ---- Show images side-by-side ----
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Original image")
        st.image(cv2_to_pil(orig_bgr), use_container_width=True)
        st.caption(f"Image size: {orig_bgr.shape[1]}×{orig_bgr.shape[0]} px")

    with c2:
        st.subheader("Post-processed image")
        if processed_display.ndim == 2:
            st.image(processed_display, clamp=True, use_container_width=True)
        else:
            st.image(cv2_to_pil(processed_display), use_container_width=True)

        # Brief stats
        if processed_display.ndim == 2:
            mean_val = float(np.mean(processed_display))
            std_val  = float(np.std(processed_display))
            st.caption(f"Mean intensity: {mean_val:.2f} | Std: {std_val:.2f}")
        else:
            gray_tmp = cv2.cvtColor(processed_display, cv2.COLOR_BGR2GRAY)
            mean_val = float(np.mean(gray_tmp))
            std_val  = float(np.std(gray_tmp))
            st.caption(f"Mean intensity (gray): {mean_val:.2f} | Std: {std_val:.2f}")

    st.markdown("---")

    # ---- Feature Graphs: RGB Histograms (Combined + Separate) ----
    st.subheader("📈 RGB Histograms")

    # Prepare BGR images for histograms.
    if processed_display.ndim == 2:
        bgr_for_hist = cv2.cvtColor(processed_display, cv2.COLOR_GRAY2BGR)
    else:
        bgr_for_hist = processed_display.copy()

    b, g, r = cv2.split(bgr_for_hist)

    tabs = st.tabs(["Combined", "Red (R)", "Green (G)", "Blue (B)", "Grayscale (optional)"])

    # --- Combined ---
    with tabs[0]:
        rng = (0, 256)
        hist_r, bins_r = np.histogram(r.flatten(), bins=bins, range=rng, density=norm)
        hist_g, bins_g = np.histogram(g.flatten(), bins=bins, range=rng, density=norm)
        hist_b, bins_b = np.histogram(b.flatten(), bins=bins, range=rng, density=norm)
        centers = (bins_r[:-1] + bins_r[1:]) / 2.0

        # smoothing (moving average)
        if mode == "Smooth lines" and smooth_win > 1:
            k = np.ones(smooth_win) / smooth_win
            hist_r = np.convolve(hist_r, k, mode="same")
            hist_g = np.convolve(hist_g, k, mode="same")
            hist_b = np.convolve(hist_b, k, mode="same")

        fig = plt.figure(figsize=(7, 4))

        if mode == "Overlay bars":
            width = (256 / bins)
            plt.bar(bins_r[:-1], hist_r, width=width, color='red', alpha=0.4, label="R", align="edge")
            plt.bar(bins_g[:-1], hist_g, width=width, color='green', alpha=0.4, label="G", align="edge")
            plt.bar(bins_b[:-1], hist_b, width=width, color='blue', alpha=0.4, label="B", align="edge")

        elif mode == "Side-by-side bars":
            base_w = (256 / bins)
            w = base_w * 0.28
            plt.bar(bins_r[:-1] - w, hist_r, width=w, color='red', label="R", align="edge")
            plt.bar(bins_g[:-1]     , hist_g, width=w, color='green', label="G", align="edge")
            plt.bar(bins_b[:-1] + w , hist_b, width=w, color='blue', label="B", align="edge")

        else:  # Smooth lines
            plt.step(centers, hist_r, where="mid", color='red', label="R")
            plt.step(centers, hist_g, where="mid", color='green', label="G")
            plt.step(centers, hist_b, where="mid", color='blue', label="B")

        if use_logy:
            plt.yscale("log")

        plt.xlabel("Intensity (0-255)")
        plt.ylabel("Density" if norm else "Pixel count")
        plt.title(f"Combined RGB Histograms ({mode})")
        plt.legend()
        st.pyplot(fig, clear_figure=True, use_container_width=True)

    # --- R only ---
    with tabs[1]:
        hist_r, bins_r = np.histogram(r.flatten(), bins=bins, range=(0, 256), density=norm)
        fig = plt.figure(figsize=(7, 4))
        plt.bar(bins_r[:-1], hist_r, width=(256/bins), color='red', align="edge")
        if use_logy:
            plt.yscale("log")
        plt.xlabel("Intensity (0-255)")
        plt.ylabel("Density" if norm else "Pixel count")
        plt.title("Red Channel Histogram (R)")
        st.pyplot(fig, clear_figure=True, use_container_width=True)

    # --- G only ---
    with tabs[2]:
        hist_g, bins_g = np.histogram(g.flatten(), bins=bins, range=(0, 256), density=norm)
        fig = plt.figure(figsize=(7, 4))
        plt.bar(bins_g[:-1], hist_g, width=(256/bins), color='green', align="edge")
        if use_logy:
            plt.yscale("log")
        plt.xlabel("Intensity (0-255)")
        plt.ylabel("Density" if norm else "Pixel count")
        plt.title("Green Channel Histogram (G)")
        st.pyplot(fig, clear_figure=True, use_container_width=True)

    # --- B only ---
    with tabs[3]:
        hist_b, bins_b = np.histogram(b.flatten(), bins=bins, range=(0, 256), density=norm)
        fig = plt.figure(figsize=(7, 4))
        plt.bar(bins_b[:-1], hist_b, width=(256/bins), color='blue', align="edge")
        if use_logy:
            plt.yscale("log")
        plt.xlabel("Intensity (0-255)")
        plt.ylabel("Density" if norm else "Pixel count")
        plt.title("Blue Channel Histogram (B)")
        st.pyplot(fig, clear_figure=True, use_container_width=True)

    # --- Grayscale (optional) ---
    with tabs[4]:
        gray_for_hist = ensure_gray(bgr_for_hist)
        hist_gray, bins_gray = np.histogram(gray_for_hist.flatten(), bins=bins, range=(0, 256), density=norm)
        fig = plt.figure(figsize=(7, 4))
        plt.bar(bins_gray[:-1], hist_gray, width=(256/bins), color='black', align="edge")
        if use_logy:
            plt.yscale("log")
        plt.xlabel("Intensity (0-255)")
        plt.ylabel("Density" if norm else "Pixel count")
        plt.title("Grayscale Histogram (optional)")
        st.pyplot(fig, clear_figure=True, use_container_width=True)

    # ---- Extra metrics: Edge ratio (if Canny on) ----
    if edge_on:
        edge_ratio = float(np.count_nonzero(processed_display)) / processed_display.size
        st.info(f"Edge pixel ratio: {edge_ratio*100:.2f}%")

else:
    st.warning("Please select the image source on the left and upload/Webcam/insert URL to get started.")
