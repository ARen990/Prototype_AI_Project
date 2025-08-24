# Explore the rapid prototype in AI Project

This project is an **interactive Image Processing web application** built with **[Streamlit](https://streamlit.io/)**.
Users can choose different image sources (file upload, camera input, URL, sample image, or live webcam), apply basic image processing operations, adjust parameters via the sidebar, and instantly view both processed results and image property graphs.



## Features

* **Image Sources**

  * Upload image (`.jpg`, `.jpeg`, `.png`)
  * Capture via `st.camera_input`
  * Load image from URL
  * Use a predefined sample image

* **Image Processing (toggleable)**

  * Adjust Brightness & Contrast
  * Convert to Grayscale
  * Apply Gaussian Blur (custom kernel size)
  * Perform Canny Edge Detection (custom thresholds)

* **Output & Visualization**

  * Display **original** and **processed** images side by side
  * Show basic statistics (image size, mean, standard deviation of intensity)
  * Plot **RGB Histograms** in multiple styles:

    * Combined (Overlay / Side-by-side / Smooth lines)
    * Red / Green / Blue channels separately
    * Grayscale histogram (optional)
  * Histogram customization: bins, normalization, log scale, smoothing


## Installation

1. Clone or download this repository:

   ```bash
   git clone (https://github.com/ARen990/Prototype_AI_Project.git)
   cd (https://github.com/ARen990/Prototype_AI_Project.git)
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:

   ```bash
   pip install streamlit opencv-python pillow matplotlib numpy requests
   ```


## Usage

Run the app:

```bash
streamlit run app.py
```

Open the link shown in your terminal (e.g., `http://localhost:8501`) in your browser.

## How to Use

1. Select an **Image Source** from the sidebar: Upload, Camera, URL, Sample, or Live Webcam.
2. Adjust image processing parameters (Brightness, Contrast, Blur, Canny, etc.).
3. See the processed image next to the original in real-time.
4. Explore histograms via tabs to analyze color distributions.
5. In **Live Webcam Mode**, RGB histograms update automatically every *N* frames.


## Histogram Options

* **Bins**: number of histogram bins
* **Normalize**: normalize histogram to density
* **Log Y scale**: view details on a logarithmic y-axis
* **Combined Style**:

  * Overlay bars
  * Side-by-side bars
  * Smooth lines (with configurable smoothing window)

## Requirements

* Python 3.8+
* Libraries:

  * streamlit
  * opencv-python
  * pillow
  * matplotlib
  * numpy
  * requests


## 📌 Notes

* For ** Webcam **, make sure your browser has camera permissions enabled.
* On Windows, Python 64-bit is recommended.
* Keep `pip` updated to avoid installation issues.

## 📬 Contact
- **GitHub:** [ARen990](https://github.com/ARen990)
- **Email:** krittimonp28@gmail.com
- **X:** [Aenijin](https://x.com/Aenijin)

