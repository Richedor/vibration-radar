# Vibration Radar Pro

> **See the unseen.** A real-time computer vision tool that reveals invisible mechanical vibrations using standard video feeds.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![Status](https://img.shields.io/badge/Status-Stable-success?style=for-the-badge)

---

## Demo

*(Place your recorded GIF or screenshot here to show the tool in action)*
![Demo Vibration](https://via.placeholder.com/800x400?text=Insert+Your+Demo+GIF+Here)

## What is it?

Vibration Radar is a **non-contact metrology tool**. Unlike physical accelerometers that need to be glued to machinery, this software uses **Motion Magnification** principles to detect micro-movements from a distance.

Whether you are inspecting a bridge, checking a motor's alignment, or analyzing structural integrity, this tool turns your webcam into a precision sensor.

## Key features

*  Real-Time CPU Optimization: Runs at 30+ FPS on standard processors thanks to a **Gaussian Pyramid** downsampling architecture. No expensive GPU required.
*  Farnebäck Optical Flow: Uses dense vector field estimation to track pixel-level displacements.
*  Dynamic Heatmap: visualizes vibration intensity using a JET colormap overlay.
*  HD HUD Interface: Professional heads-up display with clean, anti-aliased text (Pillow rendering) over a high-definition video feed.
*  Session Recording: Built-in video recorder to archive your inspections with data overlays (FPS, Sensitivity, Intensity).

## Technology stack

This project is built with **Python** and relies on a deterministic Computer Vision pipeline:

1.  **Preprocessing:** Spatial Low-Pass Filtering (Denoising).
2.  **Optimization:** Image Pyramid Downscaling.
3.  **Estimation:** Polynomial Expansion (Farnebäck Algorithm).
4.  **Filtering:** Magnitude Gating & Thresholding.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Richedor/vibration-radar.git](https://github.com/Richedor/vibration-radar.git)
    cd vibration-radar
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to use

Run the main script:
```bash
python main.py
```

### Controls

| Key | Action |
| :--- | :--- |
| **`+`** | Increase Sensitivity (Detect smaller vibrations) |
| **`-`** | Decrease Sensitivity (Filter out noise) |
| **`ESC`** | Quit the application |

---

## Author

**© 2026 richedor**
*Electronics & Computer Vision Enthusiast*

> *"Engineering is about making the invisible, visible."*
