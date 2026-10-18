"""
Vidéo de démo semi-synthétique : une vraie image (la main qui tient un verre,
tirée de test_vibration.mp4) dans laquelle seule la main tremble, à une
fréquence et une amplitude connues. La vidéo est compressée en mp4.

On mesure ensuite la main et un morceau de mur, comme dans l'appli (touche Z),
et on enregistre un GIF de ce que l'appli affiche.

Usage : python validation/demo_video.py
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import fourier_shift

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from frequency_analyzer import FrequencyAnalyzer  # noqa: E402
from pre_processing import VideoProcessor  # noqa: E402

FREQ, AMP, FPS, N = 5.0, 0.3, 30.0, 300   # 5 Hz : ordre de grandeur d'un tremblement
RESULTS = os.path.join(ROOT, "validation", "results")
VIDEO = os.path.join(RESULTS, f"demo_main_{FREQ:g}Hz_{AMP:g}px.mp4")
GIF = os.path.join(RESULTS, "demo_zone.gif")

HAND = (480, 405, 150, 130)   # zone qui tremble (x, y, w, h) dans l'image 1280x720
ZONE = (495, 420, 120, 100)   # zone mesurée, à l'intérieur (bords du masque adoucis)
WALL = (470, 40, 140, 140)    # mur immobile, sert de témoin


def load_background():
    cap = cv2.VideoCapture(os.path.join(ROOT, "test_vibration.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    _, frame = cap.read()
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (int(w * 720 / h), 720), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((720, 1280, 3), np.float32)
    x0 = (1280 - frame.shape[1]) // 2
    canvas[:, x0:x0 + frame.shape[1]] = frame
    return canvas


def make_video():
    bg = load_background()
    spectra = [np.fft.fft2(bg[..., c]) for c in range(3)]

    # Masque adouci : seule la main bouge, sans bord net
    x, y, w, h = HAND
    mask = np.zeros(bg.shape[:2], np.float32)
    mask[y:y + h, x:x + w] = 1
    mask = cv2.GaussianBlur(mask, (0, 0), 6)[..., None]

    rng = np.random.default_rng(0)
    ux, uy = np.cos(np.radians(20)), np.sin(np.radians(20))
    writer = cv2.VideoWriter(VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    for i in range(N):
        s = AMP * np.sin(2 * np.pi * FREQ * i / FPS)
        moved = np.stack([np.real(np.fft.ifft2(fourier_shift(sp, (s * uy, s * ux))))
                          for sp in spectra], axis=-1)
        img = mask * moved + (1 - mask) * bg + rng.normal(0, 2.0, bg.shape)
        writer.write(np.clip(img, 0, 255).astype(np.uint8))
    writer.release()


def measure(roi):
    cap = cv2.VideoCapture(VIDEO)
    an = FrequencyAnalyzer(cap.get(cv2.CAP_PROP_FPS), window_size=N)
    x, y, w, h = roi
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        an.update(frame[y:y + h, x:x + w])
    return an.analyze()


def make_gif():
    """Rejoue la boucle de l'appli (carte de chaleur + zone) et garde une image sur deux."""
    from main import RadarApp
    app = RadarApp()
    app.analyzer = FrequencyAnalyzer(FPS)
    proc = VideoProcessor()
    cap = cv2.VideoCapture(VIDEO)
    x, y, w, h = ZONE
    images, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        app.analyzer.update(frame[y:y + h, x:x + w])
        if i % 5 == 0:
            app.freq_result = app.analyzer.analyze()
        _, out = proc.get_vibration_map(frame, app.sensitivity)
        if 60 <= i < 240 and i % 3 == 0:
            app.draw_zone_panel(out, ZONE)
            small = cv2.resize(out[:, 380:1280], (450, 360), interpolation=cv2.INTER_AREA)
            images.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).quantize(64))
        i += 1
    images[0].save(GIF, save_all=True, append_images=images[1:], duration=int(3000 / FPS), loop=0)


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    make_video()
    for label, roi in (("main", ZONE), ("mur (témoin)", WALL)):
        r = measure(roi)
        print(f"{label:13s} : f = {r['freq']:.2f} Hz, A = {r['amplitude_px']:.3f} px, "
              f"SNR = {r['snr']:.1f}, vibration nette = {r['reliable']}")
    print(f"vérité : main {FREQ:g} Hz / {AMP:g} px, mur immobile")
    make_gif()
    print("vidéo :", VIDEO, "| gif :", GIF)
