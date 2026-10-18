import cv2
import numpy as np
from scipy.ndimage import fourier_shift


def make_texture(size=160, seed=0):
    rng = np.random.default_rng(seed)
    tex = rng.uniform(0, 255, (size * 2, size * 2)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (0, 0), 2.0)
    return cv2.normalize(tex, None, 20, 235, cv2.NORM_MINMAX)


def vibrating_frames(freq, amplitude_px, fps, n_frames, angle_deg=30.0,
                     noise_std=2.0, size=160, seed=0):
    """Texture déplacée de amplitude_px * sin(2*pi*freq*t) dans une direction donnée.

    Le décalage est fait dans le domaine de Fourier (exact au sous-pixel).
    cv2.warpAffine fausse les petits décalages (~+40 % à 0.1 px), d'où ce choix.
    """
    tex_f = np.fft.fft2(make_texture(size, seed))
    rng = np.random.default_rng(seed + 1)
    ux, uy = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    a = size // 2
    frames = []
    for i in range(n_frames):
        s = amplitude_px * np.sin(2 * np.pi * freq * i / fps)
        img = np.real(np.fft.ifft2(fourier_shift(tex_f, (s * uy, s * ux))))
        img = img[a:a + size, a:a + size] + rng.normal(0, noise_std, (size, size))
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames
