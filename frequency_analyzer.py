import cv2
import numpy as np
from collections import deque
from scipy.signal import detrend


class FrequencyAnalyzer:
    """Mesure la fréquence et l'amplitude de vibration d'une zone de l'image.

    Déplacement sous-pixel par recalage ECC (translation), puis FFT du signal.
    Attention : rien n'est visible au-delà de fps / 2 (Nyquist).
    """

    def __init__(self, fps, window_size=256, min_freq=1.0, min_snr=6.0, min_amplitude=0.02):
        self.fps = float(fps)
        self.window_size = window_size
        self.min_freq = min_freq
        self.min_snr = min_snr
        # 0.02 px : le double de la dérive mesurée sur un mur immobile (compression mp4)
        self.min_amplitude = min_amplitude
        self.reset()

    @property
    def nyquist(self):
        return self.fps / 2.0

    def reset(self):
        self.ref = None
        self.warp = np.eye(2, 3, dtype=np.float32)
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
        self.dx = deque(maxlen=self.window_size)
        self.dy = deque(maxlen=self.window_size)
        self.failures = 0

    def update(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        img = np.float32(gray)

        if self.ref is None or img.shape != self.ref.shape:
            self.reset()
            self.ref = img

        # On repart du déplacement précédent, le mouvement est continu
        try:
            _, self.warp = cv2.findTransformECC(self.ref, img, self.warp,
                                                cv2.MOTION_TRANSLATION, self.criteria, None, 1)
        except cv2.error:
            self.failures += 1  # pas de convergence : on garde la dernière valeur

        sx, sy = float(self.warp[0, 2]), float(self.warp[1, 2])
        self.dx.append(sx)
        self.dy.append(sy)
        return sx, sy

    def analyze(self):
        n = len(self.dx)
        if n < 32:
            return None

        d = np.column_stack([self.dx, self.dy])
        d = d - d.mean(axis=0)

        # On projette sur l'axe où ça bouge le plus
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        signal = detrend(d @ vt[0])

        win = np.hanning(n)
        nfft = max(4096, 1 << (n - 1).bit_length())
        spec = np.abs(np.fft.rfft(signal * win, nfft))
        freqs = np.fft.rfftfreq(nfft, 1.0 / self.fps)

        band = np.where(freqs >= self.min_freq)[0]
        if band.size < 3:
            return None
        k = band[np.argmax(spec[band])]

        # Interpolation parabolique pour affiner le pic
        freq = freqs[k]
        if 0 < k < len(spec) - 1:
            a, b, c = spec[k - 1], spec[k], spec[k + 1]
            if a - 2 * b + c != 0:
                freq += 0.5 * (a - c) / (a - 2 * b + c) * (freqs[1] - freqs[0])

        # Pour A*sin(2*pi*f*t), le pic vaut A * somme(fenêtre) / 2
        amplitude = 2.0 * spec[k] / win.sum()
        snr = spec[k] / (np.median(spec[band]) + 1e-12)

        # Un pic collé au bas de la bande, c'est une dérive, pas une vibration
        reliable = (snr >= self.min_snr and amplitude >= self.min_amplitude
                    and k > band[0] + 1)

        return {
            "freq": float(freq),
            "amplitude_px": float(amplitude),
            "snr": float(snr),
            "reliable": bool(reliable),
            "signal": signal,
            "freqs": freqs,
            "spectrum": spec,
            "nyquist": self.nyquist,
        }
