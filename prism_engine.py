import numpy as np
from scipy.signal import detrend
from scipy.fft import fft, fftfreq

class PrismEngine:
    def __init__(self, fps, mode='health'):
        self.fps = fps
        self.mode = mode
        self.prev_freq = None
        # Bandes de fréquences cibles
        self.freq_range = (0.7, 3.0) if mode == 'health' else (5.0, 50.0)

    def _calculate_metrics(self, signal):
        n = len(signal)
        yf = np.abs(fft(signal))
        xf = fftfreq(n, 1/self.fps)
        
        mask = (xf >= self.freq_range[0]) & (xf <= self.freq_range[1])
        if not np.any(mask): return 0, 0, 0
        
        peak_idx = np.argmax(yf[mask])
        freq = xf[mask][peak_idx]
        
        # C (Concentration Spectrale) : Pureté du signal
        concentration = np.max(yf[mask]) / (np.sum(yf[mask]) + 1e-6)
        
        # TV (Temporal Variation) : Stabilité temporelle
        tv_penalty = abs(freq - self.prev_freq) if self.prev_freq is not None else 0
            
        return concentration, tv_penalty, freq

    def optimize(self, rgb_buffer):
        data = np.array(rgb_buffer).T
        # Normalisation pour annuler les variations globales d'intensité
        data_norm = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-6)
        
        best_score = -np.inf
        best_freq = 0
        
        # Recherche de l'alpha optimal (Secret Sauce PRISM)
        for alpha in np.linspace(0.5, 1.0, 15):
            s = data_norm[1] - (alpha * data_norm[2] + (1 - alpha) * data_norm[0])
            s = detrend(s)
            
            conc, tv, freq = self._calculate_metrics(s)
            score = conc - (0.1 * tv) # Fonction de coût L
            
            if score > best_score:
                best_score = score
                best_freq = freq
        
        self.prev_freq = best_freq
        return best_freq, best_score