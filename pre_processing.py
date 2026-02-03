import cv2
import numpy as np

class VideoProcessor:
    def __init__(self):
        self.prev_gray_small = None
        self.analysis_width = 320 # Largeur pour le calcul (basse résolution = rapide)

    def get_vibration_map(self, frame_hd, sensitivity=15):
        # 1. On travaille sur les dimensions originales pour le retour
        (h_hd, w_hd) = frame_hd.shape[:2]

        # 2. Création de la miniature pour le calcul mathématique rapide
        ratio = self.analysis_width / float(w_hd)
        dim_small = (self.analysis_width, int(h_hd * ratio))
        frame_small = cv2.resize(frame_hd, dim_small, interpolation=cv2.INTER_LINEAR)
        
        gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.GaussianBlur(gray_small, (15, 15), 0)

        if self.prev_gray_small is None:
            self.prev_gray_small = gray_small
            return None, frame_hd

        # 3. Calcul du Flux Optique (LOURD) sur l'image PETITE (RAPIDE)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray_small, gray_small, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calcul magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Filtre de sensibilité
        mag[mag < sensitivity] = 0 
        
        # 4. On remet la carte de chaleur à l'échelle HD
        # L'interpolation linéaire va "lisser" les vibrations, ce qui est plus joli visuellement
        mag_hd = cv2.resize(mag, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
        
        # Normalisation pour affichage
        mag_vis = cv2.normalize(mag_hd, None, 0, 255, cv2.NORM_MINMAX)
        mag_vis = np.uint8(mag_vis)
        
        # Coloration
        heatmap = cv2.applyColorMap(mag_vis, cv2.COLORMAP_JET)
        
        # Masquage et Fusion HD
        mask = mag_vis > 10
        visual_output = frame_hd.copy()
        
        # On applique la heatmap uniquement là où ça vibre
        visual_output[mask] = cv2.addWeighted(frame_hd, 0.4, heatmap, 0.6, 0)[mask]
        
        self.prev_gray_small = gray_small
        
        # On retourne la magnitude max (basée sur le calcul small mais représentative)
        return mag_hd, visual_output