import cv2
import numpy as np
import time
import datetime
from collections import deque
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont
from pre_processing import VideoProcessor
from frequency_analyzer import FrequencyAnalyzer

class RadarApp:
    def __init__(self):
        self.proc = VideoProcessor()
        self.sensitivity = 4
        self.project_name = "PROJET : DETECTION DE VIBRATION"
        self.copyright = "© 2026 richedor"
        self.source = None
        self.record_session = False # État de la case à cocher

        # --- MESURE SUR UNE ZONE (touche Z) ---
        self.roi_raw = None      # (x, y, w, h) dans l'image d'origine
        self.analyzer = None
        self.freq_result = None
        
        # --- POLICES HD ---
        try:
            self.font_header = ImageFont.truetype("arialbd.ttf", 18)
            self.font_info = ImageFont.truetype("arial.ttf", 14)
            self.font_fps = ImageFont.truetype("arial.ttf", 12)
        except:
            self.font_header = ImageFont.load_default()
            self.font_info = ImageFont.load_default()
            self.font_fps = ImageFont.load_default()

        # --- COULEURS ---
        self.COL_YELLOW = (255, 215, 0)
        self.COL_ORANGE = (255, 140, 0)
        self.COL_WHITE = (255, 255, 255)
        self.COL_GREY = (180, 180, 180)

    # --- PARTIE 1 : LE LANCEUR (INTERFACE DE DÉPART) ---
    def start_launcher(self):
        self.root = tk.Tk()
        self.root.title("Richedor - Configuration")
        self.root.geometry("350x250")
        
        # Titre
        tk.Label(self.root, text="Radar de Vibration", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Case à cocher pour l'enregistrement
        self.chk_var = tk.BooleanVar()
        chk = tk.Checkbutton(self.root, text="Enregistrer la vidéo de sortie (avec textes)", 
                             var=self.chk_var, font=("Arial", 10))
        chk.pack(pady=10)

        # Boutons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Utiliser Webcam", command=self.set_webcam, 
                  width=20, bg="#dddddd").pack(pady=5)
        tk.Button(btn_frame, text="Charger Fichier Vidéo", command=self.set_file, 
                  width=20, bg="#dddddd").pack(pady=5)

        self.root.mainloop()

    def set_webcam(self):
        self.source = 0
        self.record_session = self.chk_var.get()
        self.root.destroy()
        self.run_engine()

    def set_file(self):
        f = filedialog.askopenfilename(title="Choisir une vidéo")
        if f:
            self.source = f
            self.record_session = self.chk_var.get()
            self.root.destroy()
            self.run_engine()

    # --- PARTIE 2 : MOTEUR VIDÉO ---
    def resize_frame(self, frame, target_width=1280):
        (h, w) = frame.shape[:2]
        ratio = target_width / float(w)
        dim = (target_width, int(h * ratio))
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def draw_text_hd(self, img, text, pos, font, color):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        x, y = pos
        draw.text((x+1, y+1), text, font=font, fill=(0, 0, 0)) # Ombre
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_zone_panel(self, img, roi_hd):
        x, y, w, h = roi_hd
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        H, W = img.shape[:2]
        pw, ph = 380, 150
        x0, y0 = W - pw - 15, H - ph - 40
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (20, 20, 20), -1)
        img[:] = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)

        r = self.freq_result
        nyq = self.analyzer.nyquist
        if r is None:
            lines = ["Zone : acquisition...", f"Nyquist : {nyq:.1f} Hz"]
        elif r["reliable"]:
            lines = [f"FRÉQUENCE : {r['freq']:.2f} Hz",
                     f"AMPLITUDE : {r['amplitude_px']:.3f} px   SNR {r['snr']:.0f}",
                     f"Au-delà de {nyq:.1f} Hz : fréquence repliée"]
        else:
            lines = ["Pas de vibration nette", f"SNR {r['snr']:.1f} (seuil {self.analyzer.min_snr:.0f})",
                     f"Nyquist : {nyq:.1f} Hz"]

        # Mini spectre
        if r is not None:
            fr, sp = r["freqs"], r["spectrum"]
            sx0, sy0, sw, sh = x0 + 10, y0 + 75, pw - 20, 65
            sp_n = sp / (sp.max() + 1e-12)
            pts = np.stack([sx0 + fr / nyq * sw, sy0 + sh - sp_n * sh], axis=1).astype(np.int32)
            cv2.polylines(img, [pts], False, (0, 200, 255), 1, cv2.LINE_AA)
            if r["reliable"]:
                px = int(sx0 + r["freq"] / nyq * sw)
                cv2.line(img, (px, sy0), (px, sy0 + sh), (0, 0, 255), 1)

        for i, t in enumerate(lines):
            img[:] = self.draw_text_hd(img, t, (x0 + 10, y0 + 8 + i * 22), self.font_info, self.COL_WHITE)

    def run_engine(self):
        if self.source is None: return

        cap = cv2.VideoCapture(self.source)
        is_file = isinstance(self.source, str)
        window_name = "Radar de Vibration Pro (REC)" if self.record_session else "Radar de Vibration Pro"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Initialisation de l'enregistrement
        video_writer = None
        if self.record_session:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"Vibration_Report_{timestamp}.mp4"
            print(f"--- ENREGISTREMENT ACTIVÉ : {filename} ---")
            # Le writer sera créé à la première frame pour avoir la bonne taille

        prev_time = 0
        # Base de temps : fps du fichier, ou fps mesurés pour la webcam
        native_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0
        if not native_fps or native_fps <= 0:
            native_fps = 30.0
        stamps = deque(maxlen=256)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.proc.prev_gray_small = None
                    if self.analyzer: self.analyzer.reset()
                    self.freq_result = None
                    continue
                else: break

            raw = frame
            stamps.append(time.time())
            frame_idx += 1

            # 1. Force HD (1280px)
            frame = self.resize_frame(frame, target_width=1280)
            scale = raw.shape[1] / float(frame.shape[1])

            # 1b. Mesure sur la zone, en pleine résolution
            if self.roi_raw is not None:
                x, y, w, h = self.roi_raw
                self.analyzer.update(raw[y:y + h, x:x + w])
                if not is_file and len(stamps) > 10:
                    self.analyzer.fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
                if frame_idx % 5 == 0:
                    self.freq_result = self.analyzer.analyze()

            # 2. Analyse
            mag, visual_output = self.proc.get_vibration_map(frame, self.sensitivity)

            # 3. Calcul FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            if mag is not None:
                h, w = visual_output.shape[:2]
                max_vib = np.max(mag)

                # --- DESSIN DU HUD (Ce qui sera enregistré) ---
                overlay = visual_output.copy()
                cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
                visual_output = cv2.addWeighted(overlay, 0.85, visual_output, 0.15, 0)

                # Textes
                visual_output = self.draw_text_hd(visual_output, self.project_name, (15, 15), self.font_header, self.COL_WHITE)
                
                sens_txt = f"SENSIBILITE (+/-) : {self.sensitivity}"
                visual_output = self.draw_text_hd(visual_output, sens_txt, (380, 15), self.font_header, self.COL_YELLOW)

                val_color = self.COL_ORANGE if max_vib < 80 else (50, 50, 255)
                vib_txt = f"VIBRATION MAX : {int(max_vib)}"
                visual_output = self.draw_text_hd(visual_output, vib_txt, (w - 280, 15), self.font_header, val_color)

                visual_output = self.draw_text_hd(visual_output, self.copyright, (15, h - 25), self.font_info, self.COL_GREY)
                visual_output = self.draw_text_hd(visual_output, f"FPS: {int(fps)}", (w - 80, 60), self.font_fps, (0, 255, 0))

                if self.roi_raw is not None:
                    x, y, w, h = self.roi_raw
                    roi_hd = tuple(int(v / scale) for v in (x, y, w, h))
                    self.draw_zone_panel(visual_output, roi_hd)
                else:
                    visual_output = self.draw_text_hd(visual_output, "Z : mesurer une zone", (15, 60), self.font_info, self.COL_GREY)

                if self.record_session:
                    # Point rouge clignotant pour indiquer l'enregistrement
                    if int(curr_time * 2) % 2 == 0:
                        cv2.circle(visual_output, (w - 30, 25), 8, (0, 0, 255), -1)

            # 4. ENREGISTREMENT DISQUE
            if self.record_session:
                if video_writer is None:
                    # Initialisation au premier tour (pour avoir width/height corrects)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (visual_output.shape[1], visual_output.shape[0]))
                
                video_writer.write(visual_output)

            cv2.imshow(window_name, visual_output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key == ord('+'): self.sensitivity += 2
            elif key == ord('-'): self.sensitivity = max(1, self.sensitivity - 2)
            elif key in (ord('z'), ord('Z')):
                sel = cv2.selectROI(window_name, frame, showCrosshair=True)
                if sel[2] > 16 and sel[3] > 16:
                    self.roi_raw = tuple(int(v * scale) for v in sel)
                    self.analyzer = FrequencyAnalyzer(native_fps)
                    self.freq_result = None
            elif key in (ord('c'), ord('C')):
                self.roi_raw, self.analyzer, self.freq_result = None, None, None

        if video_writer:
            video_writer.release()
            print(f"Vidéo sauvegardée sous : {filename}")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = RadarApp()
    app.start_launcher()