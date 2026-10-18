"""
Validation sur vidéos synthétiques (on connaît la vraie fréquence et la vraie amplitude).

E1  précision selon l'amplitude : Farnebäck (réglages d'origine), corrélation de phase, ECC
E2  plus petite amplitude détectée, selon le bruit
E3  repliement au-delà de Nyquist (30 fps)
E4  éclairage qui scintille à 10 Hz (néon 100 Hz filmé à 30 fps)

5 répétitions par cas. Résultats (csv + figure) dans validation/results/.
Lancer depuis la racine : python validation/run_study.py  (~10 min)
"""
import csv
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]
from frequency_analyzer import FrequencyAnalyzer  # noqa: E402
from synthetic import vibrating_frames  # noqa: E402

OUT = os.path.join(ROOT, "validation", "results")
FPS, N, FREQ = 30.0, 256, 7.0
SEEDS = range(5)


# --- Estimation du déplacement (par rapport à la première image) ---

def track_ecc(frames):
    an = FrequencyAnalyzer(FPS, window_size=len(frames))
    for f in frames:
        an.update(f)
    return np.array(an.dx), np.array(an.dy)


def track_phase(frames):
    from skimage.registration import phase_cross_correlation
    ref = np.float32(frames[0])
    # skimage renvoie le décalage pour recaler, donc le signe opposé
    d = np.array([phase_cross_correlation(ref, np.float32(f), upsample_factor=100)[0]
                  for f in frames])
    return -d[:, 1], -d[:, 0]


def track_farneback(frames):
    # Mêmes réglages que pre_processing.py, flux moyen de la zone
    ref = cv2.GaussianBlur(frames[0], (15, 15), 0)
    dx, dy = [], []
    for f in frames:
        g = cv2.GaussianBlur(f, (15, 15), 0)
        flow = cv2.calcOpticalFlowFarneback(ref, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx.append(flow[..., 0].mean())
        dy.append(flow[..., 1].mean())
    return np.array(dx), np.array(dy)


METHODS = {"Farneback (origine)": track_farneback,
           "Correlation de phase": track_phase,
           "ECC (nouveau)": track_ecc}


def spectral_estimate(dx, dy, fps=FPS):
    an = FrequencyAnalyzer(fps, window_size=len(dx))
    an.dx.extend(dx)
    an.dy.extend(dy)
    return an.analyze()


# --- E1 : précision ---

def experiment_precision():
    amps = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    rows = []
    for amp in amps:
        for seed in SEEDS:
            frames = vibrating_frames(FREQ, amp, FPS, N, noise_std=2.0, seed=seed)
            truth = amp * np.sin(2 * np.pi * FREQ * np.arange(N) / FPS)
            tx, ty = truth * np.cos(np.radians(30)), truth * np.sin(np.radians(30))
            for name, fn in METHODS.items():
                dx, dy = fn(frames)
                r = spectral_estimate(dx, dy)
                rows.append({"methode": name, "amplitude_px": amp, "seed": seed,
                             "err_depl_rms_px": float(np.sqrt(np.mean((dx - tx) ** 2 + (dy - ty) ** 2))),
                             "freq_mesuree": r["freq"], "ampl_mesuree": r["amplitude_px"],
                             "err_freq_hz": abs(r["freq"] - FREQ),
                             "err_ampl_rel": abs(r["amplitude_px"] - amp) / amp})
    return rows


# --- E2 : limite de détection ---

def experiment_detection_limit():
    amps = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    noises = [1.0, 2.0, 5.0, 10.0]
    rows = []
    for noise in noises:
        for amp in amps:
            ok = 0
            for seed in SEEDS:
                frames = vibrating_frames(FREQ, amp, FPS, N, noise_std=noise, seed=seed)
                r = spectral_estimate(*track_ecc(frames))
                ok += abs(r["freq"] - FREQ) < 0.1
            rows.append({"bruit_std": noise, "amplitude_px": amp, "detections_sur_5": ok})
    return rows


# --- E3 : repliement ---

def experiment_aliasing():
    rows = []
    for f_true in np.arange(2.0, 29.0, 2.0):
        frames = vibrating_frames(f_true, 0.5, FPS, N, noise_std=2.0)
        r = spectral_estimate(*track_ecc(frames))
        rows.append({"freq_reelle": f_true, "freq_mesuree": r["freq"],
                     "freq_attendue_repliee": abs(f_true - FPS * round(f_true / FPS))})
    return rows


# --- E4 : éclairage ---

def experiment_illumination():
    depths = [0.0, 0.01, 0.02, 0.05, 0.1]
    f_light, amp = 10.0, 0.1
    rows = []
    for depth in depths:
        for seed in SEEDS:
            frames = vibrating_frames(FREQ, amp, FPS, N, noise_std=2.0, seed=seed)
            w = frames[0].shape[1]
            side = np.linspace(0.0, 1.0, w)[None, :]  # lumière qui vient de la droite
            lit = []
            for i, f in enumerate(frames):
                gain = 1.0 + depth * np.sin(2 * np.pi * f_light * i / FPS) * side
                lit.append(np.clip(f * gain, 0, 255).astype(np.uint8))
            for name, fn in METHODS.items():
                r = spectral_estimate(*fn(lit))
                rows.append({"methode": name, "scintillement": depth, "seed": seed,
                             "freq_mesuree": r["freq"],
                             "vibration_trouvee": bool(abs(r["freq"] - FREQ) < 0.1),
                             "err_ampl_rel": abs(r["amplitude_px"] - amp) / amp})
    return rows


# --- Sorties ---

def save_csv(rows, name):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize_precision(rows):
    out = []
    for name in METHODS:
        for amp in sorted({r["amplitude_px"] for r in rows}):
            sel = [r for r in rows if r["methode"] == name and r["amplitude_px"] == amp]
            ef = np.array([r["err_freq_hz"] for r in sel])
            ea = np.array([r["err_ampl_rel"] for r in sel]) * 100
            ed = np.array([r["err_depl_rms_px"] for r in sel])
            out.append((name, amp, ef.mean(), ef.std(), ea.mean(), ea.std(),
                        int((ef < 0.1).sum()), ed.mean(), ed.std()))
    return out


def plot_all(prec_summary, det, alias, illum):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    axes = axes.ravel()

    ax = axes[0]
    for name in METHODS:
        s = [p for p in prec_summary if p[0] == name]
        ax.errorbar([p[1] for p in s], [p[7] for p in s], yerr=[p[8] for p in s],
                    marker="o", capsize=3, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Amplitude reelle (px)"); ax.set_ylabel("Erreur RMS du deplacement (px)")
    ax.set_title("E1. Precision (7 Hz, bruit 2, 5 repetitions)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for noise in sorted({r["bruit_std"] for r in det}):
        s = [r for r in det if r["bruit_std"] == noise]
        ax.plot([r["amplitude_px"] for r in s], [r["detections_sur_5"] for r in s],
                marker="o", label=f"bruit sigma = {noise:g}")
    ax.set_xscale("log"); ax.set_ylim(-0.3, 5.3)
    ax.axhline(4, ls="--", c="grey", lw=1)
    ax.set_xlabel("Amplitude reelle (px)"); ax.set_ylabel("Frequence retrouvee (sur 5)")
    ax.set_title("E2. Limite de detection (ECC)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    fr = [r["freq_reelle"] for r in alias]
    ax.plot(fr, fr, ls=":", c="grey", label="identite")
    ax.plot(fr, [r["freq_attendue_repliee"] for r in alias], c="tab:orange", label="theorie (repliement)")
    ax.plot(fr, [r["freq_mesuree"] for r in alias], "o", c="tab:blue", label="mesure")
    ax.axvline(FPS / 2, ls="--", c="red", lw=1); ax.text(FPS / 2 + 0.3, 1, "Nyquist 15 Hz", color="red")
    ax.set_xlabel("Frequence reelle (Hz)"); ax.set_ylabel("Frequence mesuree (Hz)")
    ax.set_title("E3. Repliement a 30 i/s")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[3]
    for name in METHODS:
        s = [r for r in illum if r["methode"] == name]
        depths = sorted({r["scintillement"] for r in s})
        ok = [sum(r["vibration_trouvee"] for r in s if r["scintillement"] == d) for d in depths]
        ax.plot([d * 100 for d in depths], ok, marker="o", label=name)
    ax.set_ylim(-0.3, 5.3)
    ax.set_xlabel("Profondeur du scintillement (%)"); ax.set_ylabel("Vibration 7 Hz retrouvee (sur 5)")
    ax.set_title("E4. Eclairage variable (scintillement 10 Hz, vibration 0,1 px)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "validation.png"), dpi=130)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    prec = experiment_precision(); save_csv(prec, "e1_precision.csv")
    det = experiment_detection_limit(); save_csv(det, "e2_limite_detection.csv")
    alias = experiment_aliasing(); save_csv(alias, "e3_repliement.csv")
    illum = experiment_illumination(); save_csv(illum, "e4_eclairage.csv")
    summary = summarize_precision(prec)
    plot_all(summary, det, alias, illum)

    print("E1  methode | ampl px | err depl RMS px | err freq Hz | err ampl % (moy+-ec) | freq ok /5")
    for s in summary:
        print(f"{s[0]:22s} | {s[1]:5.2f} | {s[7]:.4f} +- {s[8]:.4f} | {s[2]:.3f} | {s[4]:6.1f} +- {s[5]:5.1f} | {s[6]}")
    print("\nE2  bruit | ampl px | detections /5")
    for r in det:
        print(f"{r['bruit_std']:5.1f} | {r['amplitude_px']:6.3f} | {r['detections_sur_5']}")
    print("\nE3  f reelle | f mesuree | f theorique")
    for r in alias:
        print(f"{r['freq_reelle']:5.1f} | {r['freq_mesuree']:6.2f} | {r['freq_attendue_repliee']:5.1f}")
    print("\nE4  methode | scintillement % | vibration 7 Hz trouvee /5 | freq mesurees")
    for name in METHODS:
        for d in sorted({r["scintillement"] for r in illum}):
            sel = [r for r in illum if r["methode"] == name and r["scintillement"] == d]
            fq = ", ".join(f"{r['freq_mesuree']:.1f}" for r in sel)
            print(f"{name:22s} | {d * 100:4.0f} | {sum(r['vibration_trouvee'] for r in sel)} | {fq}")
    print(f"\nDuree : {time.time() - t0:.0f} s. Resultats dans {OUT}")


if __name__ == "__main__":
    main()
