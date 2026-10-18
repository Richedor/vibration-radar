# python -m unittest discover tests
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frequency_analyzer import FrequencyAnalyzer  # noqa: E402
from synthetic import vibrating_frames  # noqa: E402


def measure(freq, amp, fps, n=256, noise=2.0):
    an = FrequencyAnalyzer(fps, window_size=n)
    for f in vibrating_frames(freq, amp, fps, n, noise_std=noise):
        an.update(f)
    return an.analyze()


class TestFrequency(unittest.TestCase):
    def test_frequence_et_amplitude(self):
        r = measure(7.0, 0.5, 30)
        self.assertAlmostEqual(r["freq"], 7.0, delta=0.1)
        self.assertAlmostEqual(r["amplitude_px"], 0.5, delta=0.05)
        self.assertTrue(r["reliable"])

    def test_sous_pixel(self):
        r = measure(5.0, 0.1, 30)
        self.assertAlmostEqual(r["freq"], 5.0, delta=0.1)
        self.assertAlmostEqual(r["amplitude_px"], 0.1, delta=0.02)

    def test_repliement_au_dela_de_nyquist(self):
        # 20 Hz filmé à 30 fps : on voit 30 - 20 = 10 Hz
        r = measure(20.0, 0.5, 30)
        self.assertAlmostEqual(r["freq"], 10.0, delta=0.1)

    def test_camera_rapide(self):
        # à 120 fps on retrouve la vraie fréquence
        r = measure(20.0, 0.5, 120)
        self.assertAlmostEqual(r["freq"], 20.0, delta=0.2)

    def test_image_immobile_non_fiable(self):
        r = measure(5.0, 0.0, 30)
        self.assertFalse(r["reliable"])


if __name__ == "__main__":
    unittest.main()
