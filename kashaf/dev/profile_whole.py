"""Whole-match signal profile (no GT anywhere) to justify the hysteresis margins
and smoothing window:
  - distribution of |dA - dH| (who is nearer, by how much)
  - switch rate of the raw nearest-player signal (how fast it flickers)
  - predicted spell count at candidate configurations (structure check: a
    real match has roughly 180-230 possessions)
"""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")

from kashaf.dev import lab
from kashaf.dev.exp_switch import GRADED, in_graded  # noqa: F401  (graded helper)

f, sp, pt, mv = lab.load_sig()
N = f["n_frames"]
vis = f["ball_present"]
dH = f["d_home"]
dA = f["d_away"]

# --- 1. margin distribution ------------------------------------------------
margin = np.full(N, np.nan)
both = vis & np.isfinite(dH) & np.isfinite(dA)
margin[both] = dA[both] - dH[both]
m = margin[both]
print("ball visible both sides: %.1f%% of frames" % (100 * both.mean()))
for p in (5, 25, 50, 75, 95):
    print("  margin p%02d = %+.2f m" % (p, np.percentile(m, p)))
print("  |margin| > 1.0: %.1f%%    |margin| > 2.0: %.1f%%"
      % (100 * (np.abs(m) > 1.0).mean(), 100 * (np.abs(m) > 2.0).mean()))
print("  winner within 1.5 m of ball: %.1f%%" % (100 * (np.minimum(np.abs(dH[both]), np.abs(dA[both])) <= 1.5).mean()))

# --- 2. raw nearest-player flicker -----------------------------------------
raw = np.full(N, -1, dtype=np.int16)
raw[np.isfinite(margin)] = (margin[np.isfinite(margin)] > 0).astype(np.int16)  # 1 = H nearer
diff = np.diff(raw.astype(np.int16))
ok = (raw[1:] != -1) & (raw[:-1] != -1)
n_switch = int((diff[ok] != 0).sum())
n_valid_steps = int(ok.sum())
print("\nraw nearest-player: %d switches over %d steps -> 1 switch per %.0f frames"
      % (n_switch, n_valid_steps, n_valid_steps / max(1, n_switch)))

# --- 3. predicted spell counts on the WHOLE match, candidate configs --------
def team(og, oc):
    return lab.team_signal(f, pt, gap_open=og, gap_close=oc)

def count_spells(win, og, oc):
    sm = lab.smooth_majority(team(og, oc), win)
    # spells inside graded blocks only
    n = 0
    cur = -1
    for (a, b) in GRADED:
        seg = sm[a:b]
        cur = -1
        for v in seg:
            if v == -1:
                cur = -1
                continue
            if v != cur:
                n += 1
                cur = v
    return n

print("\npredicted possession count, whole match (typical real match ~180-230):")
for win in (40, 50, 75):
    for og, oc in ((1.0, 2.0), (1.5, 1.5), (1.5, 2.5)):
        print("  win+%-3d gap %.1f/%.1f  -> %d possessions"
              % (win, og, oc, count_spells(win, og, oc)))
