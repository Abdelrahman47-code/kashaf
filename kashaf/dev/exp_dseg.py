"""Examine the 4 GT D segments to find a precise dead-ball signature."""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")

from kashaf.dev import lab

f, sp, pt, mv = lab.load_sig()
N = f["n_frames"]
vis = f["ball_present"]
bv = f["ball_vel"]
n_mov = f["n_moving"]
crowdH = f["crowd_home_2"]
crowdA = f["crowd_away_2"]

GT_D = [(12851, 12989), (15889, 16400), (17380, 17838), (18027, 18389)]
# some live stretches to contrast (from GT spells)
LIVE = [(11624, 12665), (12989, 13704), (13785, 14846), (16400, 17293)]


def show(name, s, e):
    seg = slice(s, e)
    print("%-14s %5d-%5d len=%4d | mv %.2f/%.2f  n_mov %.0f/%.0f  vis %.2f  sp %.2f  "
          "ballvel %.1f  crowdH>0 %.0f%%  crowdA>0 %.0f%%"
          % (name, s, e, e - s,
             np.nanpercentile(mv[seg], 25), np.nanpercentile(mv[seg], 75),
             np.nanpercentile(n_mov[seg], 25), np.nanpercentile(n_mov[seg], 75),
             vis[seg].mean(), sp[seg].mean(),
             np.nanpercentile(bv[seg], 50),
             100 * (crowdH[seg] > 0).mean(), 100 * (crowdA[seg] > 0).mean()))


print("== GT D segments ==")
for s, e in GT_D:
    show("D", s, e)
print("\n== Live spells (contrast) ==")
for s, e in LIVE:
    show("live", s, e)
# the frame right before/after each D to see the boundary
print("\n== boundaries (50 frames) ==")
for s, e in GT_D:
    show("pre", s - 50, s)
    show("post", e, e + 50)
