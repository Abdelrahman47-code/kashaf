"""Diagnose D overlays: how they interact with live spells."""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")

from evaluation import possession_metrics as pm
from kashaf.dev import lab
from kashaf.dev.exp_team import labels_from  # type: ignore

f, sp, pt, mv = lab.load_sig()
N = f["n_frames"]
team_hys = lab.team_signal(f, pt, gap_open=0.8, gap_close=2.4)

gt_spells = pm.possession_spells(
    lab.GT_CODES, pm.valid_runs(lab.GT_CODES))
print("GT spells (start,len,team):")
for a, b, k in gt_spells:
    print("  %6d %5d %s" % (a, b - a, "HA"[k]))

sm100 = lab.smooth_majority(team_hys, 100)
base = labels_from(sm100, np.zeros(N, bool))
codes = np.zeros(N, dtype=np.uint8)
codes[base == "A"] = pm.A
codes[base == "D"] = pm.D
pred_spells = pm.possession_spells(codes, pm.valid_runs(lab.GT_CODES))
print("pred spells (hyst+100, no D): n=%d, total live frames %d"
      % (len(pred_spells), int(np.sum(base != "D"))))
for a, b, k in pred_spells[:40]:
    print("  %6d %5d %s" % (a, b - a, "HA"[k]))
