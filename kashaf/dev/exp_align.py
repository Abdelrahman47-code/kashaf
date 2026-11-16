"""Align predicted turnover boundaries to GT possession boundaries.

For each GT possession (start, team), find the predicted spell it maps to and
report the start/end offset in frames. A persistent signed offset => a lag/lead
in my turnover timing. A big spread => merge/split structure error.
"""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")

from evaluation import possession_metrics as pm
from kashaf.dev import lab

f, sp, pt, mv = lab.load_sig()
N = f["n_frames"]
GT = lab.GT_CODES
valid = pm.valid_runs(GT)
gt_spells = pm.possession_spells(GT, valid)

# predicted team (the current best: w75 hyst 1.4/3.0)
team = lab.smooth_majority(lab.team_signal(f, pt, gap_open=1.4, gap_close=3.0), 75)
codes = np.zeros(N, np.uint8)
codes[team == 1] = pm.A
pred_spells = pm.possession_spells(codes, valid)

print("GT possessions (%d) vs mapped pred spell:" % len(gt_spells))
print("gt_start gt_end gt_team | pred_start pred_end pred_team  d_start d_end  IoU")
tot_ds = []
tot_de = []
for (gs, ge, kt) in gt_spells:
    klab = "HA"[kt]
    # find pred spell with max overlap
    best, bo = None, 0
    for (ps, pe, pk) in pred_spells:
        inter = max(0, min(ge, pe) - max(gs, ps))
        if inter > bo:
            bo, best = inter, (ps, pe, pk)
    ps, pe, pk = best
    union = (ge - gs) + (pe - ps) - bo
    iou = bo / union if union else 0
    ds = ps - gs
    de = pe - ge
    tot_ds.append(ds)
    tot_de.append(de)
    same = "=" if pk == kt else "!="
    print("%7d %7d %s    | %7d %7d %s      %7d %7d  %.3f %s"
          % (gs, ge, klab, ps, pe, "HA"[pk], ds, de, iou, same))

tot_ds = np.array(tot_ds)
tot_de = np.array(tot_de)
print("\nstart-offset: mean %+.1f  med %+.1f  (positive = my turnover late)"
      % (tot_ds.mean(), np.median(tot_ds)))
print("end-offset:   mean %+.1f  med %+.1f  (positive = my spell runs long)"
      % (tot_de.mean(), np.median(tot_de)))
print("team mismatches: %d/%d" % (sum(1 for a, b in
      zip(gt_spells, pred_spells) if False), len(gt_spells)))
