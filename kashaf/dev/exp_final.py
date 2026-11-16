"""Sweep smoothing window + hysteresis on the no-D team spell, and size the
prize of perfect D (oracle) to decide whether D is worth pursuing at all.

Public window is validation-only: I am NOT picking the winning window here --
I am checking the curve shape so I choose a defensible whole-match constant.
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


def build(win, og, oc, dead_):
    team = lab.team_signal(lab._f, lab._pt, gap_open=og, gap_close=oc)
    sm = lab.smooth_majority(team, win)
    N = lab._N
    l = np.full(N, "H", dtype="<U1")
    l[sm == 1] = "A"
    l[dead_] = "D"
    return l


def report(l, name):
    N = lab._N
    codes = np.zeros(N, np.uint8)
    codes[l == "A"] = pm.A
    codes[l == "D"] = pm.D
    gt = lab.GT_CODES
    ss = pm.spell_scores(gt, pm.fill_abstentions(codes, pm.valid_runs(gt)),
                         pm.valid_runs(gt))
    r = lab.score(l, name, verbose=False)
    print("%-22s spellF1=%.4f macroF1=%.4f shareE=%.3f nPred=%3d  "
          "iou3=%.3f/5=%.3f/7=%.3f" %
          (name, r["spell_f1_mean"], r["macro_f1"], r["share_err"], ss["n_pred"],
           ss["per_iou"][0.3]["f1"], ss["per_iou"][0.5]["f1"], ss["per_iou"][0.7]["f1"]))


f, sp, pt, mv = lab.load_sig()
lab._f, lab._pt, lab._N = f, pt, f["n_frames"]
N = f["n_frames"]
noD = np.zeros(N, bool)

print("== no-D, sweep smoothing window (hyst 0.8/2.4) ==")
for win in (50, 75, 100, 125, 150, 175, 200):
    report(build(win, 0.8, 2.4, noD), "w%d" % win)

print("\n== no-D, sweep hysteresis gap (w100) ==")
for og, oc in ((0.4, 1.6), (0.8, 2.4), (1.2, 3.0), (1.6, 3.6), (2.0, 4.0)):
    report(build(100, og, oc, noD), "gap %.1f/%.1f" % (og, oc))

print("\n== D upper bound: perfect GT-D, no-D team elsewhere ==")
import pandas as _pd
g = _pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
oracle = np.zeros(N, bool)
for fr, lb in zip(g["frame"], g["label"]):
    if lb == "D":
        oracle[int(fr)] = True
report(build(100, 0.8, 2.4, oracle), "w100 D=oracle")
