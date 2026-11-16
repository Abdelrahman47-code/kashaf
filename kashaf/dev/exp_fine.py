"""Final fine sweep of (smoothing window, hysteresis gap) on the no-D team spell.
Public window = validation only; I pick a constant for its whole-match shape,
not its exact public-window optimum.
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
    print("%-16s spellF1=%.4f macroF1=%.4f shareE=%.3f nPred=%3d "
          "iou3=%.3f/5=%.3f/7=%.3f" %
          (name, r["spell_f1_mean"], r["macro_f1"], r["share_err"], ss["n_pred"],
           ss["per_iou"][0.3]["f1"], ss["per_iou"][0.5]["f1"], ss["per_iou"][0.7]["f1"]))


f, sp, pt, mv = lab.load_sig()
lab._f, lab._pt, lab._N = f, pt, f["n_frames"]
N = f["n_frames"]
noD = np.zeros(N, bool)

print("== fine grid: window x gap (no D) ==")
for win in (75, 100, 125):
    for og, oc in ((1.4, 3.0), (1.6, 3.6), (1.8, 3.8), (2.0, 4.0), (2.4, 4.4)):
        report(build(win, og, oc, noD), "w%d g%.1f/%.1f" % (win, og, oc))
    print()

print("== best constant sanity: how many turnovers vs GT (17 spells) ==")
for win, og, oc in ((100, 2.0, 4.0), (100, 1.8, 3.8), (75, 1.6, 3.6)):
    l = build(win, og, oc, noD)
    codes = np.zeros(N, np.uint8)
    codes[l == "A"] = pm.A
    spell = pm.possession_spells(codes, pm.valid_runs(lab.GT_CODES))
    durs = [e - s for s, e, k in spell]
    print("w%d g%.1f/%.1f: %d spells, avg dur %d, min %d max %d"
          % (win, og, oc, len(spell), int(np.mean(durs)), min(durs), max(durs)))
