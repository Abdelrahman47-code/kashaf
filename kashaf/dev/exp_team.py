"""Experiment: which team strategy + D overlay maximizes spellF1?"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")

from evaluation import possession_metrics as pm
from kashaf.dev import lab

f, sp, pt, mv = lab.load_sig()
N = f["n_frames"]


def labels_from(team8, dead):
    lab = np.full(N, "H", dtype="<U1")
    lab[team8 == 1] = "A"
    lab[dead] = "D"
    return lab


def run(name, team8, dead):
    lab_ = labels_from(team8, dead)
    r = lab.score(lab_, name, verbose=False)
    gt_sp = r["n_gt_spells"] if "n_gt_spells" in r else -1
    print("%-26s spellF1=%.4f macroF1=%.4f  shareE=%.3f aliveDeadF1=%.3f"
          % (name, r["spell_f1_mean"], r["macro_f1"], r["share_err"],
             r["alive_dead_f1"]))
    return r


# ---- team strategies ----
# 1) hysteresis + wide majority vote (my lab function)
team_hys = lab.team_signal(f, pt, gap_open=0.8, gap_close=2.4)
for win in (25, 50, 100, 125):
    sm = lab.smooth_majority(team_hys, win)
    lab.score(labels_from(sm, np.zeros(N, bool)),
              "hyst+%d, noD" % win)

print()
# 2) dead overlays on the smoothed team
sm125 = lab.smooth_majority(team_hys, 125)
sm50 = lab.smooth_majority(team_hys, 50)

dead_tag = sp
dead_vel = mv < 1.45
dead_tagvel = sp | (mv < 1.30)

for tag, dead in (("tag", dead_tag), ("vel", dead_vel), ("tag|vel", dead_tagvel)):
    lab.score(labels_from(sm125, dead), "hys+125, D=%s" % tag)

print()
# how many graded frames does each D overlay catch?
graded = np.where(GT := lab.GT_CODES != 3)[0]
for tag, dead in (("none", np.zeros(N, bool)), ("tag", dead_tag),
                  ("vel1.45", dead_vel), ("tag|vel1.30", sp | (mv < 1.30))):
    w = dead[11185:18685]
    print("D-overlay %-10s covers %.0f%% of graded window" % (tag, 100 * w.mean()))
