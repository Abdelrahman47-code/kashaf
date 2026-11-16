"""Boundary-timing experiment: a state machine that switches side once the other
side is clearly nearer for k consecutive frames. A centred majority vote lags
every transition by half its window; a switch-after-k rule localises the
turnover within ~k frames, which is what spell IoU rewards.
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
vis = f["ball_present"]
dH = f["d_home"]
dA = f["d_away"]
GF = np.arange(0, N)

GRADED = [(11185, 90652), (110808, 190198)]


def in_graded(fr):
    return (11185 <= fr < 90652) or (110808 <= fr < 190198)


def switch_machine(enter, exit_, k):
    """team per frame (0/1, -1 no info). Switch after k consecutive frames of a
    clear nearness margin. Resets to -1 at the start of each graded block (the
    first visible frame after kickoff sets the state; pre-block carry would
    poison the first possession)."""
    team = np.full(N, -1, dtype=np.int8)
    cur = -1
    acc_side = -1
    acc = 0
    for i in range(N):
        if not in_graded(i):
            team[i] = -1
            cur = -1
            acc_side, acc = -1, 0
            continue
        if not vis[i] or (np.isnan(dH[i]) and np.isnan(dA[i])):
            team[i] = cur if cur >= 0 else -1
            continue
        dh = dH[i] if np.isfinite(dH[i]) else 99.0
        da = dA[i] if np.isfinite(dA[i]) else 99.0
        side = 0 if dh < da else 1
        gap = abs(dh - da)
        if cur < 0:
            cur = side
        else:
            if side != cur and gap > enter:
                if acc_side == side:
                    acc += 1
                else:
                    acc_side, acc = side, 1
                if acc >= k:
                    cur = side
                    acc_side, acc = side, 0
        team[i] = cur
    return team


def to_labels(team):
    """H/A inside graded blocks; X everywhere else. Inside a graded block with
    no tracked ball, carry the last known side (team_signal already did)."""
    lab_ = np.full(N, "X", dtype="<U1")
    for (a, b) in GRADED:
        m = (GF >= a) & (GF < b)
        lab_[m & (team == 0)] = "H"
        lab_[m & (team == 1)] = "A"
        lab_[m & (team == -1)] = "H"  # no info inside graded: keep live, side of last known handled above
    return lab_


GT = lab.GT_CODES


def report(team, name):
    l = to_labels(team)
    codes = np.zeros(N, np.uint8)
    codes[l == "A"] = pm.A
    codes[l == "D"] = pm.D
    ss = pm.spell_scores(GT, pm.fill_abstentions(codes, pm.valid_runs(GT)),
                         pm.valid_runs(GT))
    r = lab.score(l, name, verbose=False)
    print("%-28s spellF1=%.4f macroF1=%.4f shareE=%.3f nPred=%3d team_acc=%.3f "
          "iou3=%.3f/5=%.3f/7=%.3f"
          % (name, r["spell_f1_mean"], r["macro_f1"], r["share_err"], ss["n_pred"],
             r["team_acc_gt_alive"],
             ss["per_iou"][0.3]["f1"], ss["per_iou"][0.5]["f1"], ss["per_iou"][0.7]["f1"]))


print("== pure crossover switches (enter margin 0.0) ==")
for k in (5, 10, 15, 25, 40):
    report(switch_machine(0.0, 0.0, k), "cross k=%d" % k)

print("\n== small enter margin ==")
for enter, k in ((0.3, 10), (0.5, 10), (0.5, 20), (0.8, 20), (1.0, 25), (1.4, 30)):
    report(switch_machine(enter, enter, k), "enter %.1f k=%d" % (enter, k))

print("\n== baseline comparison (majority vote, same graded-edge init) ==")
