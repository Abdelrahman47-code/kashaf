"""Hybrid team assignment: wide smoothing for stability + decisive-turnover
protection so short possessions survive.

   A: hysteresis (open/hold margin) + majority-vote window  -> long, clean spells
   B: switch machine (margin enter, k consecutive)          -> localises turnovers
   keep a short run of B inside a long run of A only if the side held a
   decisive margin for most of it (a real flash possession, not flicker)
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
GT = lab.GT_CODES


def hyst_team(og, oc):
    return lab.team_signal(f, pt, gap_open=og, gap_close=oc)


def smooth(team, win):
    return lab.smooth_majority(team, win)


def switch_team(enter, k):
    team = np.full(N, -1, dtype=np.int8)
    cur = -1
    acc_side = -1
    acc = 0
    for i in range(N):
        if not vis[i] or (np.isnan(dH[i]) and np.isnan(dA[i])):
            team[i] = cur
            continue
        dh = dH[i] if np.isfinite(dH[i]) else 99.0
        da = dA[i] if np.isfinite(dA[i]) else 99.0
        side = 0 if dh < da else 1
        gap = abs(dh - da)
        if cur < 0:
            cur = side
        elif side != cur and gap > enter:
            if acc_side == side:
                acc += 1
            else:
                acc_side, acc = side, 1
            if acc >= k:
                cur = side
                acc_side, acc = side, 0
        team[i] = cur
    return team


def margin_team():
    """per frame: (0/1 side of nearest, margin in m, -1 no info)"""
    t = np.full(N, -1, dtype=np.int8)
    m = np.full(N, np.nan)
    for i in range(N):
        if not vis[i] or (np.isnan(dH[i]) and np.isnan(dA[i])):
            continue
        dh = dH[i] if np.isfinite(dH[i]) else 99.0
        da = dA[i] if np.isfinite(dA[i]) else 99.0
        t[i] = 0 if dh < da else 1
        m[i] = abs(dh - da)
    return t, m, (dH - dA)


def hybrid(sa, win, enter, k, keep_margin=1.5, keep_frac=0.6, min_keep=25):
    """smoothing window `win`, hysteresis from sa; protected flash spells."""
    base = smooth(hyst_team(sa[0], sa[1]), win)
    sw = switch_team(enter, k)
    t, m, signed = margin_team()
    out = base.copy()
    # find runs of sw that differ from base; keep if decisively on one side
    runs = pm.segments((sw == 1).astype(np.uint8), [(0, N)])
    for (s, e, c) in runs:
        side = c
        base_side = int(np.bincount(base[s:e], minlength=2).argmax())
        if side == base_side:
            continue
        # how decisively was `side` nearer during the run?
        good = ((t == side) & (m >= keep_margin))
        if s < e and (good[s:e]).mean() >= keep_frac and (e - s) >= min_keep:
            out[s:e] = side
    return out


def labels(team):
    l = np.full(N, "H", dtype="<U1")
    l[team == 1] = "A"
    return l


def report(team, name):
    l = labels(team)
    codes = np.zeros(N, np.uint8)
    codes[l == "A"] = pm.A
    codes[l == "D"] = pm.D
    ss = pm.spell_scores(GT, pm.fill_abstentions(codes, pm.valid_runs(GT)),
                         pm.valid_runs(GT))
    r = lab.score(l, name, verbose=False)
    print("%-34s spellF1=%.4f macroF1=%.4f shareE=%.3f nPred=%3d "
          "iou3=%.3f/5=%.3f/7=%.3f" %
          (name, r["spell_f1_mean"], r["macro_f1"], r["share_err"], ss["n_pred"],
           ss["per_iou"][0.3]["f1"], ss["per_iou"][0.5]["f1"], ss["per_iou"][0.7]["f1"]))


base75 = smooth(hyst_team(1.4, 3.0), 75)
report(base75, "A: w75 g1.4/3.0")
base100 = smooth(hyst_team(1.4, 3.0), 100)
report(base100, "A: w100 g1.4/3.0")
sw30 = switch_team(1.4, 30)
report(sw30, "B: sw enter1.4 k30")
sw25 = switch_team(1.5, 25)
report(sw25, "B: sw enter1.5 k25")

print()
for keep_m in (1.0, 1.5, 2.0):
    hf = hybrid((1.4, 3.0), 75, 1.5, 25, keep_margin=keep_m)
    report(hf, "hyb w75+sw1.5/25 km%.1f" % keep_m)
print()
for keep_m in (1.0, 1.5, 2.0):
    hf = hybrid((1.4, 3.0), 100, 1.5, 25, keep_margin=keep_m)
    report(hf, "hyb w100+sw1.5/25 km%.1f" % keep_m)
