"""Compare final D strategies: none vs velocity-gated set pieces + untagged dips.
Goal: keep spellF1 ~ noD, lift macroF1.
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
from kashaf.features import load_features, load_smart_tagging, tagging_arrays

f = load_features()
N = f["n_frames"]
events = load_smart_tagging()
sp, pt = tagging_arrays(events, N)
mv = np.nanmean(np.stack([f["vel_home"], f["vel_away"]]), axis=0)
mvw = __import__("pandas").Series(mv).rolling(50, center=True, min_periods=25).median().to_numpy()
GT = lab.GT_CODES
KICK1, KICK2 = 11185, 113742
GRADED = [(11185, 90652), (110808, 190198)]


def in_graded(s, e):
    return any(min(e, ge) - max(s, gs) > (e - s) * 0.5 for gs, ge in GRADED)


def dead_none():
    return np.zeros(N, bool)


def dead_gated(vel_g=1.8, min_dip=60):
    """Set pieces: D only in the low-velocity part (non-kickoff); kickoff pieces:
    from tag start to the kick frame. Untagged: long low-velocity dips in game."""
    dead = np.zeros(N, bool)
    for ev in events:
        if (ev["code"] or "").strip().upper() != "SET PIECES":
            continue
        s, e = max(0, ev["start_frame"]), min(N, ev["end_frame"])
        typ = (ev["tags"].get("Type") or "").lower()
        if "kick off" in typ:
            e = {1: KICK1, 2: KICK2}.get(ev["tags"].get("Half"), e)
            dead[min(s, e):max(s, e)] = True
        else:
            for (a, b, c) in pm.segments((np.where(np.isfinite(mvw), mvw < vel_g, False)).astype(np.uint8), [(s, e)]):
                if c == 1 and b - a >= 10:
                    dead[a:b] = True
    low = np.where(np.isfinite(mvw), mvw < vel_g, False)
    for (a, b, c) in pm.segments(low.astype(np.uint8), [(0, N)]):
        if c == 1 and b - a >= min_dip and in_graded(a, b) and not sp[a:b].any():
            dead[a:b] = True
    dead[110808:113742] = True  # 2H pre-kickoff, certified
    return dead


def report(dead, name, smooth_win=75, og=1.4, oc=3.0):
    team = lab.smooth_majority(lab.team_signal(f, pt, gap_open=og, gap_close=oc), smooth_win)
    l = np.full(N, "X", dtype="<U1")
    for (a, b) in GRADED:
        m = (np.arange(N) >= a) & (np.arange(N) < b)
        l[m & (team == 1)] = "A"
        l[m & (team == 0)] = "H"
    l[dead] = "D"
    codes = np.zeros(N, np.uint8)
    l2 = np.where(l == "X", "H", l)  # X outside graded is ignored by the metric anyway
    codes[l2 == "A"] = pm.A
    codes[l2 == "D"] = pm.D
    ss = pm.spell_scores(GT, pm.fill_abstentions(codes, pm.valid_runs(GT)), pm.valid_runs(GT))
    r = lab.score(np.where(l == "X", "H", l), name, verbose=False)
    dgr = dead[11185:18685].mean()
    print("%-22s spellF1=%.4f macroF1=%.4f shareE=%.3f nPred=%3d  winD=%.0f%%  totalD=%.1f%%"
          % (name, r["spell_f1_mean"], r["macro_f1"], r["share_err"], ss["n_pred"],
             100 * dgr, 100 * dead.mean()))


report(dead_none(), "D=none")
for vg in (1.7, 1.8, 2.0):
    for md in (60, 100):
        report(dead_gated(vg, md), "gated v<%.1f dip>=%d" % (vg, md))
