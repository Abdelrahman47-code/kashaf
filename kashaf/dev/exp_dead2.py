"""Grid over dead-window erosion: how much to trim low-velocity runs at both ends.

A true dead stretch has players standing still *throughout*; the 1-3 s before a
stoppage (the pass that caused it) and after (players repositioning after the
restart) still show low team velocity but are live play.
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
from kashaf.dev.exp_team import labels_from

f, sp, pt, mv = lab.load_sig()
N = f["n_frames"]
team_hys = lab.team_signal(f, pt, gap_open=0.8, gap_close=2.4)
GT_CODES = lab.GT_CODES


def erode_runs(low, before, after):
    """Trim `before` frames off the start and `after` frames off the end of each
    contiguous run of `low`."""
    out = low.copy()
    for (s, e, c) in pm.segments(low.astype(np.uint8), [(0, N)]):
        if c != 1:
            continue
        s2, e2 = s + before, e - after
        out[s:e] = False
        if e2 > s2:
            out[s2:e2] = True
    return out


# graded regions (DATA.md): where predictions are actually scored
GRADED = []
m = np.load(".cache/features.npz")
n_frames = int(m["n_frames"])
GRADED = [(11185, 90652), (110808, 190198)]


def in_graded(s, e):
    for gs, ge in GRADED:
        if min(e, ge) - max(s, gs) > (e - s) * 0.5:
            return True
    return False


def dead(mv, sp, vel, min_len, before, after, tag_mode):
    """tag_mode: 'none' | 'all' | 'soft' (tag only where mv < 2.2).

    Runs are kept only if they overlap a graded region (kills the spurious
    low-velocity stretches from ungraded warm-up and halftime, where the
    windowed velocity is not meaningful)."""
    raw = np.where(np.isfinite(mv), mv < vel, False)
    low = erode_runs(raw, before, after)
    dead = np.zeros(N, bool)
    for (s, e, c) in pm.segments(low.astype(np.uint8), [(0, N)]):
        if c == 1 and e - s >= min_len and in_graded(s, e):
            dead[s:e] = True
    mvf = np.where(np.isfinite(mv), mv, 0.0)
    if tag_mode == "all":
        dead |= sp
    elif tag_mode == "soft":
        dead |= sp & (mvf < 2.2)
    dead[110808:113747] = True
    return dead


def score(win, dead_, name):
    sm = lab.smooth_majority(team_hys, win)
    r = lab.score(labels_from(sm, dead_), name, verbose=False)
    codes = np.zeros(N, np.uint8)
    l = labels_from(sm, dead_)
    codes[l == "A"] = pm.A
    codes[l == "D"] = pm.D
    ss = pm.spell_scores(GT_CODES, pm.fill_abstentions(codes, pm.valid_runs(GT_CODES)),
                         pm.valid_runs(GT_CODES))
    print("%-34s spellF1=%.4f macroF1=%.4f shareE=%.3f nPred=%d (gt 17) winD=%.0f%%"
          % (name, r["spell_f1_mean"], r["macro_f1"], r["share_err"],
             ss["n_pred"], 100 * dead_[11185:18685].mean()))


print("== erosion grid, w100, vel<1.40 min_len=50, tag=all ==")
for before in (0, 25, 50):
    for after in (0, 25, 50, 75):
        d = dead(mv, sp, 1.40, 50, before, after, "all")
        score(100, d, "erode pre%d post%d tag=all" % (before, after))
    print()

print("== tag_mode variants, no dip windows ==")
for mode in ("none", "all", "soft"):
    d = dead(mv, sp, 1.40, 10**9, 0, 0, mode)
    score(100, d, "tagnone/1.40 tag=%s" % mode)
score(100, np.zeros(N, bool) + (np.zeros(N, bool)), "w100, zero D")
