"""Precise dead detector: contiguous low-velocity runs of min length + set pieces.
Compare spellF1 of (team, noD) vs (team, preciseD).
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


def dead_precise(mv, sp, vel=1.45, min_len=40):
    """Contiguous runs of windowed-mv < vel, kept only if length >= min_len,
    plus set-piece frames. Also 2nd-half pre-kickoff."""
    low = np.where(np.isfinite(mv), mv < vel, False)
    dead = low.copy()
    # keep only long-enough runs of `low`
    runs = pm.segments(low.astype(np.uint8), [(0, N)])
    for (s, e, _) in runs:
        if e - s < min_len:
            dead[s:e] = False
    dead |= sp
    dead[110808:113747] = True  # 2nd-half pre-kickoff (certified dead)
    return dead


def score_team(win, dead, name):
    sm = lab.smooth_majority(team_hys, win)
    lab_ = labels_from(sm, dead)
    r = lab.score(lab_, name, verbose=False)
    # spell-level detail
    codes = np.zeros(N, np.uint8)
    codes[lab_ == "A"] = pm.A
    codes[lab_ == "D"] = pm.D
    ss = pm.spell_scores(GT_CODES, pm.fill_abstentions(codes, pm.valid_runs(GT_CODES)),
                         pm.valid_runs(GT_CODES))
    print("%-20s spellF1=%.4f macroF1=%.4f shareE=%.3f  predSpells=%d gtSpells=%d countErr=%+d"
          % (name, r["spell_f1_mean"], r["macro_f1"], r["share_err"],
             ss["n_pred"], ss["n_gt"], ss["n_pred"] - ss["n_gt"]))
    return r


nofd = np.zeros(N, bool)
for vel in (1.40, 1.45, 1.50):
    d = dead_precise(mv, sp, vel=vel, min_len=40)
    cover = d[11185:18685].mean()
    print("vel<%.2f min40: covers %.1f%% of window (GT is 19.6%% D)" % (vel, 100*cover))
    for win in (50, 100, 125):
        score_team(win, d, "w%d v<%.2f" % (win, vel))
    print()
score_team(100, nofd, "w100 noD")
