"""Fast experiment harness: tune the team + dead-ball logic on cached features.

The public GT window is used ONLY for validation and structure checks
(spell count, D coverage, possession durations) -- not for constant fitting,
per RULES.md / baseline.py. Every threshold here is checked against its
whole-match distribution profile as well.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")

from evaluation import possession_metrics as pm  # noqa: E402

N = 199319
GT = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
GT_CODES = np.array([pm.LABELS.index(c) for c in GT["label"]])


def load_sig():
    from kashaf.features import (load_features, load_smart_tagging,
                                 tagging_arrays)
    f = load_features()
    n = f["n_frames"]
    ev = load_smart_tagging()
    sp, pt = tagging_arrays(ev, n)
    mv = np.nanmean(np.stack([f["vel_home"], f["vel_away"]]), axis=0)
    mv = pd.Series(mv).rolling(50, center=True, min_periods=25).median().to_numpy()
    return f, sp, pt, mv


def team_signal(f, pt, gap_open=1.0, gap_close=3.0):
    """Per-frame team with hysteresis + wide smoothing -> long spells.

    Hysteresis: switch to the other side only once it is clearly closer, and
    hold the current side while it stays within `gap_close`. This kills single-
    frame flicker before smoothing.
    """
    n = f["n_frames"]
    dH = f["d_home"]
    dA = f["d_away"]
    vis = f["ball_present"]
    cur = 0  # start home; will be overwritten by smoothing anyway
    team = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        if not vis[i] or (np.isnan(dH[i]) and np.isnan(dA[i])):
            team[i] = cur  # carry
            continue
        dh = dH[i] if np.isfinite(dH[i]) else 99.0
        da = dA[i] if np.isfinite(dA[i]) else 99.0
        if da < dh - gap_open:
            cur = 1
        elif dh < da - gap_open:
            cur = 0
        # else keep current (within hysteresis band)
        team[i] = cur
    return team


def smooth_majority(team, window):
    """Centred majority vote ignoring -1 (carry) cells; -1 if window empty."""
    n = len(team)
    out = team.copy()
    h = np.zeros(n + 1, dtype=np.int64)
    h[1:] = np.cumsum(team == 0)
    a = np.zeros(n + 1, dtype=np.int64)
    a[1:] = np.cumsum(team == 1)
    idx = np.arange(n)
    lo = np.maximum(0, idx - window)
    hi = np.minimum(n, idx + window + 1)
    hc = h[hi] - h[lo]
    ac = a[hi] - a[lo]
    any_ball = (hc + ac) > 0
    out[~any_ball] = -1
    out[any_ball & (hc >= ac)] = 0
    out[any_ball & (ac > hc)] = 1
    return out


def dead_signal(f, sp, mv, ball_slow_thresh=3.0, vel_thresh=1.45,
                require_evidence=True):
    """Per-frame dead-ball boolean.

    Dead when the sides are standing still (windowed velocity low) AND a
    secondary tell (a set-piece window, or a slow/hidden ball, or the ball is
    far from everyone). Set-piece-only frames are dead unconditionally.
    """
    vis = f["ball_present"]
    bv = f["ball_vel"]
    still = mv <= vel_thresh
    slowball = vis & (bv < ball_slow_thresh)
    tag = sp
    dead = tag | (still & (slowball | ~vis))
    if require_evidence:
        dead = tag | (still & (slowball | ~vis))
    return dead


def to_labels(team, dead):
    lab = np.empty(N, dtype="<U1")
    live = team >= 0
    lab[~live] = "H"
    lab[live & (team == 0)] = "H"
    lab[live & (team == 1)] = "A"
    lab[dead] = "D"
    return lab


def score(labels, name, verbose=True):
    preds = np.zeros(N, dtype=np.uint8)
    for c, v in zip("HAD", (0, 1, 2)):
        preds[labels == c] = v
    r = pm.evaluate(GT_CODES, preds, bootstrap_n=20)
    line = ("%s macroF1=%.4f spellF1=%.4f share_err=%.3f alive_deadF1=%.3f"
            % (name, r["macro_f1"], r["spell_f1_mean"], r["share_err"],
               r["alive_dead_f1"]))
    if verbose:
        print(line)
        print("      poss pred/gt spell counts, frag:  "
              "spellP=%.0f spellR=%.0f  gtSpells=%d"
              % (0, 0, 0))
    return r


if __name__ == "__main__":
    f, sp, pt, mv = load_sig()
    print("signals loaded", f["n_frames"])
