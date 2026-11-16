"""Profile the public 5-min ground truth window against the per-frame signals.

Pure exploration: prints, for each GT label (H / A / D) and for the frames AROUND
transitions, how the candidate "dead ball" signals spread. Run:

    python -m kashaf.analyze_gt
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from kashaf.features import load_features, load_smart_tagging, tagging_arrays  # noqa: E402


def ball_speed_from_positions(ball):
    """Frame-to-frame ball speed in m/s; NaN where the ball is missing either frame."""
    prev = np.roll(ball, 1, axis=0)
    prev[0] = np.nan
    delta = np.linalg.norm(ball - prev, axis=1)
    speed = delta / (1.0 / GAMES[GAME]["fps"])
    speed[np.isnan(ball).any(axis=1) | np.isnan(prev).any(axis=1)] = np.nan
    return speed


def main():
    f = load_features()
    n = f["n_frames"]
    ball_speed = ball_speed_from_positions(f["ball"])

    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt",
                                  "BAR-ATH_possession_gt_public.csv"))
    labels = np.array(list(gt["label"]))

    # ---- signal spread per GT label in the public window --------------------
    print("== signal spread by GT label (public window 11185..18684) ==")
    def npct(x, p):
        x = x[np.isfinite(x)]
        return float(np.percentile(x, p)) if x.size else float("nan")

    rows = []
    for L in "HAD":
        m = labels == L
        if not m.any():
            continue
        rows.append((L, int(m.sum()),
                     float(100 * np.mean(f["ball_present"][m])),
                     npct(ball_speed[m], 50),
                     npct(ball_speed[m], 90),
                     npct(f["d_home"][m], 50),
                     npct(f["d_away"][m], 50),
                     npct(f["crowd_home_2"][m], 50),
                     npct(f["crowd_away_2"][m], 50)))
    print("lab  n   ballvis%  spd_p50  spd_p90  dH    dA    crowdH  crowdA")
    for r in rows:
        print("{:<3} {:>5} {:>8.0f} {:>7.2f} {:>7.1f} {:>5.2f} {:>5.2f} {:>6.1f} {:>6.1f}".format(*r))

    # ---- D-segment statistics ------------------------------------------------
    print("\n== D segment structure ==")
    d = (labels == "D").astype(np.int8)
    pad = np.concatenate(([0], d, [0]))
    up = np.flatnonzero(np.diff(pad) > 0)
    down = np.flatnonzero(np.diff(pad) < 0)
    seg_len = down - up
    print("segments: {}  length: mean {:.1f}f  median {:.1f}  <5f {:.0%}  <15f {:.0%}  >50f {:.0%}".format(
        len(seg_len), seg_len.mean(), np.median(seg_len),
        (seg_len < 5).mean(), (seg_len < 15).mean(), (seg_len > 125).mean()))

    # how much of each D segment sits (frame-wise) inside a SET PIECES window?
    events = load_smart_tagging()
    set_piece, _ = tagging_arrays(events, n)
    fractions = np.array([(set_piece[a:b]).mean() for a, b in zip(up, down)])
    weighted = (fractions * seg_len).sum() / seg_len.sum()
    print("set-piece coverage of D-frames: weighted {:.0%}  per-segment [min {:.0%} / max {:.0%}]".format(
        weighted, fractions.min(), fractions.max()))
    for (a, b), frac in sorted(zip(zip(up, down), fractions), key=lambda x: -x[1])[:6]:
        print("  D seg [{}..{}] len {}  in set-piece window {:.0%}".format(a, b, b - a, frac))

    # ---- transition structure --------------------------------------------------
    print("\n== H<->A transitions in the window ==")
    alive = np.where((labels == "H") | (labels == "A"),
                     np.where(labels == "H", 0, 1), -1)
    trans = []
    for i in range(1, len(alive)):
        if alive[i] in (0, 1) and alive[i - 1] in (0, 1) and alive[i] != alive[i - 1]:
            trans.append(i)
    print("{} direct H/A transitions".format(len(trans)))
    for t in trans:
        print("  frame {:>5} -> {}  dH={:.1f} dA={:.1f} ballvis={}".format(
            t, "H" if alive[t] == 0 else "A",
            f["d_home"][t], f["d_away"][t], int(f["ball_present"][t])))

    # ---- what does raw nearest-player say about the window? ---------------
    print("\n== raw nearest-player agreement by GT label ==")
    near_home = f["d_home"] <= f["d_away"]
    vis = f["ball_present"]
    for L in "HA":
        m = (labels == L) & vis
        agree = (0 if L == "H" else 1)
        pred_home = near_home[m].mean()
        correct = near_home[m].mean() if L == "H" else (1 - near_home[m].mean())
        print("GT {}: nearest player on GT team in {:.0%} of visible frames".format(L, correct))


if __name__ == "__main__":
    main()
