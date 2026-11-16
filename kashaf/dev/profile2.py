"""Quick GT-window signal profile to choose alive/dead discriminators."""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")
from kashaf.features import load_features, load_smart_tagging, tagging_arrays  # noqa: E402

f = load_features()
n = f["n_frames"]
ev = load_smart_tagging()
sp, pt = tagging_arrays(ev, n)
gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
lab = gt["label"].values
Hf = np.where(lab == "H")[0]
Af = np.where(lab == "A")[0]
Df = np.where(lab == "D")[0]
LIVE = np.concatenate([Hf, Af])


def prof(name, arr):
    out = {}
    for tag, ix in (("live", LIVE), ("D", Df)):
        v = arr[ix]
        v = v[np.isfinite(v)]
        out[tag] = tuple(np.percentile(v, p) for p in (25, 50, 75, 90))
    print("%-12s live p25/50/75/90 = %s   D = %s"
          % (name, " ".join("%.2f" % x for x in out["live"]),
             " ".join("%.2f" % x for x in out["D"])))


mv = np.nanmean(np.stack([f["vel_home"], f["vel_away"]]), axis=0)
prof("ball_vel", f["ball_vel"])
prof("mv", mv)
prof("mv_p95", np.nanmax(np.stack([f["vel_home"], f["vel_away"]]), axis=0))
prof("med_pvel", f["med_player_vel"])
prof("n_moving", f["n_moving"])
prof("n_players", f["n_players"])
prof("crowdH", f["crowd_home_2"])
prof("crowdA", f["crowd_away_2"])
prof("d_min", np.minimum(f["d_home"], f["d_away"]))

print("vis            live %.3f  D %.3f" % (f["ball_present"][LIVE].mean(), f["ball_present"][Df].mean()))
print("ball_vel>3     live %.3f  D %.3f"
      % ((f["ball_vel"][LIVE] > 3).mean(), (f["ball_vel"][Df] > 3).mean()))
print("n_moving>=10   live %.3f  D %.3f"
      % ((f["n_moving"][LIVE] >= 10).mean(), (f["n_moving"][Df] >= 10).mean()))
print("mv>=1.5        live %.3f  D %.3f" % ((mv[LIVE] >= 1.5).mean(), (mv[Df] >= 1.5).mean()))
print("setpiece       D %.2f  live %.2f  (D-covered %d/%d frames)"
      % (sp[Df].mean(), sp[LIVE].mean(), int(sp[Df].sum()), len(Df)))
print("phase_team     %d frames home, %d away, %d uncovered in window"
      % (int((pt == 0).sum()), int((pt == 1).sum()), int((pt == -1).sum())))

# the 2 untagged D stretches: where exactly?
runs = []
starts = np.flatnonzero(lab[1:] != lab[:-1]) + 1
for a, b in zip(np.concatenate(([0], starts)), np.concatenate((starts, [n]))):
    if lab[a] == "D":
        runs.append((int(a), int(b), int(b - a)))
print("D segments (start,len):", runs)
for s, e, ln in runs:
    print("  %d-%d len %d: setpiece %.2f  mv med %.2f  n_mov med %.0f  vis %.2f"
          % (s, e, ln, sp[s:e].mean(),
             np.nanmedian(mv[s:e]), np.nanmedian(f["n_moving"][s:e]),
             f["ball_present"][s:e].mean()))
