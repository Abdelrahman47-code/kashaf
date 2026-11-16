"""Inspect frames 14846-15889 (GT: A H A H A fast exchanges) to see why the
team signal misses 4 turnovers there.
"""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")

import pandas as pd
from kashaf.features import load_features
from kashaf.dev import lab

f = load_features()
N = f["n_frames"]
vis = f["ball_present"]
dH = f["d_home"]
dA = f["d_away"]
cH = f["crowd_home_2"]
cA = f["crowd_away_2"]

gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
gtmap = dict(zip(gt["frame"], gt["label"]))

# GT sub-segments in this stretch (from earlier probe):
segs = [(14846, "A"), (15346, "H"), (15365, "A"), (15766, "H"), (15799, "A")]
print("GT labels in stretch 14846-15889 (boundaries):")
prev = None
for fr in range(14846, 15890):
    l = gtmap.get(fr)
    if l != prev:
        print("   %d -> %s" % (fr, l))
        prev = l

print("\nper 25-frame sample: dH  dA  margin(dA-dH)  crowdH crowdA  ballvis")
for i in range(14846, 15890, 25):
    print("  %d  %5.2f %5.2f  %+6.2f  %d      %d      %d"
          % (i, dH[i] if np.isfinite(dH[i]) else -1,
             dA[i] if np.isfinite(dA[i]) else -1,
             (dA[i] - dH[i]) if np.isfinite(dH[i]) and np.isfinite(dA[i]) else -99,
             cH[i], cA[i], int(vis[i])))
