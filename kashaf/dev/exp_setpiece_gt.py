"""For every SET PIECES window inside the public grade window, show what the GT
labels those frames. This tells us whether a set piece is D (dead ball) or
H/A (possession continues) in the reference labels."""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")

from kashaf.features import load_features, load_smart_tagging, tagging_arrays

f = load_features()
N = f["n_frames"]
events = load_smart_tagging()
sp, pt = tagging_arrays(events, N)

import pandas as pd
gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
gtmap = dict(zip(gt["frame"], gt["label"]))

s0, s1 = 11185, 18685
print("SET PIECES windows inside the public window:")
print("frame range  type            GT-labels (counted)              length")
for ev in events:
    code = (ev["code"] or "").strip().upper()
    if code != "SET PIECES":
        continue
    a, b = ev["start_frame"], min(ev["end_frame"], N)
    if a >= s1 or b <= s0:
        continue
    a2, b2 = max(a, s0), min(b, s1)
    if a2 >= b2:
        continue
    counts = {}
    for fr in range(a2, b2):
        l = gtmap.get(fr)
        if l:
            counts[l] = counts.get(l, 0) + 1
    typ = (ev["tags"].get("Type") or "?")
    half = (ev["tags"].get("Half") or "?")
    print("%6d-%-6d  %-14s %s  %s  len %d" % (
        a2, b2, typ, half,
        " ".join("%s=%d" % (k, v) for k, v in sorted(counts.items())),
        b2 - a2))
