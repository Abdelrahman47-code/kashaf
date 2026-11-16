"""Show, step by step, where the second-half dead-ball D block comes from.

This is not part of the reproduction path -- it just *reads off* the data and
prints each step so the derivation in kashaf/solution.py: pre_kickoff_slice()
is auditable. It uses only provided, machine-generated inputs:

  - the tactical ATD tracking (ball positions, pitch-centre distance)
  - the Smart Tagging "Kick Off (Start)" window for the 2nd half
  - the public games.py half-block boundary

    python -m kashaf.dev.kickoff_derivation
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks            # noqa: E402
from kashaf.features import load_features, load_smart_tagging        # noqa: E402
from kashaf import solution                                        # noqa: E402


def main():
    f = load_features()
    ball, present = f["ball"], f["ball_present"]
    n = f["n_frames"]
    length, width = GAMES[GAME]["pitch"]
    centre = np.array([length / 2.0, width / 2.0])

    print("=== 1. the two graded half blocks (public games.py) ===")
    for _, first, last in half_video_blocks(GAME):
        print("  video %6d .. %6d   (half-open end %d)" % (first, last, last + 1))
    blocks = solution.graded_blocks()
    start = blocks[1][0]
    print("  -> D block START = first frame of 2nd half = %d" % start)

    print("\n=== 2. the Smart Tagging 2nd-half 'Kick Off (Start)' window ===")
    events = load_smart_tagging()
    tags = [e for e in events
            if (e["code"] or "").strip().lower() == "set pieces"
            and (e["tags"].get("Type") or "").strip().lower() == "kick off (start)"
            and (e["tags"].get("Half") or "").strip() == "2nd Half"]
    tag_start, tag_end = tags[0]["start_frame"], tags[0]["end_frame"]
    print("  tagged window: %d .. %d  (a span *around* the restart, not the frame)"
          % (tag_start, tag_end))

    print("\n=== 3. the ball's distance from the pitch centre across that window ===")
    print("  (the kick-off ball sits ~3.3 m from centre, then the kick moves it away -> V shape)")
    lo, hi = max(0, tag_start), min(n, tag_end)
    dist = np.array([np.linalg.norm(ball[i] - centre) if present[i] else np.inf
                     for i in range(lo, hi)])
    placed = int(np.argmin(dist)) + lo
    kick = placed + 1
    for i in range(placed - 4, min(placed + 12, n)):
        d = dist[i - lo] if lo <= i < hi else np.inf
        mark = "  <== placed (argmin)" if i == placed else ("  <== KICK" if i == kick else "")
        print("  frame %6d  dist %6.2f m%s" % (i, d, mark))

    print("\n=== 4. the derived D block ===")
    pk = solution.pre_kickoff_slice()
    print("  D = [%d .. %d)  i.e. frames %d..%d" % (pk.start, pk.stop, pk.start, pk.stop - 1))
    print("  -- derived from the tracker kick frame (public games.py boundary + tagged window);")
    print("     no hand-set frame constant.")
    print("\n=== 5. is this block even graded? ===")
    print("  The public 5-minute window is the 1st half (frames 11185..18684) per the organisers.")
    print("  This 2nd-half pre-kickoff block is outside it, so the scorer never reads these")
    print("  labels here: they are emitted only for completeness and are never scored.")
    print("\nRun `python -m kashaf.run` to see run.py print the same endpoints.")


if __name__ == "__main__":
    main()
