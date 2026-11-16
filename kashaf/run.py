"""Kashaf's possession pipeline -- canonical entrypoint.

    python -m kashaf.run
    python -m kashaf.run --out evaluation/results/Kashaf.csv

Reproduces the submitted CSV exactly from the provided data (ATD tracking
+ Smart Tagging). See kashaf/solution.py for the method and kashaf/README.md
for the write-up.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from kashaf import solution  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "Kashaf.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--data-dir", default=None,
                        help="match bundle (else MTAH_DATA_DIR, else ./data)")
    args = parser.parse_args()
    data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or None

    labels = solution.build_labels(data_dir=data_dir)
    solution.write(args.out, labels)
    counts = {c: int((labels == c).sum()) for c in "HAD"}
    print("wrote {}".format(args.out))
    print("counts over all {} frames: {}".format(len(labels), counts))
    blocks = solution.graded_blocks(data_dir)
    print("graded half blocks (from public games.py): {}".format(blocks))
    pk = solution.pre_kickoff_slice(data_dir)
    if pk is not None:
        print("2nd-half dead block D = [{}..{})  <- derived from the "
              "tracker kick frame + public half boundary".format(pk.start, pk.stop))
    print("\nNow score:  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
