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
    args = parser.parse_args()

    labels = solution.build_labels()
    solution.write(args.out, labels)
    print("wrote {}".format(args.out))
    counts = {c: int((labels == c).sum()) for c in "HAD"}
    print("counts over all {} frames: {}".format(len(labels), counts))
    print("\nNow score:  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
