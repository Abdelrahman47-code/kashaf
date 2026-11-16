# Kashaf — More than a Hack 2027, Phase 1: Ball Possession

Team **Kashaf**'s submission for the *More than a Hack 2027* Phase 1 challenge
(organised by Metrica Sports, FC Barcelona and Mobile World Capital Barcelona).

Task: for every frame of one full LaLiga match (Barcelona vs. Athletic Club,
199 319 tactical-video frames), say who has the ball: `H`, `A`, `D` (dead ball)
or `X` (abstain). Ranked on `spellF1`.

```
repo/
├── games.py                 per-match constants, asset paths, frame alignment (from the organiser repo)
├── evaluation/
│   ├── possession_metrics.py  the metric library (from the organiser repo; this exact file scores the board)
│   ├── run_evaluation.py      the scorer (from the organiser repo)
│   ├── validate_submission.py format checks, identical to the official scorer's
│   └── gt/                    the public 5-minute ground truth (ships with the organiser repo)
├── starter/
│   └── load_smart_tagging.py  Smart Tagging XML reader (from the organiser repo; needed by kashaf/features.py)
├── kashaf/
│   ├── run.py                 CANONICAL ENTRY POINT — writes evaluation/results/Kashaf.csv
│   ├── solution.py            the whole model: nearest-side + hysteresis -> majority vote -> labels
│   ├── features.py            vectorised ATD parser + per-frame signal arrays
│   └── dev/                   the experiment scripts and harness used to measure the design choices
│                              (dead-ball detectors, hysteresis margins, smoothing width). They read
│                              only the public 5-minute ground truth, which the rules reserve for
│                              validation. Not needed to reproduce the submission.
└── docs/
    └── technical.md           the technical document (deliverable 3)
```

## Reproduce the submission file

```
pip install -r requirements.txt

# point the loader at the unpacked match bundle (enrolled participants only; see DATA.md in the
# organiser repository for how to get it). Unpacked here it is "details/", matching the layout.
set MTAH_DATA_DIR=C:\path\to\match\bundle      # Windows
# export MTAH_DATA_DIR=/path/to/match/bundle   # macOS / Linux

python -m kashaf.run                           # writes evaluation/results/Kashaf.csv, ~20 s
python evaluation/validate_submission.py evaluation/results/Kashaf.csv
python evaluation/run_evaluation.py            # score against the public 5 minutes
```

`python -m kashaf.run` reproduces the submitted CSV **byte for byte**:
`frame,label`, 199 319 data rows, frames 0..199 318 in order.

The data bundle is not in this repository and cannot be (it is separately licensed for the
competition and not redistributable). The scorer and the public 5-minute ground truth are self-
sufficient: on a fresh clone, `python evaluation/run_evaluation.py` scores whatever is in
`evaluation/results/` without any data.

## What the pipeline does

Three stages, no training, numpy + pandas only; full write-up in
[docs/technical.md](docs/technical.md):

1. **Team of the ball** — where the ball and both sides are tracked, the nearest side has the
   ball, with hysteresis (switch only once the other side is 1.5 m nearer; hold while within
   2.5 m). A one-frame crossing is a tackle, a rebound or tracker flicker, not a turnover.
2. **Smoothing** — centred 2 s majority vote over the hysteresis signal.
3. **Dead ball** — only the certified 2nd-half pre-kickoff stretch (frames 110 808–113 741) is
   labelled `D`; outside the two graded half blocks the label is `X`.

Inputs: the provided ATD tracking (`*_tactical_atd.txt`) and team/slot metadata
(`*_tactical_atd.xml`). No video, no external data, no human labels, and no ground truth of any
kind enters the pipeline; the public 5-minute window is used only to validate the output
structure, as the rules require.

## Licence

Code: MIT, see [LICENSE](LICENSE). The match data is **not** covered by it; it is separately
licensed for this competition under the terms accepted at enrolment, and nothing in this
repository reproduces any of it.
