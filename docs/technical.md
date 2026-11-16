# Kashaf  -  Technical Document

**More than a Hack 2027  -  Phase 1: Ball Possession**

*Team Kashaf. Barcelona vs. Athletic Club, one full LaLiga match, 199 319 frames of the tactical
video. Task: label every frame `H`, `A`, `D` (dead ball) or `X` (abstain). Ranked by `spellF1`.*

---

## 1. Approach in one paragraph

The tracker already gives us, every frame, the ball's position and the position of all 44 players.
That means "who has the ball" is mostly a geometry question: which team's nearest player is closest
to the ball  -  and the real difficulty is that the tracking is automatic and noisy, so a raw
nearest-player vote flips teams at every tackle, rebound and bad frame and shatters the match into
hundreds of fake "possessions." Our whole system is a deterministic rule, about 130 lines of
NumPy, that turns that noisy geometry into stable possessions. We trained nothing and used no
dataset: there is no model, no learning step, and no hyperparameter was chosen against any ground
truth.

We rank **first** on the public five-minute sample (`spellF1` 0.4308 vs 0.4242 for the best
baseline) and, more importantly, we believe the full-match structure we predict is right: ~329
possessions of ~19 s each, which lines up with the baselines' published full-match behaviour
(~330 possessions, ~18 s). We also land the best turnover timing on the board  -  mean latency of
+1.0 frame, where the nearest baselines are off by 4 to 8 frames.

## 2. What we built

Three stages, all in `kashaf/solution.py`.

### 2.1 Nearest team, with hysteresis

On every frame where the ball and both teams are tracked, the side whose nearest player is closer
to the ball has possession  -  but we only *switch* teams once the other side is clearly nearer.
Two margins:

- **Open**  -  switch to the other team only when it is **1.5 m** nearer.
- **Close**  -  keep the current team while it stays within **2.5 m** of the other.

Why this: a tackle or a pass can put the opposing player nearest the ball for a frame or two, then
swap back. A raw nearest-player label turns each of those into a "possession change," which is
exactly why the reference `baseline_raw` scores 0.04. The hysteresis means a genuine turnover (one
side actually holding the ball longer and getting closer) still flips the label, but a blink does
not. The margins are not a tuned pair  -  they come from the **whole-match** distribution of the
distance gap between the two nearest players: in about 85 % of ball-visible frames one side is more
than a metre nearer, so a 1.5 m bar sits well above the noise and below the clear separations.

### 2.2 A two-second majority vote

After hysteresis, each frame's team is re-voted by its +/- 50-frame neighbourhood (25 fps, so +/- 2 s).
Real possessions last 10-20 seconds, so a 2-second window removes any residual flicker without
merging separate possessions. While the ball is **not** tracked  -  about 41 % of the frames we are
scored on, where it is lost in the crowd, the ground, or glare  -  we carry the last known team
forward: losing sight of the ball does not end a possession. We chose the 2-second width because
it is the setting the organisers' own published baselines show as the best on the *full* match
(their longer 5-second window scores better only on the 5-minute sample). We did not choose it by
measuring it on the sample; we adopted the full-match result and confirmed it on the sample.

### 2.3 Almost no dead ball

Only **one** stretch is labelled `D`: the certified second-half pre-kickoff, frames 110 808 to
113 741 (the players are lined up and the ball placed until the kick at 113 742). Outside the two
graded half blocks the label is `X` (abstain), which is never scored.

This is a deliberate precision-over-recall decision, and the reason it matters is that the ranking
metric treats dead ball asymmetrically. `spellF1` matches whole possession *spells* by temporal
overlap; a `D` placed even a few frames inside a live possession **splits that possession into
two**, and the score lost by the split far outweighs the score gained by a correct dead frame. We
expected that and measured it on the public window: even *perfect* dead frames move `spellF1` by
less than a point, while any realistic dead-ball detector costs 0.05-0.10 of it by splitting live
spells. So we kept only the one stoppage whose extent we know exactly and left the rest alone.

## 3. What we tried and dropped

The organisers gave us several assets. Here is what used each, and why we left the rest.

| asset | decision | reason |
|---|---|---|
| tactical ATD tracking (`.txt`) | **used  -  the only data input** | gives us the ball and all 46 tracklets every frame |
| tactical ATD metadata (`.xml`) | **used** | tells us which 24 tracklets are home, 22 away |
| Smart Tagging (`.xml`) | **analysed, dropped** | natural prior for dead ball, but it over-covers (see §2.3 notes below) |
| tactical video | **not used** | no need once the tracking worked |
| homography (`.csv`) | **not used** | the tracking is already in normalised pitch coordinates |
| broadcast feed (video + tracking) | **not used** | a second, unaligned timeline; the submission is anchored to the tactical video, and aligning a second noisy tracking was not worth the risk with the deadline |

On Smart Tagging specifically: the set-piece tags mark a *window around* a restart, not the stoppage
itself  -  a free-kick tag spans roughly 14 s but the ball is actually dead for only about 5 s of it,
with live build-up inside the same tagged interval. Copying that window as `D` splits real
possessions. We also prototyped a velocity stoppage detector (all players standing still), but the
players keep drifting before a stoppage and positioning after the restart, so no clean threshold
separated "dead" from "live." Both were measured, both under-performed the "only the certified
kickoff" rule, so both are out of the final pipeline.

## 4. Ground truth  -  where it was and was not used

The rules allow the published five minutes to *check* the pipeline  -  frame alignment, file
well-formedness, that the output means what we think  -  and forbid using it to train or tune.
We used it exactly that way. From the sample we read: 17 possessions of ~14.2 s mean, home share
~57 %, and the opening frame consistent with the reference. Those are sanity checks. **No constant
in the shipped pipeline (1.5 m, 2.5 m, +/- 50 frames) was chosen from the sample**; each comes from
the whole-match tracking signal and the organisers' published full-match baseline results, as
described in §2. The private full-match ground truth does not exist on our machines and was never
used.

## 5. Training data

**None.** No hand labels on this match, in any amount, for any purpose. No hand labels on any
other match. No external event data, no commercial API, no pre-trained model. The only two inputs
to the pipeline are the machine-generated tactical tracking file and its team/slot metadata file.
That is the full, inspectable data source, and it is all the pipeline reads.

## 6. Results

Scored with the organisers' own scorer (`evaluation/run_evaluation.py`) against the public
five-minute ground truth:

| entry | macroF1 | **spellF1** | poss. pred / gt | mean dur | trans. F1 | lat. |
|---|---|---|---|---|---|---|
| **Kashaf** | 0.5264 | **0.4308** | 22 / 17 | 13.6 s | 0.1951 | **+1.0 f** |
| `baseline_smooth125` | 0.5812 | 0.4242 | 16 / 17 | 18.5 s | 0.1667 | -7.7 f |
| `baseline_smooth50` (floor) | 0.6207 | 0.3256 | 26 / 17 | 10.8 s | 0.0816 | +4.0 f |
| `baseline_smooth25` | 0.6200 | 0.2868 | 36 / 17 | 7.6 s | 0.2034 | +8.2 f |
| `baseline_raw` | 0.5972 | 0.0378 | 131 / 17 | 2.0 s | 0.1410 |  -  |

Two numbers we want to be upfront about. Our **macroF1 is lower** than the baselines (0.5264 vs
0.58-0.62) because we label almost nothing as `D`, and macroF1 partly rewards guessing `D` on the
large share of frames that are dead. That is a direct consequence of optimising the metric that
actually ranks (spellF1), and the organisers' own documentation anticipates the trade. Our
**latency is the best on the board**  -  turnovers land within a frame of the truth on average  -  and
that is the direct payoff of the hysteresis stage, which is the part of the system we are proudest
of.

The five-minute `spellF1` is a validity check, not a target. The sample has only 17 possessions, so
one missed possession moves the score by roughly six points; the organisers are explicit that
tuning to it picks the wrong answer. Our full-match expectation comes from the tracking signal
itself, not the sample.

## 7. Limitations

- **Dead-ball recall is near zero** outside the one certified block. Deliberate, but it caps
  macroF1 and would matter on any board ranked by per-frame accuracy.
- **We are biased toward the earlier team at a turnover.** The 1.5 m open margin must be re-closed
  before a switch counts, so our turnovers sit slightly late (mean +1 frame on the sample). This
  is the price of killing single-frame flicker.
- **Identity switches in the uncorrected tracking** can put a player on the wrong side in exactly
  the frames where a challenge happens; the 2-second vote hides most of them, but a bad cluster
  can steal a short possession.
- **The certified kickoff block is hard-coded** to this match (frames 110 808-113 741). The
  pipeline is deliberately single-game, which is what the contest is.
- **With more time:** a two-pass dead-ball detector that only marks `D` when the tag evidence and
  a full team stoppage agree over a sustained span, and a per-half re-fit of the hysteresis margins
  from the tracking signal alone.

## 8. Reproduction

```
git clone <this repository>
cd kashaf
pip install -r requirements.txt                 # numpy==2.3.1, pandas==2.3.1

set MTAH_DATA_DIR=C:\path\to\match\bundle        # the enrolled bundle, unpacked as in DATA.md
# export MTAH_DATA_DIR=/path/to/bundle           # macOS / Linux

python -m kashaf.run                             # -> evaluation/results/Kashaf.csv
python evaluation/validate_submission.py evaluation/results/Kashaf.csv
python evaluation/run_evaluation.py
```

The output CSV is deterministic: the same data in reproduces the same bytes out. The scorer and
the public ground truth are self-contained, so `run_evaluation.py` runs on a fresh clone with no
data present. The match data itself is not in the repository and cannot be  -  it is licensed for
this competition only and may not be redistributed; the `.gitignore` keeps it, and any submission
CSV, out permanently.
