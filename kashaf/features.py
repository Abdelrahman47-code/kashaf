"""Fast vectorized feature extraction from the tactical ATD and Smart Tagging.

The raw ATD line is
    "<frame>:g1;g2;...;g46:<ball>"   with each group "x,y,s" or "NaN,NaN,NaN"

Re-writing ':' and ';' as ',' turns every line into 1 + 46*3 + 3 = 142
comma-separated fields, so the whole file parses with one pandas pass:

    frame, p0x p0y p0s, p1x p1y p1s, ..., p45x p45y p45s, bxx byy bss

NaN parses to np.nan automatically. This gives 3-D position arrays
(N_frames, 46, 2) and (N_frames, 3) for the ball in ~10 seconds instead of
a minute-plus of python line parsing, and leaves room for derived features.

Everything downstream (dead detection, possession state model) is plain
numpy on these arrays, so the whole pipeline reproduces with numpy+pandas.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

N_SLOTS = 46


def _parse_atd(raw_path, n_frames):
    """(players (N, S, 2), ball (N, 3)) with NaN where the tracker had nothing.

    positions are normalised 0-1 exactly as in the file; multiply by games.py
    pitch dimensions when a distance in metres is needed.
    """
    with open(raw_path, "r", newline="") as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",")
    # 1 + 46*3 + 3 = 142 columns; NaN parses to np.nan automatically.
    df = pd.read_csv(io.StringIO(text), header=None)
    assert len(df) == n_frames, "expected {} records, parsed {}".format(n_frames, len(df))
    assert int(df.iloc[0, 0]) == 1 and int(df.iloc[-1, 0]) == n_frames

    m = df.to_numpy(float)                                   # (N, 142)
    players = m[:, 1:1 + 3 * N_SLOTS]                        # (N, 138)
    ball = m[:, 1 + 3 * N_SLOTS:]                            # (N, 3)
    players = players.reshape(n_frames, N_SLOTS, 3)          # x, y, s per slot
    return players[..., :2], players[..., 2], ball


CACHE_DIR = os.path.join(ROOT, ".cache")
CACHE_PATH = os.path.join(CACHE_DIR, "features.npz")


def load_features(game=GAME, data_dir=None, use_cache=True):
    """All per-frame signals the pipeline needs, as one dict of numpy arrays.

    The full ATD parse takes ~15 s, and tuning runs reload it many times, so
    the derived arrays are cached to disk (never committed, never a leak:
    same data the submission pipeline itself reads).
    """
    if use_cache and os.path.exists(CACHE_PATH):
        with np.load(CACHE_PATH) as z:
            f = {k: z[k] for k in z.files}
        return f
    return _load_features_uncached(game, data_dir)


CACHE_KEYS = ("n_frames", "ball", "ball_speed", "ball_vel", "ball_present",
              "d_home", "d_away", "crowd_home_2", "crowd_away_2",
              "player_vel", "vel_home", "vel_away", "n_players",
              "med_player_vel", "n_moving", "vel_home_med", "vel_away_med",
              "n_home_slots", "n_away_slots")


def _load_features_uncached(game=GAME, data_dir=None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]

    players_norm, players_speed, ball_norm = _parse_atd(raw_path, n_frames)

    # metres. The third ball column is the tracker's speed estimate and is only
    # filled for a minority of frames (it needs two positions); ball presence is
    # taken from the first position coordinate, matching the documented 56.9%
    # and the starter baseline's check.
    scale = np.array([[length, width]])
    players = players_norm * scale                    # (N, S, 2)
    ball = ball_norm[:, :2] * scale                   # (N, 2)
    present = ~np.isnan(ball_norm[:, 0])
    ball_speed = ball_norm[:, 2]                      # m/s, often NaN
    ball_speed = np.where(present, ball_speed, np.nan)

    home = _team_of_each_track(meta_path)             # (S,) 0/1
    n_home = int(home.sum())
    n_away = N_SLOTS - n_home

    # per-slot distance to the ball, then reduced per side. A side with no
    # tracked slot on a frame becomes NaN, not inf, so statistics stay clean.
    d_all = _slot_dists(players, ball)                # (N, S) m
    d_home = np.nanmin(np.where(home[None, :], d_all, np.inf), axis=1)
    d_away = np.nanmin(np.where(~home[None, :], d_all, np.inf), axis=1)
    d_home[np.isposinf(d_home)] = np.nan
    d_away[np.isposinf(d_away)] = np.nan

    # velocities from position deltas (m/s), NaN where a delta is impossible.
    fps = GAMES[game]["fps"]
    prev = np.roll(ball, 1, axis=0)
    prev[0] = np.nan
    ball_vel = np.linalg.norm(ball - prev, axis=1) * fps
    ball_vel[np.isnan(ball).any(axis=1) | np.isnan(prev).any(axis=1)] = np.nan

    pprev = np.roll(players, 1, axis=0)
    pprev[0] = np.nan
    pvel = np.linalg.norm(players - pprev, axis=2) * fps
    pvel[np.isnan(players).any(axis=2) | np.isnan(pprev).any(axis=2)] = np.nan

    n_players = (~np.isnan(players[..., 0])).sum(axis=1)
    tracked = np.isfinite(pvel)                      # (N, S)
    with np.errstate(invalid="ignore", divide="ignore"):
        vel_home = np.nanmean(pvel[:, home], axis=1)
        vel_away = np.nanmean(pvel[:, ~home], axis=1)
        med_player_vel = np.nanmedian(pvel, axis=1)
        med_vel_home = np.nanmedian(pvel[:, home], axis=1)
        med_vel_away = np.nanmedian(pvel[:, ~home], axis=1)
    n_moving = (pvel > 0.5).sum(axis=1).astype(np.float64)
    vel_home[~tracked[:, home].any(axis=1)] = np.nan
    vel_away[~tracked[:, ~home].any(axis=1)] = np.nan
    med_player_vel[~tracked.any(axis=1)] = np.nan
    med_vel_home[~tracked[:, home].any(axis=1)] = np.nan
    med_vel_away[~tracked[:, ~home].any(axis=1)] = np.nan

    f = {
        "n_frames": n_frames,
        "ball": ball,                                 # (N, 2) m, NaN when missing
        "ball_speed": ball_speed,                     # (N,) m/s
        "ball_vel": ball_vel,                         # (N,) m/s
        "ball_present": present,
        "d_home": d_home,                             # (N,) m
        "d_away": d_away,                             # (N,) m
        "crowd_home_2": np.nansum((d_all <= 2.0) & home[None, :], axis=1),
        "crowd_away_2": np.nansum((d_all <= 2.0) & ~home[None, :], axis=1),
        "player_vel": pvel,                           # (N, S) m/s
        "n_moving": n_moving,                         # (N,) players > 0.5 m/s
        "vel_home": vel_home,                         # (N,) m/s
        "vel_away": vel_away,                         # (N,) m/s
        "med_player_vel": med_player_vel,             # (N,) m/s
        "vel_home_med": med_vel_home,                 # (N,) m/s
        "vel_away_med": med_vel_away,                 # (N,) m/s
        "n_players": n_players,                       # (N,)
        "n_home_slots": n_home,
        "n_away_slots": n_away,
    }

    save = {k: f[k] for k in CACHE_KEYS}
    save["pitch"] = np.array([length, width])
    np.savez_compressed(CACHE_PATH, **save)
    return f


def _slot_dists(players, ball):
    """Per-slot distance to the ball, in metres.

    NaN wherever the ball is missing (nothing can be measured). Untracked player
    slots count as infinite distance so per-side minima skip them.
    """
    d = np.linalg.norm(
        np.where(np.isnan(players), np.inf, players) - ball[:, None, :], axis=2)
    d[np.isnan(ball[:, 0])] = np.nan
    return d


def _team_of_each_track(meta_path):
    """0 = home team, 1 = away, per track slot. See starter/baseline.py."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    assert len(team_ids) == 2
    home_id = team_ids[0]
    return np.array([team_id == home_id for team_id in
                     (player.get("teamId") for player in root.iter("Player"))])


def load_smart_tagging(game=GAME, data_dir=None):
    """Smart Tagging intervals as video frames + tag dict. Stdlib only."""
    from starter.load_smart_tagging import load
    return load(game, data_dir)


def tagging_arrays(events, n_frames):
    """Dense per-frame arrays derived from the Smart Tagging intervals.

    - "set_piece" is 1 inside any SET PIECES interval (restart = end of a D run)
    - "phase_team" is 0/1/None per frame from the Team tag of the covering phase
      (None where N/A or uncovered)
    """
    n = n_frames
    set_piece = np.zeros(n, dtype=bool)
    phase_team = np.full(n, -1, dtype=np.int8)
    team_map = {"barcelona": 0, "atletic club": 1, "athletic club": 1}
    for ev in events:
        s, e = ev["start_frame"], ev["end_frame"]
        s, e = max(0, s), min(n, e)
        if s >= e:
            continue
        code = (ev["code"] or "").strip().upper()
        if code == "SET PIECES":
            set_piece[s:e] = True
        team = (ev["tags"].get("Team") or "").strip().lower()
        if team in team_map:
            phase_team[s:e] = team_map[team]
    return set_piece, phase_team
