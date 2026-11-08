"""Player-level evidence. SQL does the heavy aggregation, pandas does the statistics.

Key idea: every signal is a *z-score against the human ceiling* computed from binomial /
normal sampling error, so a player with 5 matches can never produce a huge z by luck, and a
genuine elite player (high but below the ceiling) produces a negative z.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C
from .db import connect, query

CHANNELS = ["aim_x", "react_x", "wall_x", "speed"]   # independent evidence channels


def _binom_z(x, n, p0):
    n = np.maximum(n, 1)
    return (x - n * p0) / np.sqrt(n * p0 * (1 - p0))


def build_features(con=None) -> pd.DataFrame:
    own = con is None
    con = con or connect()
    params = dict(ping_max=C.CLEAN_PING_MAX, loss_max=C.CLEAN_LOSS_MAX, p_hs=C.CEIL_HS,
                  react_match_ceil=C.REACT_MATCH_CEIL, speed_limit=C.SPEED_LIMIT)
    f = query(con, "player_features.sql", params)
    pl = pd.read_sql_query("SELECT player_id, mmr FROM players", con)
    if own:
        con.close()
    f = f.merge(pl, on="player_id", how="left")
    f["hs_rate"] = f["hs_hits"] / f["shots_hit"].clip(lower=1)
    f["acc"] = f["shots_hit"] / f["shots_fired"].clip(lower=1)
    f["wall_rate"] = f["wall_kills"] / f["kills"].clip(lower=1)
    f["kills_pm"] = f["kills"] / f["n_matches"]
    f["reports_pm"] = f["reports"] / f["n_matches"]
    f["dc_rate"] = f["disconnects"] / f["n_matches"]
    # statistical evidence vs. the human ceiling
    f["hs_z"] = _binom_z(f["hs_hits"], f["shots_hit"], C.CEIL_HS)
    f["acc_z"] = _binom_z(f["shots_hit"], f["shots_fired"], C.CEIL_ACC)
    f["wall_z"] = _binom_z(f["wall_kills"], f["kills"], C.CEIL_WALL)
    f["react_z"] = ((C.CEIL_REACT_MS - f["react_adj_mean"]) * np.sqrt(f["react_n"]) / C.REACT_SD).fillna(0)
    f["speed_z"] = _binom_z(f["clean_speed_viol"], f["clean_matches"], C.CLEAN_SPEED_VIOL_RATE)
    f.loc[f["clean_matches"] < 5, "speed_z"] = 0.0     # no clean-network evidence -> no speed claim
    # persistence evidence: cheaters toggle, so averages dilute them. Count *extreme matches*
    # against the false-flag rate a clean player produces (C.FLAG_NULL_RATE).
    for col, name in [("f_aim_m", "aim_x"), ("f_react_m", "react_x"), ("f_wall_m", "wall_x")]:
        f[f"{name}_z"] = _binom_z(f[col], f["n_matches"], C.FLAG_NULL_RATE)
    # performance relative to rating-tier peers (what a naive K/D rule looks at)
    f["tier"] = pd.qcut(f["mmr"], 10, labels=False, duplicates="drop")
    g = f.groupby("tier")["kills_pm"]
    med = g.transform("median"); mad = g.transform(lambda s: (s - s.median()).abs().median()) * 1.4826
    f["perf_z"] = (f["kills_pm"] - med) / mad.clip(lower=0.05)
    f["log_matches"] = np.log1p(f["n_matches"])
    z = [f"{c}_z" for c in CHANNELS]
    f["n_channels"] = (f[z] > C.Z_STRONG).sum(axis=1)
    f["max_z"] = f[z].max(axis=1)
    f["eligible"] = f["n_matches"] >= C.MIN_MATCHES
    return f


MODEL_FEATURES = ["hs_z", "acc_z", "react_z", "wall_z", "speed_z", "hs_rate", "acc", "wall_rate",
                  "kills_pm", "reports_pm", "perf_z", "ping_mean", "loss_mean", "dc_rate",
                  "log_matches", "flagged_matches", "f_aim_m", "f_react_m", "f_wall_m",
                  "clean_speed_viol", "aim_x_z", "react_x_z", "wall_x_z", "n_channels", "max_z"]
