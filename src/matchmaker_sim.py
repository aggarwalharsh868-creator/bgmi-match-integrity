"""Queue simulation: fixed MMR window vs progressive (wait-time based) widening.

Problem: a fixed window gives tight, fair lobbies but top-rated players wait forever and
the game fills lobbies with bots. Fix: start tight, widen the window as a player waits, with a
hard cap so lobbies never become wildly unfair. Simulated per tick (5s) for a single region.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C

TICK = 5.0
BOT_AFTER_S = 90.0
MIN_HUMANS = 30


def window(strategy: str, wait: float) -> float:
    if strategy == "fixed":
        return 150.0
    return min(100.0 + 6.0 * wait, 500.0)        # progressive: +6 MMR per second waited, cap 500


def simulate(strategy: str, rate_per_s: float, minutes=90, seed=7, lobby=C.LOBBY_SIZE):
    rng = np.random.default_rng(seed)
    mmr = np.empty(0); t_arr = np.empty(0)
    waits, lobby_sd, bots, off_pairs = [], [], [], []
    n_ticks = int(minutes * 60 / TICK)
    for k in range(n_ticks):
        now = k * TICK
        n_new = rng.poisson(rate_per_s * TICK)
        mmr = np.concatenate([mmr, rng.normal(1500, 300, n_new)])
        t_arr = np.concatenate([t_arr, np.full(n_new, now)])
        order = np.argsort(t_arr)
        used = np.zeros(len(mmr), dtype=bool)
        for idx in order[:60]:
            if used[idx]:
                continue
            wait = now - t_arr[idx]
            w = window(strategy, wait)
            cand = np.where((~used) & (np.abs(mmr - mmr[idx]) <= w))[0]
            if len(cand) >= lobby or (wait >= BOT_AFTER_S and len(cand) >= MIN_HUMANS):
                pick = cand[np.argsort(np.abs(mmr[cand] - mmr[idx]))[:lobby]]
                used[pick] = True
                waits.extend(now - t_arr[pick]); lobby_sd.append(mmr[pick].std())
                bots.append(lobby - len(pick))
                off_pairs.extend(np.abs(mmr[pick] - mmr[pick].mean()) > 400)
        mmr, t_arr = mmr[~used], t_arr[~used]
    waits = np.array(waits)
    return dict(strategy=strategy, players_per_s=rate_per_s, lobbies=len(bots),
                avg_queue_s=round(float(waits.mean()), 1), p95_queue_s=round(float(np.percentile(waits, 95)), 1),
                avg_bots_per_lobby=round(float(np.mean(bots)), 2), lobby_mmr_sd=round(float(np.mean(lobby_sd)), 1),
                unfair_slot_pct=round(100 * float(np.mean(off_pairs)), 2), still_waiting=int(len(mmr)))


def compare():
    rows = []
    for name, rate in [("peak", 4.0), ("off-peak", 0.5)]:
        for strat in ["fixed", "progressive"]:
            rows.append(dict(scenario=name, **simulate(strat, rate)))
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_DIR / "matchmaker_comparison.csv", index=False)
    return df


if __name__ == "__main__":
    print(compare().to_string())
