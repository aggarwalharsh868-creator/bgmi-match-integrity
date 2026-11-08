"""Synthetic BGMI-style telemetry generator.

Real BGMI telemetry is not public, so we simulate it with realistic structure:
heavy-tailed play frequency, skill-based lobbies, 3 cheat types that only fire in
some matches, and deliberately *hard genuine cases* a naive detector would punish:
  elite   - genuinely top-1% players (high K/D, high headshots)
  smurf   - skilled players on fresh low-MMR accounts (crush their lobbies)
  high_ping - lag creates fake "speed hack" spikes
  new     - accounts with very few matches (tiny samples = noisy stats)
Also injected: a regional network incident, a telemetry logging gap, and dirty rows.
Ground truth is written to a separate folder and used for evaluation only.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C

ARCHETYPES = ["normal", "elite", "smurf", "high_ping", "new"]


def make_players(rng, n):
    arche = rng.choice(ARCHETYPES, size=n, p=[0.80, 0.04, 0.03, 0.08, 0.05])
    skill = rng.normal(0, 1, n)
    m = arche == "elite"; skill[m] = rng.normal(2.0, 0.25, m.sum())
    m = arche == "smurf"; skill[m] = rng.normal(1.7, 0.3, m.sum())
    can_cheat = ~np.isin(arche, ["elite", "smurf"])
    is_cheater = can_cheat & (rng.random(n) < C.CHEATER_RATE)
    skill[is_cheater] = rng.normal(-0.1, 0.9, is_cheater.sum())
    ctype = np.where(is_cheater, rng.choice(["aimbot", "wallhack", "speedhack", "aimbot+wallhack"],
                                           size=n, p=[0.35, 0.25, 0.20, 0.20]), "none")
    intensity = np.where(is_cheater, rng.uniform(0.2, 1.0, n), 0.0)
    cheat_prob = np.where(is_cheater, rng.uniform(0.15, 0.8, n), 0.0)
    mmr = 1500 + 300 * skill + rng.normal(0, 60, n)
    sm = arche == "smurf"
    mmr[sm] = 1500 + 300 * rng.normal(-0.6, 0.5, sm.sum())      # fresh account, low rating
    mmr[is_cheater] += 150                                      # cheaters inflate their rating
    act = rng.lognormal(0, 0.9, n)
    act *= np.where(arche == "new", 0.06, 1.0) * np.where(arche == "elite", 1.5, 1.0) \
        * np.where(is_cheater, 1.2, 1.0)
    base_ping = rng.lognormal(np.log(50), 0.35, n)
    base_loss = rng.gamma(1.2, 0.35, n)
    hp = arche == "high_ping"
    base_ping[hp] = rng.lognormal(np.log(150), 0.25, hp.sum())
    base_loss[hp] *= 3
    home = rng.choice(C.REGIONS, size=n, p=C.REGION_P)
    day0 = pd.Timestamp(C.START_DATE)
    signup = day0 - pd.to_timedelta(rng.integers(30, 900, n), unit="D")
    nw = arche == "new"
    signup = pd.Series(signup)
    signup[nw] = day0 + pd.to_timedelta(rng.integers(20, 28, nw.sum()), unit="D")
    return pd.DataFrame(dict(
        player_id=np.arange(1, n + 1), archetype=arche, skill=skill, is_cheater=is_cheater,
        cheat_type=ctype, intensity=intensity, cheat_prob=cheat_prob, mmr=mmr.round(1),
        activity=act, base_ping=base_ping, base_loss=base_loss, home_region=home,
        device=rng.choice(["android_low", "android_mid", "android_high", "ios"], size=n,
                          p=[0.25, 0.35, 0.28, 0.12]),
        signup_date=signup.dt.strftime("%Y-%m-%d").values))


def _simulate_lobby(rng, P, ids, inc, offpeak):
    k = len(ids)
    s = P["skill"][ids]; ping_b = P["base_ping"][ids]; loss_b = P["base_loss"][ids]
    inten = P["intensity"][ids]
    act = P["is_cheater"][ids] & (rng.random(k) < P["cheat_prob"][ids])
    aim = act & P["has_aim"][ids]; wall = act & P["has_wall"][ids]; spd = act & P["has_speed"][ids]

    ping = ping_b * rng.lognormal(0, 0.2, k) * (3.5 if inc else 1.0)
    loss = np.clip(loss_b * rng.lognormal(0, 0.4, k) * (3.0 if inc else 1.0), 0, 30)
    lam = 1.0 * np.exp(0.38 * s) * (1 + 1.6 * inten * aim) * (1 + 0.7 * inten * wall)
    kills = np.minimum(rng.poisson(lam), 30)
    fired = rng.poisson(45 + 22 * kills) + 5
    p_hit = np.clip(0.22 + 0.03 * s + rng.normal(0, 0.02, k) + 0.20 * inten * aim, 0.05, 0.9)
    hit = rng.binomial(fired, p_hit)
    p_hs = np.clip(0.13 + 0.022 * s + rng.normal(0, 0.015, k) + 0.38 * inten * aim, 0.03, 0.85)
    hs = rng.binomial(hit, p_hs)
    react = 300 - 22 * s + 0.5 * ping + rng.normal(0, 20, k)
    react = np.where(aim, (300 - 22 * s) * (1 - 0.5 * inten) + 0.5 * ping + rng.normal(0, 20, k), react)
    react = np.clip(react, 60, None)
    react = np.where(hit > 0, react, np.nan)
    r_wall = np.clip(0.012 + 0.004 * s + rng.normal(0, 0.004, k), 0.002, 0.04)
    r_wall = np.where(wall, 0.08 + 0.25 * inten, r_wall)
    wall_k = rng.binomial(kills, r_wall)
    speed = 1 + np.abs(rng.normal(0, 0.012, k))
    spike_p = 0.003 + 0.05 * loss + 0.2 * np.clip((ping - 100) / 200, 0, 1)
    spike = rng.random(k) < np.clip(spike_p, 0, 0.6)
    speed = np.where(spike, speed + rng.uniform(0.10, 0.45, k), speed)
    speed = np.where(spd, 1.10 + rng.uniform(0, 0.6, k) * inten + 0.02, speed)
    p_dc = 0.008 + 0.00012 * ping + 0.02 * loss
    p_dc = np.clip(p_dc * (6 if inc else 1), 0, 0.9)
    dc = (rng.random(k) < p_dc).astype(int)
    perf = 0.6 * s + 0.12 * kills + 0.4 * inten * wall + rng.normal(0, 1, k)
    place = (np.argsort(np.argsort(-perf)) + 1)
    surv = np.clip(rng.normal(700 + 140 * s + 25 * kills, 180, k), 30, 1800)
    surv = np.where(dc == 1, surv * rng.uniform(0.1, 0.7, k), surv)
    dmg = np.round(hit * 30 + hs * 25 + rng.normal(0, 40, k)).clip(0)
    reports = rng.poisson(0.01 + 0.010 * kills + 0.30 * (aim | wall | spd))
    tier_z = (P["mmr"][ids] - 1500) / 300
    q = (25 * np.exp(0.5 * np.maximum(np.abs(tier_z) - 0.5, 0)) * (1 + 1.5 * offpeak)
         * rng.lognormal(0, 0.3, k))
    return dict(placement=place, kills=kills, damage=dmg.astype(int), shots_fired=fired,
                shots_hit=hit, headshot_hits=hs, avg_reaction_ms=np.round(react, 1),
                wall_kills=wall_k, max_speed_ratio=np.round(speed, 3),
                avg_ping_ms=np.round(ping, 1), packet_loss_pct=np.round(loss, 2),
                queue_time_s=np.round(q, 1), survival_s=surv.astype(int), disconnected=dc,
                reports_received=reports), q.mean()


def simulate(n_players=C.N_PLAYERS, n_matches=C.N_MATCHES, seed=C.SEED):
    rng = np.random.default_rng(seed)
    pl = make_players(rng, n_players)
    ct = pl["cheat_type"].astype(str)
    P = {c: pl[c].values for c in ["skill", "is_cheater", "intensity", "cheat_prob", "mmr",
                                   "base_ping", "base_loss"]}
    P["has_aim"] = ct.str.contains("aimbot").values
    P["has_wall"] = ct.str.contains("wallhack").values
    P["has_speed"] = (ct == "speedhack").values
    mmr = P["mmr"]; act = pl["activity"].values; home = pl["home_region"].values
    act_p = act / act.sum()

    hour_w = np.array([2, 1.5, 1, 1, 1, 1, 1.5, 2, 3, 3.5, 4, 4, 4, 4, 4, 4.5, 5, 6, 7, 8, 9, 9, 7, 4.0])
    day = rng.integers(0, C.DAYS, n_matches)
    hour = rng.choice(24, size=n_matches, p=hour_w / hour_w.sum())
    minute = rng.integers(0, 60, n_matches)
    ts = pd.Timestamp(C.START_DATE) + pd.to_timedelta(day * 1440 + hour * 60 + minute, unit="m")
    order = np.argsort(ts.values)
    ts = ts[order]; hour = hour[order]; day = day[order]
    regs = rng.choice(C.REGIONS, size=n_matches, p=C.REGION_P)

    rows, mrows = [], []
    for mid in range(n_matches):
        offpeak = 1 if 2 <= hour[mid] <= 9 else 0
        w = rng.uniform(450, 900) if offpeak else rng.uniform(180, 320)
        anchor = rng.choice(mmr, p=act_p)
        wts = act * np.exp(-0.5 * ((mmr - anchor) / w) ** 2) * np.where(home == regs[mid], 1.0, 0.05)
        ids = rng.choice(n_players, size=C.LOBBY_SIZE, replace=False, p=wts / wts.sum())
        inc = int(regs[mid] == C.INCIDENT_REGION and day[mid] == C.INCIDENT_DAY and hour[mid] >= 12)
        d, qmean = _simulate_lobby(rng, P, ids, inc, offpeak)
        d["match_id"] = mid + 1; d["player_id"] = ids + 1
        rows.append(pd.DataFrame(d))
        bots = int(min(30, rng.poisson(max(0.0, (qmean - 60) / 12))))
        mrows.append(dict(match_id=mid + 1, started_at=ts[mid].strftime("%Y-%m-%d %H:%M:%S"),
                          region=regs[mid], bot_count=bots,
                          duration_s=int(np.max(d["survival_s"])), server_incident=inc))
    stats = pd.concat(rows, ignore_index=True)
    cols = ["match_id", "player_id", "placement", "kills", "damage", "shots_fired", "shots_hit",
            "headshot_hits", "avg_reaction_ms", "wall_kills", "max_speed_ratio", "avg_ping_ms",
            "packet_loss_pct", "queue_time_s", "survival_s", "disconnected", "reports_received"]
    stats = stats[cols]
    matches = pd.DataFrame(mrows)
    return pl, matches, stats


def write_raw(pl, matches, stats, seed=C.SEED):
    """Write raw CSVs the way a game backend would export them: messy on purpose."""
    rng = np.random.default_rng(seed + 1)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True); C.TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    # logging gap: most telemetry for one region/day silently lost
    ts = pd.to_datetime(matches["started_at"])
    gap = (matches["region"] == C.LOGGING_GAP_REGION) & ((ts - ts.dt.normalize().min()).dt.days == C.LOGGING_GAP_DAY)
    drop_ids = matches.loc[gap & (rng.random(len(matches)) < 0.7), "match_id"]
    matches_raw = matches[~matches["match_id"].isin(drop_ids)].drop(columns=["server_incident"])
    stats = stats[~stats["match_id"].isin(drop_ids)].copy()
    # dirty rows
    n = len(stats)
    dups = stats.sample(int(0.003 * n), random_state=seed)
    stats.loc[stats.sample(int(0.002 * n), random_state=seed + 2).index, "avg_ping_ms"] = np.nan
    bad = stats.sample(int(0.001 * n), random_state=seed + 3).index
    stats.loc[bad, ["shots_hit", "shots_fired"]] = stats.loc[bad, ["shots_fired", "shots_hit"]].values
    neg = stats.sample(int(0.0005 * n), random_state=seed + 4).index
    stats.loc[neg, "kills"] = -1
    stats = pd.concat([stats, dups], ignore_index=True).sample(frac=1, random_state=seed)
    pl[["player_id", "home_region", "device", "signup_date", "mmr"]].to_csv(C.RAW_DIR / "players.csv", index=False)
    matches_raw.to_csv(C.RAW_DIR / "matches.csv", index=False)
    stats.to_csv(C.RAW_DIR / "player_match_stats.csv", index=False)
    pl[["player_id", "archetype", "is_cheater", "cheat_type", "intensity", "cheat_prob", "skill"]] \
        .to_csv(C.TRUTH_DIR / "players_truth.csv", index=False)
    matches[["match_id", "server_incident"]].assign(logging_gap=matches["match_id"].isin(drop_ids).astype(int)) \
        .to_csv(C.TRUTH_DIR / "matches_truth.csv", index=False)
    return dict(rows=len(stats), dropped_matches=len(drop_ids))


def generate(n_players=C.N_PLAYERS, n_matches=C.N_MATCHES, seed=C.SEED):
    pl, matches, stats = simulate(n_players, n_matches, seed)
    return write_raw(pl, matches, stats, seed)


if __name__ == "__main__":
    print(generate())
