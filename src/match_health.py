"""Matchmaking & match-health monitoring: metrics, anomaly alerts, cheater-exposure impact."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from . import config as C
from .db import connect, query


def robust_alerts(daily: pd.DataFrame, col: str, group="region", window=7, z_thr=4.0, direction="up"):
    """Rolling robust z-score (median/MAD of the previous `window` days) per group."""
    out = []
    for g, s in daily.sort_values("day").groupby(group):
        s = s.copy()
        med = s[col].shift(1).rolling(window, min_periods=4).median()
        mad = s[col].shift(1).rolling(window, min_periods=4).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        scale = (1.4826 * mad).clip(lower=0.002 if col == "dc_rate" else 1e-9)
        s["robust_z"] = (s[col] - med) / scale
        s["baseline"] = med
        out.append(s)
    r = pd.concat(out)
    mask = r["robust_z"] > z_thr if direction == "up" else r["robust_z"] < -z_thr
    return r, r[mask]


def run(scores: pd.DataFrame | None = None):
    con = connect()
    lobby = query(con, "lobby_health.sql")
    rows = pd.read_sql_query(
        "SELECT s.match_id, s.player_id, s.queue_time_s, s.disconnected, s.avg_ping_ms, s.kills, p.mmr, "
        "mt.region, DATE(mt.started_at) AS day, CAST(strftime('%H', mt.started_at) AS INT) AS hour "
        "FROM player_match_stats s JOIN matches mt USING(match_id) JOIN players p USING(player_id)", con)
    con.close()
    lobby["day"] = lobby["started_at"].str[:10]
    lobby["wide_lobby"] = lobby["mmr_sd"] > 300      # lobby spread (sd of MMR) well above peak norm

    daily = (rows.groupby(["day", "region"]).agg(matches=("match_id", "nunique"), avg_queue_s=("queue_time_s", "mean"),
                                                 p95_queue_s=("queue_time_s", lambda x: x.quantile(.95)),
                                                 dc_rate=("disconnected", "mean"), ping=("avg_ping_ms", "mean")).reset_index())
    daily = daily.merge(lobby.groupby(["day", "region"]).agg(bot_per_match=("bot_count", "mean"),
                                                            wide_lobby_share=("wide_lobby", "mean")).reset_index(), on=["day", "region"])
    _, dc_alerts = robust_alerts(daily, "dc_rate", direction="up")
    # volume monitor: matches vs trailing median (catches silent telemetry loss)
    full = daily.pivot(index="day", columns="region", values="matches")
    ratio = (full / full.shift(1).rolling(7, min_periods=4).median()).stack().rename("ratio").reset_index()
    vol_alerts = ratio[ratio["ratio"] < 0.5]

    # queue time by rating tier and time of day (problem: long waits for high rating off-peak)
    rows["tier"] = pd.cut(rows["mmr"], [0, 1200, 1500, 1800, 2100, 5000], labels=["<1200", "1200-1500", "1500-1800", "1800-2100", "2100+"])
    rows["period"] = np.where(rows["hour"].between(2, 9), "off-peak (2-9h)", "peak")
    q_tier = rows.groupby(["tier", "period"], observed=True)["queue_time_s"].mean().unstack().round(1)
    lobby["period"] = np.where(lobby["started_at"].str[11:13].astype(int).between(2, 9), "off-peak (2-9h)", "peak")
    lobby_period = lobby.groupby("period").agg(mmr_range=("mmr_range", "mean"), wide_share=("wide_lobby", "mean"),
                                              bots=("bot_count", "mean")).round(3)

    truth = pd.read_csv(C.TRUTH_DIR / "matches_truth.csv")
    inc_days = sorted(lobby.merge(truth, on="match_id").query("server_incident == 1")["day"].unique())
    gap_days = sorted(lobby.merge(truth, on="match_id").query("logging_gap == 1")["day"].unique())
    # logging_gap matches are removed from the warehouse, so use truth on days with missing ids
    expected_gap_day = str((pd.Timestamp(C.START_DATE) + pd.Timedelta(days=C.LOGGING_GAP_DAY)).date())
    expected_inc_day = str((pd.Timestamp(C.START_DATE) + pd.Timedelta(days=C.INCIDENT_DAY)).date())
    health = dict(
        n_matches=int(len(lobby)),
        avg_queue_s=round(float(rows["queue_time_s"].mean()), 1), p95_queue_s=round(float(rows["queue_time_s"].quantile(.95)), 1),
        disconnect_rate_pct=round(100 * float(rows["disconnected"].mean()), 2),
        avg_bots_per_match=round(float(lobby["bot_count"].mean()), 2),
        matches_with_bots_pct=round(100 * float((lobby["bot_count"] > 0).mean()), 1),
        wide_lobby_pct=round(100 * float(lobby["wide_lobby"].mean()), 1),
        alerts=dict(
            disconnect_spikes=[dict(day=r.day, region=r.region, dc_rate_pct=round(100 * r.dc_rate, 2), robust_z=round(r.robust_z, 1))
                               for r in dc_alerts.itertuples()],
            volume_drops=[dict(day=r.day, region=r.region, ratio=round(float(r.ratio), 2)) for r in vol_alerts.itertuples()]),
        injected=dict(network_incident=f"{C.INCIDENT_REGION} {expected_inc_day}", logging_gap=f"{C.LOGGING_GAP_REGION} {expected_gap_day}"),
        queue_by_tier_period=q_tier.reset_index().astype(str).to_dict("records"),
        lobby_by_period=lobby_period.reset_index().to_dict("records"))

    # cheater exposure and the effect of a quarantine (shadow) pool, using out-of-sample flags
    if scores is not None:
        truec = scores.set_index("player_id")["is_cheater"]
        r2 = rows[["match_id", "player_id"]].copy()
        r2["cheater"] = r2["player_id"].map(truec).fillna(False).astype(bool)
        base = r2.groupby("match_id")["cheater"].sum()
        expo = dict(matches_with_cheater_pct=round(100 * float((base > 0).mean()), 1),
                    avg_cheaters_per_match=round(float(base.mean()), 2), policies={})
        for name, tiers in [("quarantine_auto_only", ["AUTO_ACTION"]), ("quarantine_auto_plus_review", ["AUTO_ACTION", "HUMAN_REVIEW"])]:
            flag = scores.set_index("player_id")["tier"].isin(tiers)
            r2["flagged"] = r2["player_id"].map(flag).fillna(False).astype(bool)
            left = r2[~r2["flagged"]].groupby("match_id")["cheater"].sum().reindex(base.index).fillna(0)
            expo["policies"][name] = dict(
                matches_with_cheater_pct=round(100 * float((left > 0).mean()), 1),
                avg_cheaters_per_match=round(float(left.mean()), 2),
                quarantined_players=int(flag.sum()), quarantined_genuine=int((flag & ~truec).sum()),
                cheaters_isolated_pct=round(100 * float((flag & truec).sum() / truec.sum()), 1))
        health["cheater_exposure"] = expo
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(C.OUT_DIR / "daily_health.csv", index=False)
    (C.OUT_DIR / "match_health.json").write_text(json.dumps(health, indent=2, default=str))
    return health, daily, lobby, rows, q_tier


if __name__ == "__main__":
    print(json.dumps(run()[0], indent=1, default=str)[:3000])
