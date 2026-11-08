import numpy as np
import pandas as pd
import pytest
from src import config as C, simulate, ingest, features, detect, matchmaker_sim, match_health
from src.db import connect


@pytest.fixture(scope="module")
def small():
    return simulate.simulate(n_players=500, n_matches=120, seed=3)


def test_simulation_shape_and_no_label_leak(small):
    pl, matches, stats = small
    assert len(stats) == 120 * C.LOBBY_SIZE
    assert stats.groupby("match_id")["player_id"].nunique().eq(C.LOBBY_SIZE).all()
    # labels must never be inside the telemetry the pipeline reads
    assert not {"is_cheater", "cheat_type", "archetype"} & set(stats.columns)
    assert pl["is_cheater"].mean() > 0


def test_validation_quarantines_bad_rows(small):
    _, _, stats = small
    s = stats.head(50).copy()
    s = pd.concat([s, s.head(3)])                           # duplicates
    s.iloc[5, s.columns.get_loc("shots_hit")] = s.iloc[5]["shots_fired"] + 10   # hits > shots
    s.iloc[6, s.columns.get_loc("kills")] = -2
    s.iloc[7, s.columns.get_loc("avg_ping_ms")] = np.nan
    clean, rej, rep = ingest.validate_stats(s)
    assert rep["rows_in"] == 53 and rep["rows_clean"] + rep["rows_rejected"] == 53
    assert set(rej["reason"]) >= {"duplicate_match_player", "hits_exceed_shots", "negative_value", "null_required_field"}
    assert not clean.duplicated(["match_id", "player_id"]).any()


def test_warehouse_and_features(small, tmp_path):
    pl, matches, stats = small
    raw = tmp_path / "raw"; raw.mkdir()
    pl[["player_id", "home_region", "device", "signup_date", "mmr"]].to_csv(raw / "players.csv", index=False)
    matches.drop(columns=["server_incident"]).to_csv(raw / "matches.csv", index=False)
    stats.to_csv(raw / "player_match_stats.csv", index=False)
    db = tmp_path / "t.db"
    rep = ingest.build_warehouse(db_path=db, raw_dir=raw)
    assert rep["rows_clean"] > 0
    con = connect(db); f = features.build_features(con); con.close()
    assert f["player_id"].is_unique
    f = f.merge(pl[["player_id", "archetype", "is_cheater"]], on="player_id")
    f = f[f["eligible"]]
    # genuine elite players sit BELOW the human ceiling; cheaters' persistence evidence is higher
    elite = f[(f["archetype"] == "elite")]
    assert (elite["hs_z"] < 0).mean() > 0.9
    assert f[f["is_cheater"]]["max_z"].median() > f[~f["is_cheater"]]["max_z"].median()


def test_binomial_z_small_samples_are_small():
    # same 40% rate: 5 shots can't produce the evidence 500 shots can
    assert features._binom_z(2, 5, 0.26) < 1.5 < features._binom_z(200, 500, 0.26)


def _row(**kw):
    base = dict(prob=0.99, n_channels=3, flagged_matches=9, eligible=True)
    base.update(kw); return base


def test_policy_hard_rules():
    th = dict(tau_auto=0.9, tau_review=0.3)
    d = pd.DataFrame([
        _row(),                                   # clear case -> AUTO
        _row(eligible=False),                     # too few matches -> never acted on
        _row(n_channels=1),                       # only one evidence channel -> review, not auto
        _row(flagged_matches=2),                  # not persistent -> review, not auto
        _row(prob=0.5, n_channels=0, flagged_matches=0),   # mid prob -> review
        _row(prob=0.01, n_channels=0, flagged_matches=0),  # clean
    ])
    t = detect.apply_policy(d, th)["tier"].tolist()
    assert t == ["AUTO_ACTION", "INSUFFICIENT_DATA", "HUMAN_REVIEW", "HUMAN_REVIEW", "HUMAN_REVIEW", "CLEAR"]


def test_progressive_window_cuts_queue_time():
    fixed = matchmaker_sim.simulate("fixed", 0.5, minutes=40)
    prog = matchmaker_sim.simulate("progressive", 0.5, minutes=40)
    assert prog["avg_queue_s"] < fixed["avg_queue_s"]


def test_robust_alert_catches_spike():
    days = pd.date_range("2026-10-01", periods=20).strftime("%Y-%m-%d")
    dc = np.full(20, 0.027) + np.random.default_rng(0).normal(0, 0.002, 20); dc[14] = 0.3
    daily = pd.DataFrame(dict(day=days, region="R", dc_rate=dc))
    _, alerts = match_health.robust_alerts(daily, "dc_rate")
    assert list(alerts["day"]) == [days[14]]
