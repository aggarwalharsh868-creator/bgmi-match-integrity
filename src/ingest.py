"""Raw CSV -> validated SQLite warehouse. Bad rows are quarantined, not silently dropped."""
from __future__ import annotations
import json, sqlite3
import pandas as pd
from . import config as C


def validate_stats(df: pd.DataFrame):
    """Return (clean, rejected) where rejected has a `reason` column."""
    df = df.copy()
    n0 = len(df)
    dup_mask = df.duplicated(["match_id", "player_id"], keep="first")
    rejected = [df[dup_mask].assign(reason="duplicate_match_player")]
    df = df[~dup_mask]
    rules = {
        "null_required_field": df[["avg_ping_ms", "packet_loss_pct", "kills", "shots_hit", "shots_fired",
                                   "headshot_hits", "wall_kills", "max_speed_ratio"]].isna().any(axis=1),
        "negative_value": (df[["kills", "shots_fired", "shots_hit", "headshot_hits", "wall_kills"]] < 0).any(axis=1),
        "hits_exceed_shots": df["shots_hit"] > df["shots_fired"],
        "headshots_exceed_hits": df["headshot_hits"] > df["shots_hit"],
        "wall_kills_exceed_kills": df["wall_kills"] > df["kills"],
        "impossible_speed": (df["max_speed_ratio"] < 0.5) | (df["max_speed_ratio"] > 5),
    }
    bad = pd.Series(False, index=df.index)
    for reason, mask in rules.items():
        m = mask & ~bad
        if m.any():
            rejected.append(df[m].assign(reason=reason))
        bad |= mask
    clean = df[~bad]
    rej = pd.concat(rejected, ignore_index=True) if rejected else df.iloc[0:0].assign(reason="")
    report = dict(rows_in=n0, rows_clean=len(clean), rows_rejected=len(rej),
                  by_reason=rej["reason"].value_counts().to_dict())
    return clean, rej, report


def build_warehouse(db_path=C.DB_PATH, raw_dir=C.RAW_DIR):
    players = pd.read_csv(raw_dir / "players.csv")
    matches = pd.read_csv(raw_dir / "matches.csv")
    stats = pd.read_csv(raw_dir / "player_match_stats.csv")
    clean, rej, report = validate_stats(stats)
    # referential integrity: stats must point to a known match and player
    orphan = ~clean["match_id"].isin(matches["match_id"]) | ~clean["player_id"].isin(players["player_id"])
    if orphan.any():
        rej = pd.concat([rej, clean[orphan].assign(reason="orphan_foreign_key")], ignore_index=True)
        clean = clean[~orphan]
    report.update(rows_clean=len(clean), rows_rejected=len(rej), orphan_rows=int(orphan.sum()))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript((C.SQL_DIR / "schema.sql").read_text())
    for name, df in [("players", players), ("matches", matches)]:
        df.to_sql(name, con, if_exists="append", index=False)
    cols = ["match_id", "player_id", "placement", "kills", "damage", "shots_fired", "shots_hit",
            "headshot_hits", "avg_reaction_ms", "wall_kills", "max_speed_ratio", "avg_ping_ms",
            "packet_loss_pct", "queue_time_s", "survival_s", "disconnected", "reports_received"]
    clean[cols].to_sql("player_match_stats", con, if_exists="append", index=False, chunksize=50000)
    rows = [("player_match_stats", r["reason"], json.dumps({k: (None if pd.isna(v) else v) for k, v in r.drop("reason").items()}, default=str))
            for _, r in rej.iterrows()]
    con.executemany("INSERT INTO rejected_rows VALUES (?,?,?)", rows)
    con.commit(); con.close()
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.OUT_DIR / "dq_report.json").write_text(json.dumps(report, indent=2, default=int))
    return report


if __name__ == "__main__":
    print(build_warehouse())
