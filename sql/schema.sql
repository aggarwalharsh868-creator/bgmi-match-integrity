DROP TABLE IF EXISTS player_match_stats; DROP TABLE IF EXISTS matches; DROP TABLE IF EXISTS players;
DROP TABLE IF EXISTS rejected_rows;
CREATE TABLE players (
  player_id INTEGER PRIMARY KEY, home_region TEXT NOT NULL, device TEXT,
  signup_date TEXT, mmr REAL NOT NULL);
CREATE TABLE matches (
  match_id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, region TEXT NOT NULL,
  bot_count INTEGER NOT NULL DEFAULT 0, duration_s INTEGER);
CREATE TABLE player_match_stats (
  match_id INTEGER NOT NULL REFERENCES matches(match_id),
  player_id INTEGER NOT NULL REFERENCES players(player_id),
  placement INTEGER, kills INTEGER NOT NULL, damage INTEGER,
  shots_fired INTEGER NOT NULL, shots_hit INTEGER NOT NULL, headshot_hits INTEGER NOT NULL,
  avg_reaction_ms REAL, wall_kills INTEGER NOT NULL, max_speed_ratio REAL NOT NULL,
  avg_ping_ms REAL NOT NULL, packet_loss_pct REAL NOT NULL, queue_time_s REAL,
  survival_s INTEGER, disconnected INTEGER NOT NULL, reports_received INTEGER NOT NULL,
  PRIMARY KEY (match_id, player_id)) WITHOUT ROWID;
CREATE INDEX idx_pms_player ON player_match_stats(player_id);
CREATE TABLE rejected_rows (source TEXT, reason TEXT, row_json TEXT);
