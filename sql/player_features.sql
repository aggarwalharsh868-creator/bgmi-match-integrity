-- One row per player: raw evidence counts. Statistical tests are done in features.py.
WITH m AS (
  SELECT s.*, mt.started_at,
    CASE WHEN s.avg_ping_ms < :ping_max AND s.packet_loss_pct < :loss_max THEN 1 ELSE 0 END AS clean_net,
    CASE WHEN s.shots_hit >= 10
         THEN (s.headshot_hits - s.shots_hit * :p_hs) / SQRT(s.shots_hit * :p_hs * (1 - :p_hs)) END AS hs_z_m
  FROM player_match_stats s JOIN matches mt USING (match_id)
), f AS (
  SELECT *,
    CASE WHEN hs_z_m > 3 THEN 1 ELSE 0 END AS f_aim,
    CASE WHEN avg_reaction_ms IS NOT NULL AND shots_hit >= 8
          AND (avg_reaction_ms - 0.5 * avg_ping_ms) < :react_match_ceil THEN 1 ELSE 0 END AS f_react,
    CASE WHEN kills >= 3 AND wall_kills >= 2 AND 1.0 * wall_kills / kills >= 0.25 THEN 1 ELSE 0 END AS f_wall,
    CASE WHEN clean_net = 1 AND max_speed_ratio > :speed_limit THEN 1 ELSE 0 END AS f_speed
  FROM m
)
SELECT player_id,
  COUNT(*) AS n_matches,
  COUNT(DISTINCT DATE(started_at)) AS active_days,
  SUM(kills) AS kills, SUM(shots_fired) AS shots_fired, SUM(shots_hit) AS shots_hit,
  SUM(headshot_hits) AS hs_hits, SUM(wall_kills) AS wall_kills,
  AVG(avg_reaction_ms - 0.5 * avg_ping_ms) AS react_adj_mean,
  COUNT(avg_reaction_ms) AS react_n,
  AVG(avg_ping_ms) AS ping_mean, AVG(packet_loss_pct) AS loss_mean,
  SUM(reports_received) AS reports, SUM(disconnected) AS disconnects,
  AVG(queue_time_s) AS queue_mean,
  SUM(clean_net) AS clean_matches, SUM(f_speed) AS clean_speed_viol,
  SUM(f_aim) AS f_aim_m, SUM(f_react) AS f_react_m, SUM(f_wall) AS f_wall_m,
  SUM(CASE WHEN f_aim + f_react + f_wall + f_speed > 0 THEN 1 ELSE 0 END) AS flagged_matches
FROM f GROUP BY player_id;
