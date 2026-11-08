-- One row per match: lobby quality and health signals.
SELECT mt.match_id, mt.started_at, mt.region, mt.bot_count,
       COUNT(*) AS humans,
       MAX(p.mmr) - MIN(p.mmr) AS mmr_range,
       AVG(p.mmr) AS mmr_mean,
       SQRT(MAX(AVG(p.mmr * p.mmr) - AVG(p.mmr) * AVG(p.mmr), 0)) AS mmr_sd,
       AVG(s.avg_ping_ms) AS ping_mean,
       AVG(s.queue_time_s) AS queue_mean,
       1.0 * SUM(s.disconnected) / COUNT(*) AS dc_rate,
       MAX(s.kills) AS top_kills
FROM matches mt
JOIN player_match_stats s USING (match_id)
JOIN players p USING (player_id)
GROUP BY mt.match_id;
