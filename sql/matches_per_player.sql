-- "Kaun kitne match khel raha hai": activity per player + distribution buckets.
WITH per_player AS (
  SELECT player_id, COUNT(*) AS matches_played,
         COUNT(DISTINCT DATE(mt.started_at)) AS active_days,
         MIN(mt.started_at) AS first_match, MAX(mt.started_at) AS last_match
  FROM player_match_stats s JOIN matches mt USING (match_id)
  GROUP BY player_id
)
SELECT player_id, matches_played, active_days,
       ROUND(1.0 * matches_played / active_days, 2) AS matches_per_active_day,
       NTILE(10) OVER (ORDER BY matches_played) AS activity_decile,
       RANK() OVER (ORDER BY matches_played DESC) AS activity_rank
FROM per_player;
