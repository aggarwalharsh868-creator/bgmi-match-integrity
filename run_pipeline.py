"""End-to-end run:  python run_pipeline.py   (about 30-60 seconds)."""
import argparse, json, time
from src import config as C, simulate, ingest, run_detection, match_health, matchmaker_sim, report


def main(players, matches):
    t = time.time()
    print("1/6 simulate telemetry ...", simulate.generate(players, matches))
    print("2/6 ingest + data quality ...", ingest.build_warehouse())
    print("3/6 detect cheaters ...")
    scores, res = run_detection.run()
    print(json.dumps(res["policy"], indent=1))
    print("4/6 match health ...")
    health, daily, lobby, rows, q_tier = match_health.run(scores)
    print("5/6 matchmaker simulation ...")
    sim = matchmaker_sim.compare()
    print("6/6 charts ...")
    report.activity_chart(); report.detection_charts(scores, res); report.health_charts(daily, q_tier, sim)
    print(f"done in {time.time() - t:.0f}s -> outputs/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=C.N_PLAYERS)
    ap.add_argument("--matches", type=int, default=C.N_MATCHES)
    a = ap.parse_args(); main(a.players, a.matches)
