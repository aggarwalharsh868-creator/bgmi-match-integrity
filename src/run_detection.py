import json
import pandas as pd
from . import config as C
from .features import build_features
from .detect import score_players, choose_thresholds, apply_policy, baselines, evaluate


def run():
    feat = build_features()
    truth = pd.read_csv(C.TRUTH_DIR / "players_truth.csv")
    d = score_players(feat, truth)
    th = choose_thresholds(d)
    d = apply_policy(d, th)
    d = baselines(d)
    res = evaluate(d, th)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(C.OUT_DIR / "player_scores.csv", index=False)
    (C.OUT_DIR / "detection_metrics.json").write_text(json.dumps(res, indent=2))
    return d, res


if __name__ == "__main__":
    print(json.dumps(run()[1], indent=2))
