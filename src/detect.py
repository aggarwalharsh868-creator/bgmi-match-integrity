"""Cheat detection with a safety-first decision policy.

Pipeline:  evidence features -> calibrated probability -> tiered action policy.
The policy is the point: probability alone never triggers a ban. Auto-action also needs
  (a) enough matches, (b) persistence across several matches, (c) >=2 independent evidence
channels (e.g. headshot share AND reaction time), and (d) a probability above a threshold
chosen on held-out data so the genuine-player false-positive rate stays under a hard cap.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from . import config as C
from .features import MODEL_FEATURES


def _model(seed):
    base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.07, max_iter=140,
                                          l2_regularization=1.0, random_state=seed)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def score_players(feat: pd.DataFrame, truth: pd.DataFrame, seed=C.SEED):
    """Out-of-sample probabilities for every player.
    dev (70%): 5-fold out-of-fold scores.  holdout (30%): model trained on all of dev.
    Labels play the role of 'confirmed past ban decisions' used for training."""
    d = feat.merge(truth[["player_id", "archetype", "is_cheater", "cheat_type"]], on="player_id")
    strat = d["is_cheater"].astype(str) + d["archetype"]
    dev_idx, hold_idx = train_test_split(d.index, test_size=0.30, random_state=seed, stratify=strat)
    X, y = d[MODEL_FEATURES].fillna(0), d["is_cheater"].astype(int)
    d["prob"] = np.nan
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    dev = d.loc[dev_idx]
    for tr, te in skf.split(dev, y.loc[dev_idx]):
        m = _model(seed).fit(X.loc[dev.index[tr]], y.loc[dev.index[tr]])
        d.loc[dev.index[te], "prob"] = m.predict_proba(X.loc[dev.index[te]])[:, 1]
    final = _model(seed).fit(X.loc[dev_idx], y.loc[dev_idx])
    d.loc[hold_idx, "prob"] = final.predict_proba(X.loc[hold_idx])[:, 1]
    d["split"] = "dev"; d.loc[hold_idx, "split"] = "holdout"
    return d


def choose_thresholds(d: pd.DataFrame):
    """Pick thresholds on dev (out-of-fold) data only."""
    dev = d[(d["split"] == "dev") & d["eligible"]]
    genuine = dev[~dev["is_cheater"]]
    gate = (dev["n_channels"] >= C.MIN_CHANNELS) & (dev["flagged_matches"] >= C.MIN_FLAGGED_MATCHES)
    # auto tier: lowest prob threshold (>=0.90) where <= FPR cap of genuine pass the full gate
    allowed = int(np.floor(C.FPR_CAP_AUTO * len(genuine)))
    tau_auto = 0.90
    for t in np.arange(0.90, 1.0, 0.005):
        fp = ((dev["prob"] >= t) & gate & ~dev["is_cheater"]).sum()
        if fp <= allowed:
            tau_auto = float(t); break
    else:
        tau_auto = 0.999
    # review tier: lowest threshold with >= REVIEW_PRECISION precision on dev
    tau_review = 0.5
    for t in np.arange(0.02, 0.95, 0.01):
        sel = dev["prob"] >= t
        if sel.sum() >= 10 and dev.loc[sel, "is_cheater"].mean() >= C.REVIEW_PRECISION:
            tau_review = float(t); break
    return dict(tau_auto=round(tau_auto, 3), tau_review=round(tau_review, 3), fp_allowed=allowed)


def apply_policy(d: pd.DataFrame, th: dict) -> pd.DataFrame:
    d = d.copy()
    gate = (d["n_channels"] >= C.MIN_CHANNELS) & (d["flagged_matches"] >= C.MIN_FLAGGED_MATCHES)
    tier = np.full(len(d), "CLEAR", dtype=object)
    tier[(d["prob"] >= C.WATCH_PROB) | (d["n_channels"] >= 1)] = "WATCHLIST"
    tier[d["prob"] >= th["tau_review"]] = "HUMAN_REVIEW"
    tier[(d["prob"] >= th["tau_auto"]) & gate] = "AUTO_ACTION"
    tier[~d["eligible"]] = "INSUFFICIENT_DATA"           # hard rule: never act on tiny samples
    d["tier"] = tier
    return d


def baselines(d: pd.DataFrame, seed=C.SEED) -> pd.DataFrame:
    """Two naive detectors for comparison: K/D-style rule and an unsupervised IsolationForest."""
    d = d.copy()
    d["naive_kd_flag"] = d["eligible"] & (d["perf_z"] > 3.0)
    Xs = d[MODEL_FEATURES].fillna(0)
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=seed).fit(Xs[d["eligible"]])
    d["iforest_flag"] = False
    d.loc[d["eligible"], "iforest_flag"] = iso.predict(Xs[d["eligible"]]) == -1
    return d


def evaluate(d: pd.DataFrame, th: dict) -> dict:
    h = d[(d["split"] == "holdout") & d["eligible"]]
    out = dict(thresholds=th,
               holdout=dict(n_players=int(len(h)), n_cheaters=int(h["is_cheater"].sum()),
                            roc_auc=round(roc_auc_score(h["is_cheater"], h["prob"]), 4),
                            pr_auc=round(average_precision_score(h["is_cheater"], h["prob"]), 4)))
    g = h[~h["is_cheater"]]; c = h[h["is_cheater"]]
    auto, rev = h["tier"] == "AUTO_ACTION", h["tier"] == "HUMAN_REVIEW"
    out["policy"] = dict(
        auto_flagged=int(auto.sum()), auto_false_positives=int((auto & ~h["is_cheater"]).sum()),
        auto_precision=round(float(h.loc[auto, "is_cheater"].mean()), 4) if auto.any() else None,
        auto_recall=round(float((auto & h["is_cheater"]).sum() / max(len(c), 1)), 4),
        review_queue=int(rev.sum()), review_precision=round(float(h.loc[rev, "is_cheater"].mean()), 4) if rev.any() else None,
        recall_auto_plus_review=round(float(((auto | rev) & h["is_cheater"]).sum() / max(len(c), 1)), 4),
        genuine_wrongly_auto_actioned_pct=round(100 * float((auto & ~h["is_cheater"]).sum() / max(len(g), 1)), 3),
        genuine_sent_to_review_pct=round(100 * float((rev & ~h["is_cheater"]).sum() / max(len(g), 1)), 3))
    # per-archetype fairness table (genuine slices that a naive system would punish)
    rows = []
    for a in ["normal", "elite", "smurf", "high_ping", "new"]:
        s = h[(h["archetype"] == a) & ~h["is_cheater"]]
        full = d[(d["split"] == "holdout") & (d["archetype"] == a) & ~d["is_cheater"]]
        rows.append(dict(group=a, genuine_players=int(len(full)),
                         auto_actioned=int((s["tier"] == "AUTO_ACTION").sum()),
                         human_review=int((s["tier"] == "HUMAN_REVIEW").sum()),
                         naive_kd_banned=int(s["naive_kd_flag"].sum()),
                         iforest_flagged=int(s["iforest_flag"].sum()),
                         insufficient_data=int((full["tier"] == "INSUFFICIENT_DATA").sum())))
    out["genuine_slices"] = rows
    out["baselines_holdout"] = dict(
        naive_kd=dict(flagged=int(h["naive_kd_flag"].sum()),
                      false_positives=int((h["naive_kd_flag"] & ~h["is_cheater"]).sum()),
                      precision=round(float(h.loc[h["naive_kd_flag"], "is_cheater"].mean()), 4) if h["naive_kd_flag"].any() else None,
                      recall=round(float((h["naive_kd_flag"] & h["is_cheater"]).sum() / max(len(c), 1)), 4)),
        isolation_forest=dict(flagged=int(h["iforest_flag"].sum()),
                              false_positives=int((h["iforest_flag"] & ~h["is_cheater"]).sum()),
                              precision=round(float(h.loc[h["iforest_flag"], "is_cheater"].mean()), 4) if h["iforest_flag"].any() else None,
                              recall=round(float((h["iforest_flag"] & h["is_cheater"]).sum() / max(len(c), 1)), 4)))
    ct = c.assign(caught=c["tier"].isin(["AUTO_ACTION", "HUMAN_REVIEW"])).groupby("cheat_type")["caught"].agg(["size", "mean"])
    out["recall_by_cheat_type"] = {k: dict(n=int(v["size"]), caught_pct=round(100 * float(v["mean"]), 1)) for k, v in ct.iterrows()}
    return out
