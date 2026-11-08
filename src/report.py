"""Charts for the README. Palette: validated categorical order (blue, orange, aqua)."""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import config as C
from .db import connect, query

BLUE, ORANGE, AQUA, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#c9c8c3",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 10, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "axes.titlelocation": "left"})


def _save(fig, name):
    C.CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(C.CHART_DIR / name, dpi=160); plt.close(fig)


def activity_chart():
    con = connect(); a = query(con, "matches_per_player.sql"); con.close()
    a.to_csv(C.OUT_DIR / "matches_per_player.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(a["matches_played"], bins=40, color=BLUE, edgecolor=SURF, linewidth=1)
    ax[0].set_title("Matches played per player (30 days)"); ax[0].set_xlabel("matches"); ax[0].set_ylabel("players")
    dec = a.groupby("activity_decile")["matches_played"].sum(); share = 100 * dec / dec.sum()
    ax[1].bar(share.index, share.values, color=BLUE, edgecolor=SURF, linewidth=2)
    ax[1].set_title("Share of all matches by activity decile"); ax[1].set_xlabel("decile (10 = most active)"); ax[1].set_ylabel("% of matches")
    ax[1].set_xticks(range(1, 11))
    _save(fig, "01_activity.png")


def detection_charts(d: pd.DataFrame, res: dict):
    h = d[(d["split"] == "holdout") & d["eligible"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = np.linspace(0, 1, 41)
    ax.hist(h.loc[~h["is_cheater"], "prob"], bins=bins, color=BLUE, label="genuine", edgecolor=SURF, linewidth=1)
    ax.hist(h.loc[h["is_cheater"], "prob"], bins=bins, color=ORANGE, label="cheater", edgecolor=SURF, linewidth=1, alpha=0.95)
    ax.set_yscale("log"); ax.set_xlabel("calibrated cheat probability (holdout players)"); ax.set_ylabel("players (log)")
    for k, lab in [("tau_review", "review"), ("tau_auto", "auto")]:
        ax.axvline(res["thresholds"][k], color=MUTED, linestyle="--", linewidth=1)
        ax.text(res["thresholds"][k] + 0.005, ax.get_ylim()[1] * 0.5, lab, color=MUTED, fontsize=9)
    ax.set_title("Scores separate cheaters from genuine players"); ax.legend(frameon=False)
    _save(fig, "02_scores.png")

    b = res["baselines_holdout"]; p = res["policy"]
    names = ["Naive K/D rule", "IsolationForest\n(top 5%)", "Our policy:\nhuman review", "Our policy:\nauto action"]
    fps = [b["naive_kd"]["false_positives"], b["isolation_forest"]["false_positives"],
           round(p["review_queue"] * (1 - (p["review_precision"] or 0))), p["auto_false_positives"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(names, fps, color=[ORANGE, ORANGE, BLUE, BLUE], edgecolor=SURF, linewidth=2)
    for r, v in zip(bars, fps):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.8, str(v), ha="center", color=INK)
    ax.set_ylabel("genuine players flagged (holdout)"); ax.set_title("Genuine players wrongly flagged")
    ax.grid(axis="x", visible=False)
    _save(fig, "03_false_positives.png")

    sl = pd.DataFrame(res["genuine_slices"]).set_index("group")
    fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(sl)); w = 0.27
    ax.bar(x - w, sl["naive_kd_banned"], w, color=ORANGE, label="naive K/D rule bans", edgecolor=SURF, linewidth=2)
    ax.bar(x, sl["human_review"], w, color=BLUE, label="our policy: sent to review", edgecolor=SURF, linewidth=2)
    ax.bar(x + w, sl["auto_actioned"], w, color=AQUA, label="our policy: auto-actioned", edgecolor=SURF, linewidth=2)
    ax.set_xticks(x); ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(sl.index, sl["genuine_players"])])
    ax.set_ylabel("genuine players"); ax.set_title("Hard genuine cases: smurfs and elites are not cheaters")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="x", visible=False)
    _save(fig, "04_genuine_slices.png")


def health_charts(daily, q_tier, sim):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cols = {"IN-North": BLUE, "IN-West": ORANGE, "IN-South": AQUA}
    for reg, s in daily.groupby("region"):
        s = s.sort_values("day"); ax.plot(pd.to_datetime(s["day"]), 100 * s["dc_rate"], color=cols[reg], label=reg, linewidth=2)
    inc = daily.loc[daily["dc_rate"].idxmax()]
    ax.annotate("alert: IN-South network incident", (pd.to_datetime(inc["day"]), 100 * inc["dc_rate"]),
                xytext=(-170, -6), textcoords="offset points", color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_ylabel("disconnect rate (%)"); ax.set_title("Daily disconnect rate by region, with spike alert"); ax.legend(frameon=False, ncol=3)
    _save(fig, "05_disconnect_alert.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.6)); x = np.arange(len(q_tier)); w = 0.38
    ax.bar(x - w / 2, q_tier["peak"], w, color=BLUE, label="peak", edgecolor=SURF, linewidth=2)
    ax.bar(x + w / 2, q_tier["off-peak (2-9h)"], w, color=ORANGE, label="off-peak (2-9h)", edgecolor=SURF, linewidth=2)
    ax.set_xticks(x); ax.set_xticklabels(q_tier.index.astype(str)); ax.set_xlabel("rating (MMR) band"); ax.set_ylabel("avg queue time (s)")
    ax.set_title("High-rated players wait longest, especially off-peak"); ax.legend(frameon=False); ax.grid(axis="x", visible=False)
    _save(fig, "06_queue_by_tier.png")

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (col, ttl) in zip(axs, [("avg_queue_s", "avg queue (s)"), ("avg_bots_per_lobby", "bots per lobby"), ("unfair_slot_pct", "unfair slots (%)")]):
        piv = sim.pivot(index="scenario", columns="strategy", values=col)[["fixed", "progressive"]]
        xx = np.arange(len(piv)); ax.bar(xx - 0.2, piv["fixed"], 0.38, color=ORANGE, edgecolor=SURF, linewidth=2, label="fixed window")
        ax.bar(xx + 0.2, piv["progressive"], 0.38, color=BLUE, edgecolor=SURF, linewidth=2, label="progressive window")
        ax.set_xticks(xx); ax.set_xticklabels(piv.index); ax.set_title(ttl); ax.grid(axis="x", visible=False)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Matchmaker simulation: faster queues vs lobby fairness", x=0.01, ha="left", fontweight="bold")
    _save(fig, "07_matchmaker_sim.png")
