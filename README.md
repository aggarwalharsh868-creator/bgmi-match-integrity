# BGMI Match Integrity & Matchmaking Health Pipeline

An end-to-end **Data Engineering + Data Science** project inspired by BGMI, designed to analyze player activity, detect suspicious gameplay patterns, monitor matchmaking health, and simulate fairer matchmaking strategies.

The project combines Python, SQL, statistical analysis, machine learning, data quality validation, and matchmaking simulation into one end-to-end pipeline.

> **Disclaimer:** This project uses synthetic/simulated telemetry data. It is not affiliated with BGMI, KRAFTON, or any official anti-cheat system.

## Project Highlights

- Processes approximately 497K simulated player-match records.
- Analyzes activity patterns across 4,000 players and 5,000 simulated matches.
- Implements safety-first cheat detection using statistical evidence and calibrated machine learning.
- Detects regional network incidents and missing telemetry.
- Simulates matchmaking strategies using rating-based queue windows.
- Generates analytical reports, CSV files, JSON metrics, and visualizations.
- Includes 7 automated tests.

## Technology Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Database | SQLite |
| Query Language | SQL |
| Visualization | Matplotlib |
| Testing | Pytest |
| Version Control | Git & GitHub |

## Problem Statement

Online multiplayer games face several operational challenges:

1. Unusual player activity and suspicious gameplay.
2. False-positive cheat detection that can punish genuine skilled players.
3. Unbalanced matchmaking and long queue times.
4. Network incidents and server disconnections.
5. Missing, duplicated, or invalid telemetry records.

This project builds a simulated analytical system to investigate these challenges and evaluate possible solutions.

## System Architecture

```text
                  run_pipeline.py
                        |
                        v
                  simulate.py
             Synthetic Game Telemetry
                        |
                        v
                   ingest.py
             Validation & Quarantine
                        |
                        v
                SQLite Warehouse
                        |
              +---------+---------+
              |                   |
              v                   v
          features.py        match_health.py
          SQL + Stats        Health Alerts
              |                   |
              v                   v
           detect.py       matchmaker_sim.py
        ML + Policy         Queue Simulation
              |                   |
              +---------+---------+
                        |
                        v
                    report.py
                        |
                        v
                     outputs/
                 CSV / JSON / Charts

## 1. Player Activity Analytics

SQL-based analysis using CTEs, NTILE, and RANK window functions.

Key findings:

- Median player activity: 88 matches in 30 days.
- Mean player activity: 124 matches.
- Maximum player activity: 1,143 matches.
- Most active 10% of players generate 33% of all matches.
- Least active 30% generate approximately 7% of matches.

![Player Activity](outputs/charts/01_activity.png)

## 2. Safety-First Cheat Detection

The detection system combines statistical evidence with a calibrated gradient-boosting classifier.

Features include:

- Headshot share
- Shooting accuracy
- Ping-adjusted reaction time
- Wall penetration indicators
- Movement speed anomalies
- Repeated suspicious matches

### Decision Policy

| Tier | Decision |
|---|---|
| INSUFFICIENT_DATA | Collect more match data |
| CLEAR / WATCHLIST | Monitor player activity |
| HUMAN_REVIEW | Send suspicious cases for human investigation |
| AUTO_ACTION | Act only when strict evidence requirements are satisfied |

Automatic action requires:
- Cheat probability of at least 0.90.
- At least 2 independent evidence channels.
- At least 4 suspicious matches.

### Model Evaluation

Evaluation uses a held-out set of 1,165 eligible players, including 62 simulated cheaters.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.998 |
| PR-AUC | 0.9722 |
| Automatic actions | 15 |
| Genuine players wrongly auto-actioned | 0 |
| Automatic action precision | 100% |
| Cheaters caught by auto-action | 24.19% |
| Cheater recall with human review | 98.39% |

### Baseline Comparison

| Detection Method | Precision | Recall |
|---|---:|---:|
| Naive K/D Rule | 19.3% | 17.74% |
| Isolation Forest | 71.15% | 59.68% |
| Proposed Safety-First Policy | 100% auto-action precision | 98.39% combined recall with review |

**Important:** The 100% precision applies only to the automatic-action tier. Human review also catches suspicious cases but includes genuine players who must not be automatically punished.

![Detection Performance](outputs/charts/03_false_positives.png)

![Genuine Player Analysis](outputs/charts/04_genuine_slices.png)

## 3. Matchmaking Health Monitoring

The system monitors:

- Player queue times
- Lobby rating distribution
- Bot participation
- Regional disconnect rates
- Match volume anomalies
- Telemetry data quality

### Detected Incidents

| Incident | Detection |
|---|---|
| IN-South network incident | 34.43% disconnect rate on 17 October 2026 |
| IN-North telemetry loss | Match volume ratio dropped to 0.32 on 9 October 2026 |

The monitoring system uses rolling median and Median Absolute Deviation (MAD) based anomaly detection.

![Disconnect Alerts](outputs/charts/05_disconnect_alert.png)

![Queue Analysis](outputs/charts/06_queue_by_tier.png)

## 4. Matchmaker Simulation

Two matchmaking strategies are compared:

- Fixed MMR window: ±150 MMR.
- Progressive MMR window: starts at ±100 MMR and expands by 6 MMR per second, up to ±500 MMR.

### Simulation Results

| Scenario | Strategy | Average Queue |
|---|---|---:|
| Peak | Fixed | 41.7 sec |
| Peak | Progressive | 20.7 sec |
| Off-peak | Fixed | 139.9 sec |
| Off-peak | Progressive | 49.1 sec |

Progressive matchmaking substantially reduces queue times, but wider rating windows can increase lobby imbalance.

![Matchmaker Simulation](outputs/charts/07_matchmaker_sim.png)

## 5. Data Quality Engineering

The ingestion pipeline validates records and quarantines invalid data instead of silently dropping it.

| Metric | Records |
|---|---:|
| Input records | 497,488 |
| Clean records | 494,266 |
| Rejected records | 3,222 |

Validation includes:
- Duplicate match-player records
- Missing required fields
- Hits exceeding shots
- Negative values


### 2. Project Structure replace karo

Purana `## Project Structure` section delete karke ye paste karo:

```text
## Project Structure

bgmi-match-integrity/
│
├── outputs/
│   ├── charts/
│   │   ├── 01_activity.png
│   │   ├── 02_scores.png
│   │   ├── 03_false_positives.png
│   │   ├── 04_genuine_slices.png
│   │   ├── 05_disconnect_alert.png
│   │   ├── 06_queue_by_tier.png
│   │   └── 07_matchmaker_sim.png
│   ├── daily_health.csv
│   ├── detection_metrics.json
│   ├── dq_report.json
│   ├── match_health.json
│   ├── matches_per_player.csv
│   ├── matchmaker_comparison.csv
│   └── player_scores.csv
│
├── sql/
│   ├── schema.sql
│   ├── matches_per_player.sql
│   ├── player_features.sql
│   └── lobby_health.sql
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   ├── detect.py
│   ├── features.py
│   ├── ingest.py
│   ├── match_health.py
│   ├── matchmaker_sim.py
│   ├── report.py
│   ├── run_detection.py
│   └── simulate.py
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
│
├── .gitignore
├── requirements.txt
├── run_pipeline.py
└── README.md

Note: data/ is generated at runtime and excluded through .gitignore.

## Installation & Execution

### 1. Clone Repository

```bash
git clone https://github.com/aggarwalharsh868-creator/bgmi-match-integrity.git
cd bgmi-match-integrity
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Complete Pipeline

```bash
python run_pipeline.py
```

### 5. Run Automated Tests

```bash
python -m pytest -q
```

Expected result: 7 tests passed.

Generated reports and charts are saved in the `outputs/` directory.

## Honest Limitations

- All gameplay telemetry is simulated, not real BGMI data.
- Model evaluation results may be optimistic because synthetic cheat patterns are more structured than real-world behavior.
- Real anti-cheat systems also require client/server integrity checks and human investigation.
- Matchmaking simulation does not model parties, latency constraints, or all production matchmaking rules.
- Human-review decisions require additional operational processes.

## Future Improvements

- Integrate Kafka for real-time event streaming.
- Use PostgreSQL for scalable data storage.
- Develop an interactive Power BI dashboard.
- Build a real-time monitoring interface.
- Explore more robust anomaly detection models.
- Introduce production-style monitoring and deployment.

## Resume Highlights

**BGMI-Inspired Match Integrity & Matchmaking Health Pipeline** | Python, SQL, SQLite, Pandas, Scikit-learn

- Developed an end-to-end data engineering and data science pipeline processing approximately 497K simulated player-match records.
- Built a safety-first cheat detection framework combining statistical evidence and calibrated machine learning, achieving 100% precision in the automatic-action tier on held-out simulated data.
- Implemented SQL-based player analytics, data-quality validation, regional anomaly monitoring, and matchmaking simulation.
- Evaluated fixed and progressive MMR matchmaking strategies, reducing simulated peak average queue time by approximately 50%.

---

**Author:** Harsh Aggarwal

**GitHub:** https://github.com/aggarwalharsh868-creator