"""Central configuration. Every threshold used by the pipeline lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRUTH_DIR = DATA_DIR / "ground_truth"      # evaluation only, never used as a feature
DB_PATH = DATA_DIR / "bgmi_integrity.db"
OUT_DIR = ROOT / "outputs"
CHART_DIR = OUT_DIR / "charts"
SQL_DIR = ROOT / "sql"

# ---- simulation ------------------------------------------------------------
SEED = 42
N_PLAYERS = 4000
N_MATCHES = 5000
LOBBY_SIZE = 100
DAYS = 30
START_DATE = "2026-10-01"
REGIONS = ["IN-North", "IN-West", "IN-South"]
REGION_P = [0.35, 0.40, 0.25]
CHEATER_RATE = 0.055
INCIDENT_REGION, INCIDENT_DAY = "IN-South", 16      # 12:00-24:00 network incident
LOGGING_GAP_REGION, LOGGING_GAP_DAY = "IN-North", 8  # 70% of telemetry lost

# ---- "human ceiling" references (what a clean top-1% human can sustain) -----
CEIL_HS = 0.26            # headshot share of hits
CEIL_ACC = 0.32           # shots_hit / shots_fired
CEIL_REACT_MS = 225       # mean ping-adjusted reaction time (lower = faster)
REACT_SD = 25.0
CEIL_WALL = 0.05          # share of kills with no line-of-sight
CLEAN_SPEED_VIOL_RATE = 0.03   # legit violation rate on a clean network
CLEAN_PING_MAX, CLEAN_LOSS_MAX = 80, 1.0   # network "clean" = lag cannot explain it
SPEED_LIMIT = 1.10        # max_speed_ratio above this is suspicious
REACT_MATCH_CEIL = 185

# ---- decision policy ---------------------------------------------------------
MIN_MATCHES = 8           # below this: "insufficient data", never any action
MIN_FLAGGED_MATCHES = 4   # persistence: must look suspicious in several matches
MIN_CHANNELS = 2          # independent evidence channels needed for auto action
Z_STRONG = 4.0
FLAG_NULL_RATE = 0.01     # share of matches a clean player trips a per-match flag in
FPR_CAP_AUTO = 0.001      # <=0.1% of genuine players may hit the auto tier
REVIEW_PRECISION = 0.60
WATCH_PROB = 0.15
