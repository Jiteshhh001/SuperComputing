"""
═══════════════════════════════════════════════════════════════════════════════
  Personalized Learning Agent — Central Configuration
  All hyperparameters, paths, URLs, and thresholds in one place.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESOURCES_DIR = DATA_DIR / "resources"

# Create directories
for _dir in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESOURCES_DIR,
             RAW_DIR / "oulad", RAW_DIR / "uci"]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Dataset URLs ─────────────────────────────────────────────────────────────
# UCI Machine Learning Repository mirror first (the Open University page now redirects to HTML)
OULAD_URLS = [
    "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip",
    "https://analyse.kmi.open.ac.uk/open_dataset/download",
]
OULAD_URL = OULAD_URLS[0]
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"

# OULAD table names (the dataset contains these CSV files)
OULAD_TABLES = [
    "assessments", "courses", "studentAssessment",
    "studentInfo", "studentRegistration", "studentVle", "vle"
]

# ── Module / Domain Mapping ─────────────────────────────────────────────────
# OULAD anonymises module names; the dataset documentation only gives the domain.
MODULE_DOMAINS = {
    "AAA": "Social Sciences", "BBB": "Social Sciences", "GGG": "Social Sciences",
    "CCC": "STEM", "DDD": "STEM", "EEE": "STEM", "FFF": "STEM",
}
ASSESSMENT_TYPE_NAMES = {"TMA": "Assignments", "CMA": "Quizzes", "Exam": "Exam"}


def format_module_name(code: str) -> str:
    """Readable label, e.g. 'CCC_TMA' -> 'CCC · STEM (Assignments)'."""
    if not isinstance(code, str):
        return str(code)
    base, _, suffix = code.partition("_")
    domain = MODULE_DOMAINS.get(base)
    label = f"{base} · {domain}" if domain else base
    if suffix:
        if suffix.startswith("vle_"):
            return f"{label} (VLE: {suffix[4:]})"
        return f"{label} ({ASSESSMENT_TYPE_NAMES.get(suffix, suffix)})"
    return label


# ── OpenRouter LLM Configuration ────────────────────────────────────────────
def get_openrouter_api_key() -> str:
    """API key from the environment / .env, or from Streamlit secrets when deployed."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st
        return str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
    except Exception:
        return ""


OPENROUTER_API_KEY = get_openrouter_api_key()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Free-tier models change often; the client tries these in order and remembers what works.
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
LLM_FALLBACK_MODELS = [
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free",
]
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1200
LLM_TIMEOUT_SECONDS = 30

# ── DKT Model (LSTM) Hyperparameters ────────────────────────────────────────
DKT_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 50,
    "early_stopping_patience": 7,
    "max_seq_length": 200,
    "embedding_dim": 64,
    "weight_decay": 1e-5,
    "grad_clip": 5.0,
}

# ── XGBoost Gap Detector ────────────────────────────────────────────────────
XGBOOST_CONFIG = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,  # Adjusted during training for imbalance
    "eval_metric": "logloss",
    "early_stopping_rounds": 20,
    "random_state": 42,
}

# ── Thresholds ──────────────────────────────────────────────────────────────
MASTERY_THRESHOLD = 0.4        # Below this = weak concept
GAP_SCORE_THRESHOLD = 0.5      # Gap detector target: next assessment score below 50%
AT_RISK_THRESHOLD = 0.5        # Below this = at-risk student
ENGAGEMENT_LOW_THRESHOLD = 0.3 # Below this = low engagement
IMPROVEMENT_SIGNIFICANT = 0.1  # 10% improvement = significant

# ── Mastery Levels (0–4 scale) ──────────────────────────────────────────────
MASTERY_LEVELS = {
    0: {"label": "No Mastery",    "range": (0.0, 0.2), "color": "#EF4444"},
    1: {"label": "Beginner",      "range": (0.2, 0.4), "color": "#F97316"},
    2: {"label": "Developing",    "range": (0.4, 0.6), "color": "#EAB308"},
    3: {"label": "Proficient",    "range": (0.6, 0.8), "color": "#22C55E"},
    4: {"label": "Mastered",      "range": (0.8, 1.0), "color": "#06B6D4"},
}

# ── Data Split Ratios ───────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── FastAPI ─────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Personalized Learning Agent API"
API_VERSION = "1.0.0"

# ── Streamlit ───────────────────────────────────────────────────────────────
STREAMLIT_PAGE_TITLE = "LearnFlow AI"
STREAMLIT_PAGE_ICON = "🧠"
STREAMLIT_LAYOUT = "wide"

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
