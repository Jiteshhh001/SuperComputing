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
OULAD_URL = "https://analyse.kmi.open.ac.uk/open_dataset/download"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"

# OULAD table names (the dataset contains these CSV files)
OULAD_TABLES = [
    "assessments", "courses", "studentAssessment",
    "studentInfo", "studentRegistration", "studentVle", "vle"
]

# ── Module/Subject Mapping ───────────────────────────────────────────────────
MODULE_NAMES = {
    "AAA": "Social Sciences",
    "BBB": "Technology & IT",
    "CCC": "Mathematics (STEM)",
    "DDD": "Health Sciences",
    "EEE": "Business & Economics",
    "FFF": "Humanities",
    "GGG": "Arts & Design"
}

def format_module_name(code: str) -> str:
    """Convert raw code (e.g., 'AAA', 'AAA_TMA') to readable format."""
    if not isinstance(code, str):
        return str(code)
    
    base = code.split('_')[0]
    subject = MODULE_NAMES.get(base, base)
    
    if "_" in code:
        suffix = code.split('_')[1]
        suffix_map = {"TMA": "Assignments", "CMA": "Quizzes", "Exam": "Exams"}
        return f"{subject} ({suffix_map.get(suffix, suffix)})"
    
    return subject

# ── OpenRouter LLM Configuration ────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-4-scout:free")
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048

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

# ── Hybrid Recommender ──────────────────────────────────────────────────────
RECOMMENDER_CONFIG = {
    # Collaborative Filtering (SVD)
    "svd_n_factors": 50,
    "svd_n_epochs": 30,
    "svd_lr_all": 0.005,
    "svd_reg_all": 0.02,
    # Content-Based Filtering
    "cbf_embedding_model": "all-MiniLM-L6-v2",
    "cbf_top_k": 10,
    # LLM Ranking
    "llm_rerank_top_n": 5,
    # Hybrid Weights
    "cf_weight": 0.4,
    "cbf_weight": 0.35,
    "llm_weight": 0.25,
    # Output
    "top_k_recommendations": 5,
}

# ── Thresholds ──────────────────────────────────────────────────────────────
MASTERY_THRESHOLD = 0.4        # Below this = weak concept
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
