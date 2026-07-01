"""
═══════════════════════════════════════════════════════════════════════════════
  Gap Detector — XGBoost classifier for weak topic identification
  Identifies exactly which concepts a student is struggling with,
  ranked by severity and confidence.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, roc_auc_score
)
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import XGBOOST_CONFIG, MODELS_DIR, MASTERY_THRESHOLD

logger = logging.getLogger(__name__)


class GapDetector:
    """
    XGBoost-based classifier that identifies weak topics for each student.

    Features (all computed from information available BEFORE the predicted assessment):
        - Score history for the concept (avg, min, max, std, trend, last 3 scores)
        - Submission behaviour (late ratio, timing)
        - VLE engagement in the module up to the last submission
        - Module / assessment type and student demographics

    Target:
        - next_is_weak: 1 if the student's NEXT assessment of the concept scores
          below the mastery threshold, 0 otherwise
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or XGBOOST_CONFIG
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_columns: list[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        self.threshold: float = 0.5   # decision threshold tuned on out-of-fold predictions

    def train(
        self,
        gap_features: pd.DataFrame,
        target_col: str = "next_is_weak",
        cv_folds: int = 5,
    ) -> dict:
        """
        Train the gap detector with cross-validation.

        Args:
            gap_features: DataFrame with features + target column
            target_col: Name of the binary target column
            cv_folds: Number of cross-validation folds

        Returns:
            dict with training metrics and feature importances
        """
        logger.info("━" * 60)
        logger.info("Training Gap Detector (XGBoost)")
        logger.info("━" * 60)

        # Separate features and target
        exclude_cols = [target_col, "id_student", "concept", "is_weak"]
        self.feature_columns = [
            c for c in gap_features.columns if c not in exclude_cols
        ]
        X = gap_features[self.feature_columns].copy()
        y = gap_features[target_col].copy()
        groups = gap_features["id_student"].values

        # Class imbalance is handled with a tuned decision threshold (not class weights),
        # so predicted probabilities stay calibrated and can be shown to students as risk.
        pos_ratio = y.sum() / len(y)
        logger.info(f"  Class distribution: weak={y.sum():,} ({pos_ratio:.1%}), strong={(1-y).sum():,}")

        # Configure model
        model_params = {k: v for k, v in self.config.items()
                        if k not in ["early_stopping_rounds", "random_state", "eval_metric"]}
        model_params["scale_pos_weight"] = 1.0
        model_params["random_state"] = self.config.get("random_state", 42)
        model_params["eval_metric"] = self.config.get("eval_metric", "logloss")
        # Note: use_label_encoder was removed in XGBoost 2.0+

        # Cross-validation grouped by student (no student appears in both train and validation)
        cv_results = {"f1": [], "precision": [], "recall": [], "auc": []}
        skf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        folds = list(skf.split(X, y, groups))

        # Pass 1: out-of-fold probabilities → pick the F1-optimal threshold
        for train_idx, val_idx in folds:
            m = xgb.XGBClassifier(**model_params)
            m.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
            oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
        candidates = np.arange(0.05, 0.71, 0.01)
        f1s = [f1_score(y, (oof >= t).astype(int)) for t in candidates]
        self.threshold = float(round(candidates[int(np.argmax(f1s))], 2))
        logger.info(f"  Decision threshold (max out-of-fold F1): {self.threshold:.2f}")

        # Pass 2: per-fold metrics at that threshold
        for fold, (train_idx, val_idx) in enumerate(folds, 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = xgb.XGBClassifier(**model_params)
            fold_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_prob = fold_model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= self.threshold).astype(int)

            cv_results["f1"].append(f1_score(y_val, y_pred))
            cv_results["precision"].append(precision_score(y_val, y_pred))
            cv_results["recall"].append(recall_score(y_val, y_pred))
            try:
                cv_results["auc"].append(roc_auc_score(y_val, y_prob))
            except ValueError:
                cv_results["auc"].append(0.5)

            logger.info(
                f"  Fold {fold}/{cv_folds}: "
                f"F1={cv_results['f1'][-1]:.4f}, "
                f"AUC={cv_results['auc'][-1]:.4f}"
            )

        # Train final model on all data
        self.model = xgb.XGBClassifier(**model_params)
        self.model.fit(X, y, verbose=False)

        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        # Summary
        metrics = {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in cv_results.items()}
        logger.info("━" * 60)
        logger.info("Gap Detector Results (CV):")
        logger.info(f"  F1-score:  {metrics['f1']['mean']:.4f} ± {metrics['f1']['std']:.4f}")
        logger.info(f"  AUC-ROC:   {metrics['auc']['mean']:.4f} ± {metrics['auc']['std']:.4f}")
        logger.info(f"  Precision: {metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}")
        logger.info("━" * 60)

        return {
            "cv_metrics": metrics,
            "threshold": self.threshold,
            "feature_importance": self.feature_importance,
        }

    def predict_gaps(
        self,
        student_features: pd.DataFrame,
        concept_map: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Predict weak topics for a student (or set of students).

        Returns DataFrame with columns:
            - concept, is_weak, confidence, severity, priority_rank
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = student_features[self.feature_columns].copy()

        probabilities = self.model.predict_proba(X)[:, 1]
        t = self.threshold

        results = student_features[["id_student", "concept"]].copy()
        results["is_weak"] = (probabilities >= t).astype(int)
        results["confidence"] = probabilities
        # Severity relative to the tuned threshold: flagged concepts are High / Critical
        results["severity"] = pd.cut(
            probabilities,
            bins=[-0.001, t / 2, t, max(0.5, t + 1e-6), 1.0],
            labels=["Low", "Moderate", "High", "Critical"],
        ).astype(str)

        # Rank by confidence (highest probability of being weak = highest priority)
        results["priority_rank"] = results.groupby("id_student")["confidence"].rank(
            ascending=False, method="min"
        ).astype(int)

        # Sort by priority
        results = results.sort_values(["id_student", "priority_rank"])

        # Map concept names if available
        if concept_map:
            results["concept_name"] = results["concept"].map(
                lambda c: concept_map.get(c, c)
            )

        return results

    def get_student_gaps(
        self,
        student_id: int,
        gap_features: pd.DataFrame,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Get top-K weak topics for a specific student.
        Returns list of dicts with concept info and severity.
        """
        ranked = self.rank_student_concepts(student_id, gap_features)
        return [g for g in ranked if g["is_weak"]][:top_k]

    def rank_student_concepts(self, student_id: int, gap_features: pd.DataFrame) -> list[dict]:
        """
        Rank ALL of a student's concepts by predicted risk of scoring below the
        threshold on the next assessment (highest risk first).
        """
        student_data = gap_features[gap_features["id_student"] == student_id]
        if student_data.empty:
            logger.warning(f"No data found for student {student_id}")
            return []

        gaps = self.predict_gaps(student_data)
        return [
            {
                "concept": row["concept"],
                "confidence": round(float(row["confidence"]), 3),
                "severity": str(row["severity"]),
                "priority": int(row["priority_rank"]),
                "is_weak": bool(row["is_weak"]),
                "avg_score": round(float(student_data.loc[idx, "avg_score"]), 3) if "avg_score" in student_data else None,
            }
            for idx, row in gaps.iterrows()
        ]

    def save(self, path: Optional[Path] = None):
        """Save trained model to disk."""
        path = path or MODELS_DIR / "gap_detector.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_columns": self.feature_columns,
                "feature_importance": self.feature_importance,
                "config": self.config,
                "threshold": self.threshold,
            }, f)
        logger.info(f"  ✓ Gap detector saved to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "GapDetector":
        """Load trained model from disk."""
        path = path or MODELS_DIR / "gap_detector.pkl"

        with open(path, "rb") as f:
            data = pickle.load(f)

        detector = cls(config=data["config"])
        detector.model = data["model"]
        detector.feature_columns = data["feature_columns"]
        detector.feature_importance = data["feature_importance"]
        detector.threshold = data.get("threshold", 0.5)

        logger.info(f"  ✓ Gap detector loaded from {path}")
        return detector
