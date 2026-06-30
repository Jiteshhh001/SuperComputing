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
from sklearn.model_selection import StratifiedKFold
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

    Features:
        - Score statistics (avg, min, max, std, trend)
        - Lag features (last 3 scores)
        - VLE engagement metrics
        - Student demographics
        - DKT mastery scores (optional, for enhanced predictions)

    Target:
        - is_weak: 1 if concept mastery < threshold, 0 otherwise
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or XGBOOST_CONFIG
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_columns: list[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None

    def train(
        self,
        gap_features: pd.DataFrame,
        target_col: str = "is_weak",
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
        exclude_cols = [target_col, "id_student", "concept"]
        self.feature_columns = [
            c for c in gap_features.columns if c not in exclude_cols
        ]
        X = gap_features[self.feature_columns].copy()
        y = gap_features[target_col].copy()

        # Handle class imbalance
        pos_ratio = y.sum() / len(y)
        scale_pos = (1 - pos_ratio) / max(pos_ratio, 0.01)
        logger.info(f"  Class distribution: weak={y.sum():,} ({pos_ratio:.1%}), strong={(1-y).sum():,}")
        logger.info(f"  Scale pos weight: {scale_pos:.2f}")

        # Configure model
        model_params = {k: v for k, v in self.config.items()
                        if k not in ["early_stopping_rounds", "random_state", "eval_metric"]}
        model_params["scale_pos_weight"] = scale_pos
        model_params["random_state"] = self.config.get("random_state", 42)
        model_params["eval_metric"] = self.config.get("eval_metric", "logloss")
        # Note: use_label_encoder was removed in XGBoost 2.0+

        # Cross-validation
        cv_results = {"f1": [], "precision": [], "recall": [], "auc": []}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = xgb.XGBClassifier(**model_params)
            fold_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = fold_model.predict(X_val)
            y_prob = fold_model.predict_proba(X_val)[:, 1]

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

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        results = student_features[["id_student", "concept"]].copy()
        results["is_weak"] = predictions
        results["confidence"] = probabilities
        results["severity"] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=["Low", "Moderate", "High", "Critical"],
        )

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
        student_data = gap_features[gap_features["id_student"] == student_id]

        if student_data.empty:
            logger.warning(f"No data found for student {student_id}")
            return []

        gaps = self.predict_gaps(student_data)
        weak_gaps = gaps[gaps["is_weak"] == 1].head(top_k)

        return [
            {
                "concept": row["concept"],
                "confidence": round(row["confidence"], 3),
                "severity": row["severity"],
                "priority": row["priority_rank"],
            }
            for _, row in weak_gaps.iterrows()
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

        logger.info(f"  ✓ Gap detector loaded from {path}")
        return detector
