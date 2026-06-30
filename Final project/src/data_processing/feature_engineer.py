"""
═══════════════════════════════════════════════════════════════════════════════
  Feature Engineer — Student-concept matrices, lag features, DKT sequences
  Transforms raw data into model-ready features for all downstream models.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import PROCESSED_DIR, MASTERY_THRESHOLD, DKT_CONFIG

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for the learning agent pipeline.
    Produces:
        1. Student-concept interaction matrix (for recommender)
        2. DKT interaction sequences (for LSTM)
        3. Gap detection features (for XGBoost)
        4. Student profile features (for analytics)
    """

    def __init__(self):
        self.concept_encoder = LabelEncoder()
        self.student_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._concept_map: dict[int, str] = {}
        self._student_map: dict[int, int] = {}

    # ═══════════════════════════════════════════════════════════════════════
    #  1. Student-Concept Interaction Matrix
    # ═══════════════════════════════════════════════════════════════════════

    def build_interaction_matrix(
        self,
        student_assessment: pd.DataFrame,
        assessments: pd.DataFrame,
        student_vle: pd.DataFrame,
        vle: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build a student × concept interaction matrix.
        Each cell = aggregated performance score (0-1) for that concept.

        Concepts are derived from: assessment types × modules + VLE activity types.
        """
        logger.info("Building student-concept interaction matrix...")

        # Merge assessment data
        sa = student_assessment.merge(assessments, on="id_assessment", how="left")

        # Create concept_id from module + assessment_type
        sa["concept"] = sa["code_module"] + "_" + sa["assessment_type"]

        # Normalize scores to 0-1
        sa["score_norm"] = sa["score"] / 100.0

        # Student × concept matrix from assessments
        assess_matrix = sa.pivot_table(
            index="id_student",
            columns="concept",
            values="score_norm",
            aggfunc="mean"
        )

        # Add VLE engagement as additional concepts
        svle = student_vle.merge(
            vle[["id_site", "activity_type"]],
            on="id_site",
            how="left"
        )
        svle["concept"] = svle["code_module"] + "_vle_" + svle["activity_type"].fillna("unknown")

        # Normalize VLE clicks using log transform
        vle_matrix = svle.pivot_table(
            index="id_student",
            columns="concept",
            values="sum_click",
            aggfunc="sum"
        )
        if not vle_matrix.empty:
            vle_matrix = np.log1p(vle_matrix)
            # Scale to 0-1
            vle_max = vle_matrix.max()
            vle_max = vle_max.replace(0, 1)
            vle_matrix = vle_matrix / vle_max

        # Combine both matrices
        interaction_matrix = assess_matrix.join(vle_matrix, how="outer")
        interaction_matrix = interaction_matrix.fillna(0)

        # Store concept mapping
        self.concept_encoder.fit(interaction_matrix.columns.tolist())
        self._concept_map = dict(enumerate(interaction_matrix.columns.tolist()))

        logger.info(
            f"  ✓ Interaction matrix: {interaction_matrix.shape[0]:,} students "
            f"× {interaction_matrix.shape[1]} concepts"
        )
        return interaction_matrix

    # ═══════════════════════════════════════════════════════════════════════
    #  2. DKT Interaction Sequences
    # ═══════════════════════════════════════════════════════════════════════

    def build_dkt_sequences(
        self,
        student_assessment: pd.DataFrame,
        assessments: pd.DataFrame,
        max_seq_length: Optional[int] = None,
    ) -> dict:
        """
        Build interaction sequences for Deep Knowledge Tracing.

        Each sequence: [(concept_id, is_correct), ...]
        where is_correct = 1 if score >= 50, else 0.

        Returns:
            dict with:
                - 'sequences': list of (concept_id, is_correct) tuples per student
                - 'student_ids': corresponding student IDs
                - 'num_concepts': total unique concepts
                - 'concept_map': id → concept name mapping
        """
        max_seq = max_seq_length or DKT_CONFIG["max_seq_length"]
        logger.info(f"Building DKT sequences (max_len={max_seq})...")

        # Merge assessment info
        sa = student_assessment.merge(assessments, on="id_assessment", how="left")

        # Create concept from module + assessment_type
        sa["concept"] = sa["code_module"] + "_" + sa["assessment_type"]

        # Encode concepts
        all_concepts = sa["concept"].unique().tolist()
        self.concept_encoder.fit(all_concepts)
        sa["concept_id"] = self.concept_encoder.transform(sa["concept"])
        num_concepts = len(all_concepts)

        # Binary correctness: score >= 50 → correct
        sa["is_correct"] = (sa["score"] >= 50).astype(int)

        # Sort by student and submission date
        sa = sa.sort_values(["id_student", "date_submitted"])

        # Build sequences per student
        sequences = []
        student_ids = []

        for student_id, group in sa.groupby("id_student"):
            seq = list(zip(
                group["concept_id"].values,
                group["is_correct"].values
            ))

            # Truncate to max length (keep most recent interactions)
            if len(seq) > max_seq:
                seq = seq[-max_seq:]

            if len(seq) >= 3:  # Minimum 3 interactions
                sequences.append(seq)
                student_ids.append(student_id)

        # Update concept map
        self._concept_map = {i: c for i, c in enumerate(all_concepts)}

        logger.info(
            f"  ✓ {len(sequences):,} sequences, {num_concepts} concepts, "
            f"avg length: {np.mean([len(s) for s in sequences]):.1f}"
        )

        return {
            "sequences": sequences,
            "student_ids": student_ids,
            "num_concepts": num_concepts,
            "concept_map": self._concept_map,
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  3. Gap Detection Features (for XGBoost)
    # ═══════════════════════════════════════════════════════════════════════

    def build_gap_features(
        self,
        student_assessment: pd.DataFrame,
        assessments: pd.DataFrame,
        student_info: pd.DataFrame,
        student_vle: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build feature matrix for XGBoost gap detector.

        Features per student per concept:
            - avg_score, min_score, max_score, score_trend
            - lag_1, lag_2, lag_3 (last 3 scores)
            - submission_timeliness
            - vle_engagement (clicks for related resources)
            - student demographics
            - is_weak (target: 1 if concept mastery < threshold)
        """
        logger.info("Building gap detection features...")

        # Merge assessment data
        sa = student_assessment.merge(assessments, on="id_assessment", how="left")
        sa["concept"] = sa["code_module"] + "_" + sa["assessment_type"]
        sa["score_norm"] = sa["score"] / 100.0
        sa = sa.sort_values(["id_student", "concept", "date_submitted"])

        features_list = []

        for (student_id, concept), group in sa.groupby(["id_student", "concept"]):
            scores = group["score_norm"].values
            dates = group["date_submitted"].values

            feat = {
                "id_student": student_id,
                "concept": concept,
                # Score statistics
                "avg_score": np.mean(scores),
                "min_score": np.min(scores),
                "max_score": np.max(scores),
                "score_std": np.std(scores) if len(scores) > 1 else 0,
                "num_attempts": len(scores),
                # Score trend (slope of linear regression)
                "score_trend": self._compute_trend(scores),
                # Lag features (last 3 scores)
                "lag_1": scores[-1] if len(scores) >= 1 else 0,
                "lag_2": scores[-2] if len(scores) >= 2 else scores[-1] if len(scores) >= 1 else 0,
                "lag_3": scores[-3] if len(scores) >= 3 else scores[-2] if len(scores) >= 2 else 0,
                # Submission timeliness
                "avg_submission_day": np.mean(dates) if len(dates) > 0 else 0,
                # Target: is_weak
                "is_weak": int(np.mean(scores) < MASTERY_THRESHOLD),
            }
            features_list.append(feat)

        features = pd.DataFrame(features_list)

        # Merge VLE engagement per student
        vle_agg = student_vle.groupby("id_student").agg(
            total_clicks=("sum_click", "sum"),
            total_days=("date", "nunique"),
            avg_clicks=("sum_click", "mean"),
        ).reset_index()
        features = features.merge(vle_agg, on="id_student", how="left")

        # Merge student demographics
        demo_cols = ["id_student", "gender", "region", "highest_education",
                     "age_band", "num_of_prev_attempts", "studied_credits"]
        demo = student_info[[c for c in demo_cols if c in student_info.columns]].copy()

        # Encode categorical features
        cat_cols = ["gender", "region", "highest_education", "age_band"]
        for col in cat_cols:
            if col in demo.columns:
                demo[col] = LabelEncoder().fit_transform(demo[col].astype(str))

        features = features.merge(demo, on="id_student", how="left")

        # Fill remaining NAs
        features = features.fillna(0)

        # Encode concept
        features["concept_encoded"] = LabelEncoder().fit_transform(features["concept"])

        logger.info(f"  ✓ Gap features: {features.shape[0]:,} rows × {features.shape[1]} cols")
        logger.info(f"  ✓ Weak topics ratio: {features['is_weak'].mean():.2%}")
        return features

    # ═══════════════════════════════════════════════════════════════════════
    #  4. Student Profile Features
    # ═══════════════════════════════════════════════════════════════════════

    def build_student_profiles(
        self,
        unified_view: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build comprehensive student profiles for the dashboard and agent.
        """
        logger.info("Building student profiles...")
        profiles = unified_view.copy()

        # Engagement level
        if "total_clicks" in profiles.columns:
            click_q33 = profiles["total_clicks"].quantile(0.33)
            click_q66 = profiles["total_clicks"].quantile(0.66)
            profiles["engagement_level"] = pd.cut(
                profiles["total_clicks"],
                bins=[-1, click_q33, click_q66, float("inf")],
                labels=["Low", "Medium", "High"]
            )

        # Risk level based on scores and engagement
        if "avg_score" in profiles.columns:
            profiles["risk_score"] = 1.0 - (profiles["avg_score"].fillna(0) / 100.0)
            profiles["risk_level"] = pd.cut(
                profiles["risk_score"],
                bins=[-1, 0.3, 0.6, float("inf")],
                labels=["Low Risk", "Medium Risk", "High Risk"]
            )

        logger.info(f"  ✓ Student profiles: {profiles.shape[0]:,} students")
        return profiles

    # ═══════════════════════════════════════════════════════════════════════
    #  Utilities
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_trend(values: np.ndarray) -> float:
        """Compute linear trend (slope) of a sequence."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            return float(slope)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    def save_processed_data(
        self,
        interaction_matrix: pd.DataFrame,
        dkt_data: dict,
        gap_features: pd.DataFrame,
        student_profiles: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> dict[str, Path]:
        """Save all processed data to disk."""
        output_dir = output_dir or PROCESSED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save interaction matrix
        path = output_dir / "interaction_matrix.csv"
        interaction_matrix.to_csv(path)
        paths["interaction_matrix"] = path

        # Save DKT data
        import pickle
        path = output_dir / "dkt_sequences.pkl"
        with open(path, "wb") as f:
            pickle.dump(dkt_data, f)
        paths["dkt_sequences"] = path

        # Save gap features
        path = output_dir / "gap_features.csv"
        gap_features.to_csv(path, index=False)
        paths["gap_features"] = path

        # Save student profiles
        path = output_dir / "student_profiles.csv"
        student_profiles.to_csv(path, index=False)
        paths["student_profiles"] = path

        logger.info(f"  ✓ Saved {len(paths)} processed files to {output_dir}")
        return paths

    @property
    def concept_map(self) -> dict[int, str]:
        return self._concept_map
