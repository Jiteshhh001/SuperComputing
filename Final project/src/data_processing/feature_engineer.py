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
from config import PROCESSED_DIR, MASTERY_THRESHOLD, GAP_SCORE_THRESHOLD, DKT_CONFIG

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

        # Add VLE engagement as additional concepts (aggregate first: studentVle has ~10M rows)
        svle = (student_vle.groupby(["id_student", "code_module", "id_site"], observed=True)["sum_click"]
                .sum().reset_index())
        svle = svle.merge(
            vle[["id_site", "activity_type"]].drop_duplicates("id_site"),
            on="id_site",
            how="left"
        )
        svle["concept"] = (svle["code_module"].astype(str) + "_vle_"
                           + svle["activity_type"].fillna("unknown").astype(str))

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

        # Concept map must follow the encoder's (sorted) class order so that
        # concept_id i in the sequences maps to the right concept name.
        self._concept_map = {i: c for i, c in enumerate(self.concept_encoder.classes_.tolist())}

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
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build the XGBoost gap-detector datasets without target leakage.

        The detector predicts whether a student will score below the mastery
        threshold on their NEXT assessment of a concept, using only what was
        known before that assessment:
            - score history for the concept (avg/min/max/std, trend, last 3 scores)
            - submission behaviour (late-submission ratio, timing)
            - VLE engagement in the module up to the last submission
            - student demographics and module / assessment type

        Returns:
            (train_df, current_df)
            train_df   — one row per (student, concept, history prefix) with the
                         target ``next_is_weak`` (score of the following attempt < 50%)
            current_df — one row per (student, concept) built from ALL attempts,
                         used at inference time to predict the next assessment
        """
        logger.info("Building gap detection features (history → next assessment)...")

        sa = student_assessment.merge(assessments, on="id_assessment", how="left")
        sa = sa.dropna(subset=["score", "code_module", "assessment_type"])
        sa["concept"] = sa["code_module"] + "_" + sa["assessment_type"]
        sa["score_norm"] = (sa["score"] / 100.0).clip(0, 1)
        deadline = sa["date"] if "date" in sa.columns else pd.Series(np.nan, index=sa.index)
        sa["is_late"] = ((sa["date_submitted"] - deadline) > 0).astype(float).where(deadline.notna(), 0.0)
        sa = sa.sort_values(["id_student", "concept", "date_submitted", "id_assessment"])

        # Cumulative VLE clicks per (student, module) by day → clicks before a cutoff day
        daily = (student_vle.groupby(["id_student", "code_module", "date"], observed=True)["sum_click"]
                 .sum().reset_index().sort_values(["id_student", "code_module", "date"]))
        vle_index = {}
        daily["code_module"] = daily["code_module"].astype(str)
        for (sid, mod), g in daily.groupby(["id_student", "code_module"], sort=False):
            d = g["date"].to_numpy()
            vle_index[(sid, mod)] = (d, np.cumsum(g["sum_click"].to_numpy()), np.arange(1, len(d) + 1))

        def vle_before(sid, mod, cutoff):
            entry = vle_index.get((sid, mod))
            if entry is None:
                return 0.0, 0
            dates, cum_clicks, cum_days = entry
            pos = np.searchsorted(dates, cutoff, side="right")
            return (float(cum_clicks[pos - 1]), int(cum_days[pos - 1])) if pos > 0 else (0.0, 0)

        module_codes = {m: i for i, m in enumerate(sorted(sa["code_module"].unique()))}
        type_codes = {"CMA": 0, "TMA": 1, "Exam": 2}

        def history_features(sid, concept, module, a_type, scores, late, days):
            n = len(scores)
            clicks, active_days = vle_before(sid, module, days[-1])
            return {
                "id_student": sid,
                "concept": concept,
                "avg_score": float(np.mean(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "score_std": float(np.std(scores)) if n > 1 else 0.0,
                "num_attempts": n,
                "score_trend": self._compute_trend(scores),
                "lag_1": float(scores[-1]),
                "lag_2": float(scores[-2]) if n >= 2 else float(scores[-1]),
                "lag_3": float(scores[-3]) if n >= 3 else float(scores[0]),
                "late_ratio": float(np.mean(late)),
                "last_submission_day": float(days[-1]),
                "vle_clicks_log": float(np.log1p(clicks)),
                "vle_active_days": active_days,
                "module_code": module_codes.get(module, -1),
                "assessment_type_code": type_codes.get(a_type, 3),
            }

        train_rows, current_rows = [], []
        for (sid, concept), g in sa.groupby(["id_student", "concept"], sort=False):
            scores = g["score_norm"].to_numpy()
            late = g["is_late"].to_numpy()
            days = g["date_submitted"].to_numpy()
            module, a_type = g["code_module"].iat[0], g["assessment_type"].iat[0]
            current_rows.append(history_features(sid, concept, module, a_type, scores, late, days))
            for k in range(1, len(scores)):
                row = history_features(sid, concept, module, a_type, scores[:k], late[:k], days[:k])
                row["next_is_weak"] = int(scores[k] < GAP_SCORE_THRESHOLD)
                train_rows.append(row)

        train_df = pd.DataFrame(train_rows)
        current_df = pd.DataFrame(current_rows)

        # Student demographics (same encoding for both frames)
        demo_cols = ["id_student", "gender", "region", "highest_education", "imd_band",
                     "age_band", "num_of_prev_attempts", "studied_credits", "disability"]
        demo = (student_info[[c for c in demo_cols if c in student_info.columns]]
                .drop_duplicates("id_student").copy())
        for col in ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]:
            if col in demo.columns:
                demo[col] = LabelEncoder().fit_transform(demo[col].astype(str))

        train_df = train_df.merge(demo, on="id_student", how="left").fillna(0)
        current_df = current_df.merge(demo, on="id_student", how="left").fillna(0)

        logger.info(f"  ✓ Gap training rows: {len(train_df):,} "
                    f"(next-assessment weak rate {train_df['next_is_weak'].mean():.2%})")
        logger.info(f"  ✓ Gap inference rows: {len(current_df):,} student-concept pairs")
        return train_df, current_df

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
