"""
═══════════════════════════════════════════════════════════════════════════════
  Data Loader — Unified interface for loading OULAD & UCI datasets
  Handles merging, type casting, and providing clean DataFrames.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import RAW_DIR, OULAD_TABLES

logger = logging.getLogger(__name__)


class OULADLoader:
    """Loader for the Open University Learning Analytics Dataset."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or RAW_DIR / "oulad"
        self._tables: dict[str, pd.DataFrame] = {}

    def load_table(self, table_name: str) -> pd.DataFrame:
        """Load a single OULAD table by name."""
        if table_name in self._tables:
            return self._tables[table_name]

        csv_path = self.data_dir / f"{table_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"OULAD table not found: {csv_path}")

        df = pd.read_csv(csv_path)
        self._tables[table_name] = df
        logger.info(f"  Loaded {table_name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all 7 OULAD tables."""
        logger.info("Loading OULAD tables...")
        for table in OULAD_TABLES:
            self.load_table(table)
        return self._tables

    @property
    def courses(self) -> pd.DataFrame:
        return self.load_table("courses")

    @property
    def student_info(self) -> pd.DataFrame:
        return self.load_table("studentInfo")

    @property
    def assessments(self) -> pd.DataFrame:
        return self.load_table("assessments")

    @property
    def student_assessment(self) -> pd.DataFrame:
        return self.load_table("studentAssessment")

    @property
    def student_vle(self) -> pd.DataFrame:
        return self.load_table("studentVle")

    @property
    def student_registration(self) -> pd.DataFrame:
        return self.load_table("studentRegistration")

    @property
    def vle(self) -> pd.DataFrame:
        return self.load_table("vle")

    def build_unified_student_view(self) -> pd.DataFrame:
        """
        Merge all OULAD tables into a unified student-level view.
        Joins: studentInfo + assessments + studentAssessment + VLE engagement.
        """
        logger.info("Building unified student view...")

        # Start with student info
        unified = self.student_info.copy()

        # Merge course info
        unified = unified.merge(
            self.courses,
            on=["code_module", "code_presentation"],
            how="left"
        )

        # Aggregate assessment scores per student
        sa = self.student_assessment.merge(
            self.assessments,
            on="id_assessment",
            how="left"
        )
        student_scores = sa.groupby("id_student").agg(
            avg_score=("score", "mean"),
            score_std=("score", "std"),
            min_score=("score", "min"),
            max_score=("score", "max"),
            num_submissions=("score", "count"),
            weighted_avg_score=("score", lambda x: x.mean()),  # Simplified
            avg_days_early=("date_submitted", "mean"),
        ).reset_index()
        student_scores["score_std"] = student_scores["score_std"].fillna(0)

        unified = unified.merge(student_scores, on="id_student", how="left")

        # Aggregate VLE engagement per student
        vle_engagement = self.student_vle.groupby("id_student").agg(
            total_clicks=("sum_click", "sum"),
            total_vle_days=("date", "nunique"),
            avg_daily_clicks=("sum_click", "mean"),
            max_daily_clicks=("sum_click", "max"),
            distinct_resources=("id_site", "nunique"),
        ).reset_index()

        unified = unified.merge(vle_engagement, on="id_student", how="left")

        # Fill NaN engagement data (students with no VLE activity)
        engagement_cols = ["total_clicks", "total_vle_days", "avg_daily_clicks",
                           "max_daily_clicks", "distinct_resources"]
        unified[engagement_cols] = unified[engagement_cols].fillna(0)

        # Merge registration data
        reg = self.student_registration.copy()
        reg["is_unregistered"] = reg["date_unregistration"].notna().astype(int)
        reg = reg[["id_student", "code_module", "code_presentation",
                    "date_registration", "is_unregistered"]]
        unified = unified.merge(
            reg,
            on=["id_student", "code_module", "code_presentation"],
            how="left"
        )

        logger.info(f"  ✓ Unified view: {unified.shape[0]:,} rows × {unified.shape[1]} cols")
        return unified


class UCILoader:
    """Loader for the UCI Student Performance Dataset."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or RAW_DIR / "uci"

    def load_math(self) -> pd.DataFrame:
        """Load Mathematics student performance data."""
        return self._load("student-mat.csv", "Mathematics")

    def load_portuguese(self) -> pd.DataFrame:
        """Load Portuguese language student performance data."""
        return self._load("student-por.csv", "Portuguese")

    def load_combined(self) -> pd.DataFrame:
        """Load and combine both datasets with a 'subject' column."""
        math = self.load_math()
        math["subject"] = "Mathematics"
        por = self.load_portuguese()
        por["subject"] = "Portuguese"
        combined = pd.concat([math, por], ignore_index=True)
        logger.info(f"  ✓ Combined UCI: {combined.shape[0]:,} rows × {combined.shape[1]} cols")
        return combined

    def _load(self, filename: str, label: str) -> pd.DataFrame:
        """Load a single UCI CSV file with proper separator handling."""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"UCI file not found: {path}")

        # UCI uses semicolon separator
        df = pd.read_csv(path, sep=";")
        logger.info(f"  Loaded {label}: {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df


def load_all_data() -> dict[str, pd.DataFrame]:
    """
    Load all datasets and return as a dictionary.

    Returns:
        dict with keys:
            - 'oulad_unified': Merged OULAD student view
            - 'oulad_tables': Dict of individual OULAD tables
            - 'uci_math': UCI Mathematics data
            - 'uci_por': UCI Portuguese data
            - 'uci_combined': Both UCI datasets combined
    """
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║           Loading All Educational Datasets              ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    oulad = OULADLoader()
    uci = UCILoader()

    result = {
        "oulad_tables": oulad.load_all(),
        "oulad_unified": oulad.build_unified_student_view(),
        "uci_math": uci.load_math(),
        "uci_por": uci.load_portuguese(),
        "uci_combined": uci.load_combined(),
    }

    total_rows = sum(df.shape[0] for df in result.values() if isinstance(df, pd.DataFrame))
    logger.info(f"✓ Total records loaded: {total_rows:,}")
    return result
