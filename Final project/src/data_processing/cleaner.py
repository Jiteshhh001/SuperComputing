"""
═══════════════════════════════════════════════════════════════════════════════
  Data Cleaner — Handles missing values, outliers, type casting, deduplication
  Production-grade cleaning pipeline for educational datasets.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Comprehensive data cleaning pipeline.
    Tracks all transformations for audit/reproducibility.
    """

    def __init__(self):
        self.audit_log: list[dict] = []

    def _log(self, action: str, details: str, rows_affected: int = 0):
        """Log a cleaning action for audit trail."""
        entry = {"action": action, "details": details, "rows_affected": rows_affected}
        self.audit_log.append(entry)
        logger.info(f"  → {action}: {details} ({rows_affected:,} rows affected)")

    def clean_oulad(self, tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Apply full cleaning pipeline to OULAD tables."""
        logger.info("Cleaning OULAD dataset...")
        cleaned = {}

        for name, df in tables.items():
            df = df.copy()
            initial_rows = len(df)

            # Remove exact duplicates
            df = self._remove_duplicates(df, name)

            # Table-specific cleaning
            if name == "studentInfo":
                df = self._clean_student_info(df)
            elif name == "studentAssessment":
                df = self._clean_student_assessment(df)
            elif name == "studentVle":
                df = self._clean_student_vle(df)
            elif name == "studentRegistration":
                df = self._clean_student_registration(df)

            cleaned[name] = df
            final_rows = len(df)
            self._log("clean_table", f"{name}: {initial_rows} → {final_rows} rows",
                       initial_rows - final_rows)

        return cleaned

    def clean_uci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full cleaning pipeline to UCI Student Performance data."""
        logger.info("Cleaning UCI dataset...")
        df = df.copy()
        initial_rows = len(df)

        # Remove duplicates
        df = self._remove_duplicates(df, "uci")

        # Type casting — encode binary yes/no columns
        binary_cols = ["schoolsup", "famsup", "paid", "activities",
                       "nursery", "higher", "internet", "romantic"]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({"yes": 1, "no": 0}).fillna(df[col])

        # Handle numeric columns
        numeric_cols = ["age", "Medu", "Fedu", "traveltime", "studytime",
                        "failures", "famrel", "freetime", "goout",
                        "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Impute missing numeric values with median
        df = self._impute_numeric(df, strategy="median")

        # Cap outliers in absences (Winsorize at 99th percentile)
        if "absences" in df.columns:
            cap = df["absences"].quantile(0.99)
            outliers = (df["absences"] > cap).sum()
            df["absences"] = df["absences"].clip(upper=cap)
            self._log("cap_outliers", f"Capped absences at {cap:.0f}", outliers)

        # Validate grade ranges (0-20)
        for g in ["G1", "G2", "G3"]:
            if g in df.columns:
                df[g] = df[g].clip(0, 20)

        self._log("clean_uci", f"UCI cleaned: {initial_rows} → {len(df)} rows",
                   initial_rows - len(df))
        return df

    def _remove_duplicates(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Remove exact duplicate rows."""
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed > 0:
            self._log("remove_duplicates", f"{name}: removed {n_removed} duplicates", n_removed)
        return df

    def _clean_student_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean OULAD studentInfo table."""
        # Fix imd_band — standardize format
        if "imd_band" in df.columns:
            df["imd_band"] = df["imd_band"].fillna("Unknown")
            df["imd_band"] = df["imd_band"].str.strip()

        # Encode final_result to numeric
        result_map = {"Withdrawn": 0, "Fail": 1, "Pass": 2, "Distinction": 3}
        if "final_result" in df.columns:
            df["final_result_numeric"] = df["final_result"].map(result_map)

        # Binary encoding for disability
        if "disability" in df.columns:
            df["has_disability"] = (df["disability"] == "Y").astype(int)

        # Encode gender
        if "gender" in df.columns:
            df["is_male"] = (df["gender"] == "M").astype(int)

        return df

    def _clean_student_assessment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean OULAD studentAssessment table."""
        # Handle missing scores
        if "score" in df.columns:
            missing_scores = df["score"].isna().sum()
            if missing_scores > 0:
                # For missing scores: use median by assessment
                df["score"] = df.groupby("id_assessment")["score"].transform(
                    lambda x: x.fillna(x.median())
                )
                # If still NaN (entire assessment missing), fill with global median
                df["score"] = df["score"].fillna(df["score"].median())
                self._log("impute_scores", f"Imputed {missing_scores} missing scores", missing_scores)

            # Clip scores to valid range
            df["score"] = df["score"].clip(0, 100)

        # Handle missing submission dates
        if "date_submitted" in df.columns:
            missing_dates = df["date_submitted"].isna().sum()
            if missing_dates > 0:
                df["date_submitted"] = df.groupby("id_assessment")["date_submitted"].transform(
                    lambda x: x.fillna(x.median())
                )
                df["date_submitted"] = df["date_submitted"].fillna(0).astype(int)

        return df

    def _clean_student_vle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean OULAD studentVle (interaction logs) table."""
        # Remove impossible click counts
        if "sum_click" in df.columns:
            invalid = (df["sum_click"] <= 0).sum()
            df = df[df["sum_click"] > 0]
            if invalid > 0:
                self._log("remove_invalid_clicks", f"Removed {invalid} zero/negative clicks", invalid)

            # Cap extreme outliers (> 99.5th percentile)
            cap = df["sum_click"].quantile(0.995)
            df["sum_click"] = df["sum_click"].clip(upper=cap)

        return df

    def _clean_student_registration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean OULAD studentRegistration table."""
        # Fill missing registration dates with 0 (registered on start day)
        if "date_registration" in df.columns:
            df["date_registration"] = df["date_registration"].fillna(0).astype(int)

        return df

    def _impute_numeric(self, df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
        """Impute missing numeric values using specified strategy."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_before = df[numeric_cols].isna().sum().sum()

        if strategy == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == "zero":
            df[numeric_cols] = df[numeric_cols].fillna(0)

        if missing_before > 0:
            self._log("impute_numeric", f"Imputed {missing_before} missing values ({strategy})",
                       missing_before)
        return df

    def get_cleaning_report(self) -> pd.DataFrame:
        """Return audit log as a DataFrame."""
        return pd.DataFrame(self.audit_log)

    def get_missing_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate a detailed missing value summary."""
        total = len(df)
        missing = df.isna().sum()
        percent = (missing / total * 100).round(2)

        summary = pd.DataFrame({
            "column": missing.index,
            "missing_count": missing.values,
            "missing_pct": percent.values,
            "dtype": [str(df[col].dtype) for col in missing.index],
            "nunique": [df[col].nunique() for col in missing.index],
        })
        summary = summary[summary["missing_count"] > 0].sort_values(
            "missing_pct", ascending=False
        )
        return summary
