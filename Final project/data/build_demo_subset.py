"""
═══════════════════════════════════════════════════════════════════════════════
  Demo Subset Builder — shrink the full OULAD outputs for the hosted dashboard
  Models are trained on the full dataset by run_pipeline.py. This script keeps a
  stratified sample of students (default 2,000) in data/raw and data/processed so
  the repository and the Streamlit deployment stay small. studentVle is written
  as a gzip-compressed CSV.

  Usage:  python data/build_demo_subset.py [--students 2000]
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def build(n_students: int = 2000, seed: int = 42) -> None:
    oulad = RAW_DIR / "oulad"
    with open(PROCESSED_DIR / "dkt_sequences.pkl", "rb") as f:
        dkt = pickle.load(f)
    gap = pd.read_csv(PROCESSED_DIR / "gap_features.csv")
    info = pd.read_csv(oulad / "studentInfo.csv")

    # Eligible: students with a DKT sequence (>= 3 assessments) and gap features
    eligible = set(int(s) for s in dkt["student_ids"]) & set(gap["id_student"].astype(int))
    first_module = (info[info["id_student"].isin(eligible)]
                    .drop_duplicates("id_student")[["id_student", "code_module"]])
    frac = min(1.0, n_students / len(first_module))
    sample = first_module.groupby("code_module").sample(frac=frac, random_state=seed)
    keep = set(sample["id_student"].astype(int))
    logger.info(f"Selected {len(keep):,} of {len(eligible):,} eligible students "
                f"({sample['code_module'].value_counts().to_dict()})")

    # ── raw tables ──
    for name in ["studentInfo", "studentRegistration", "studentAssessment"]:
        df = pd.read_csv(oulad / f"{name}.csv")
        df[df["id_student"].isin(keep)].to_csv(oulad / f"{name}.csv", index=False)
    vle_path = oulad / "studentVle.csv"
    chunks = [c[c["id_student"].isin(keep)] for c in pd.read_csv(vle_path, chunksize=1_000_000)]
    pd.concat(chunks).to_csv(oulad / "studentVle.csv.gz", index=False, compression="gzip")
    vle_path.unlink()

    # ── processed data ──
    idx = [i for i, s in enumerate(dkt["student_ids"]) if int(s) in keep]
    dkt_small = {**dkt, "student_ids": [dkt["student_ids"][i] for i in idx],
                 "sequences": [dkt["sequences"][i] for i in idx]}
    with open(PROCESSED_DIR / "dkt_sequences.pkl", "wb") as f:
        pickle.dump(dkt_small, f)
    gap[gap["id_student"].isin(keep)].to_csv(PROCESSED_DIR / "gap_features.csv", index=False)
    prof = pd.read_csv(PROCESSED_DIR / "student_profiles.csv")
    prof[prof["id_student"].isin(keep)].to_csv(PROCESSED_DIR / "student_profiles.csv", index=False)
    im = pd.read_csv(PROCESSED_DIR / "interaction_matrix.csv", index_col=0)
    im[im.index.isin(keep)].to_csv(PROCESSED_DIR / "interaction_matrix.csv")
    logger.info("✓ Demo subset written to data/raw/oulad and data/processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--students", type=int, default=2000)
    build(parser.parse_args().students)
