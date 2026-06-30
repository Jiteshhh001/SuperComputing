"""
═══════════════════════════════════════════════════════════════════════════════
  Dataset Downloader — OULAD & UCI Student Performance
  Downloads, extracts, and validates both educational datasets.
═══════════════════════════════════════════════════════════════════════════════
"""

import io
import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, OULAD_URL, UCI_URL, OULAD_TABLES

logger = logging.getLogger(__name__)


def download_file(url: str, dest: Path, description: str = "Downloading") -> Path:
    """Download a file with progress bar and retry logic."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info(f"  ✓ Already exists: {dest.name}")
        return dest

    logger.info(f"  ↓ Downloading: {description}")

    for attempt in range(3):
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with open(dest, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"  ✓ Saved: {dest}")
            return dest

        except requests.RequestException as e:
            logger.warning(f"  ⚠ Attempt {attempt + 1}/3 failed: {e}")
            if dest.exists():
                dest.unlink()
            if attempt == 2:
                raise

    return dest


def download_oulad(target_dir: Path | None = None) -> Path:
    """
    Download and extract the Open University Learning Analytics Dataset.
    The dataset contains 7 CSV files in a zip archive.
    """
    target_dir = target_dir or RAW_DIR / "oulad"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing_tables = [f.stem for f in target_dir.glob("*.csv")]
    if all(t in existing_tables for t in OULAD_TABLES):
        logger.info("✓ OULAD dataset already present — skipping download.")
        return target_dir

    logger.info("━" * 60)
    logger.info("Downloading OULAD Dataset...")
    logger.info("━" * 60)

    zip_path = target_dir / "oulad.zip"

    try:
        download_file(OULAD_URL, zip_path, "OULAD Dataset (~20MB)")

        # Extract
        logger.info("  📦 Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

        # Clean up zip
        zip_path.unlink()
        logger.info(f"  ✓ OULAD extracted to: {target_dir}")

    except Exception as e:
        logger.error(f"  ✗ Failed to download OULAD: {e}")
        logger.info("  → Generating synthetic OULAD data as fallback...")
        _generate_synthetic_oulad(target_dir)

    # Validate
    _validate_oulad(target_dir)
    return target_dir


def download_uci(target_dir: Path | None = None) -> Path:
    """
    Download and extract the UCI Student Performance Dataset.
    Contains student-mat.csv and student-por.csv.
    """
    target_dir = target_dir or RAW_DIR / "uci"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    if (target_dir / "student-mat.csv").exists() and (target_dir / "student-por.csv").exists():
        logger.info("✓ UCI dataset already present — skipping download.")
        return target_dir

    logger.info("━" * 60)
    logger.info("Downloading UCI Student Performance Dataset...")
    logger.info("━" * 60)

    zip_path = target_dir / "student.zip"

    try:
        download_file(UCI_URL, zip_path, "UCI Student Performance (~30KB)")

        # Extract
        logger.info("  📦 Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

        # Move files from subdirectory if needed
        sub_dir = target_dir / "student"
        if sub_dir.exists():
            for f in sub_dir.iterdir():
                dest = target_dir / f.name
                if not dest.exists():
                    f.rename(dest)
            sub_dir.rmdir()

        # Clean up zip
        zip_path.unlink()
        logger.info(f"  ✓ UCI extracted to: {target_dir}")

    except Exception as e:
        logger.error(f"  ✗ Failed to download UCI: {e}")
        logger.info("  → Generating synthetic UCI data as fallback...")
        _generate_synthetic_uci(target_dir)

    return target_dir


def _validate_oulad(target_dir: Path) -> None:
    """Validate that all expected OULAD tables are present."""
    for table in OULAD_TABLES:
        csv_path = target_dir / f"{table}.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path, nrows=1)
            logger.info(f"  ✓ {table}.csv — {len(df.columns)} columns")
        else:
            logger.warning(f"  ⚠ Missing: {table}.csv")


def _generate_synthetic_oulad(target_dir: Path) -> None:
    """Generate realistic synthetic OULAD data for development/demo."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n_students = 2000
    n_modules = 4
    n_assessments_per_module = 6
    n_vle_resources = 50

    modules = ["AAA", "BBB", "CCC", "DDD"]
    presentations = ["2013J", "2014B", "2014J"]
    regions = ["London", "South", "North", "Scotland", "Wales",
               "East Midlands", "West Midlands", "Yorkshire", "East Anglian"]
    education_levels = ["No Formal quals", "Lower Than A Level",
                        "A Level or Equivalent", "HE Qualification",
                        "Post Graduate Qualification"]
    age_bands = ["0-35", "35-55", "55<="]
    results = ["Pass", "Fail", "Distinction", "Withdrawn"]

    # ── courses.csv ──
    courses_rows = []
    for mod in modules:
        for pres in presentations:
            courses_rows.append({
                "code_module": mod,
                "code_presentation": pres,
                "module_presentation_length": np.random.choice([240, 260, 270])
            })
    pd.DataFrame(courses_rows).to_csv(target_dir / "courses.csv", index=False)

    # ── studentInfo.csv ──
    student_ids = list(range(10000, 10000 + n_students))
    student_info_rows = []
    for sid in student_ids:
        mod = np.random.choice(modules)
        pres = np.random.choice(presentations)
        student_info_rows.append({
            "code_module": mod,
            "code_presentation": pres,
            "id_student": sid,
            "gender": np.random.choice(["M", "F"]),
            "region": np.random.choice(regions),
            "highest_education": np.random.choice(education_levels),
            "imd_band": f"{np.random.choice(range(0, 100, 10))}-{np.random.choice(range(10, 110, 10))}%",
            "age_band": np.random.choice(age_bands, p=[0.6, 0.3, 0.1]),
            "num_of_prev_attempts": np.random.choice([0, 0, 0, 1, 1, 2]),
            "studied_credits": np.random.choice([30, 60, 90, 120]),
            "disability": np.random.choice(["N", "Y"], p=[0.9, 0.1]),
            "final_result": np.random.choice(results, p=[0.4, 0.2, 0.15, 0.25]),
        })
    student_info = pd.DataFrame(student_info_rows)
    student_info.to_csv(target_dir / "studentInfo.csv", index=False)

    # ── assessments.csv ──
    assessment_rows = []
    assessment_id = 1000
    for mod in modules:
        for pres in presentations:
            for i in range(n_assessments_per_module):
                assessment_rows.append({
                    "code_module": mod,
                    "code_presentation": pres,
                    "id_assessment": assessment_id,
                    "assessment_type": np.random.choice(["TMA", "CMA", "Exam"],
                                                         p=[0.5, 0.3, 0.2]),
                    "date": int(30 + i * 35 + np.random.randint(-5, 5)),
                    "weight": round(np.random.uniform(10, 30), 1),
                })
                assessment_id += 1
    assessments = pd.DataFrame(assessment_rows)
    assessments.to_csv(target_dir / "assessments.csv", index=False)

    # ── studentAssessment.csv ──
    sa_rows = []
    for _, student in student_info.iterrows():
        mod_assessments = assessments[
            (assessments["code_module"] == student["code_module"]) &
            (assessments["code_presentation"] == student["code_presentation"])
        ]
        for _, asmt in mod_assessments.iterrows():
            if np.random.random() < 0.85:  # 85% submission rate
                base_score = {"Pass": 60, "Fail": 35, "Distinction": 82,
                              "Withdrawn": 30}[student["final_result"]]
                score = np.clip(base_score + np.random.normal(0, 15), 0, 100)
                days_before = int(np.random.exponential(5))
                sa_rows.append({
                    "id_assessment": asmt["id_assessment"],
                    "id_student": student["id_student"],
                    "date_submitted": max(1, asmt["date"] - days_before),
                    "is_banked": 0,
                    "score": round(score, 1),
                })
    pd.DataFrame(sa_rows).to_csv(target_dir / "studentAssessment.csv", index=False)

    # ── vle.csv ──
    activity_types = ["forumng", "oucontent", "resource", "subpage", "homepage",
                      "url", "ouwiki", "glossary", "quiz", "ouelluminate",
                      "questionnaire", "page", "dualpane"]
    vle_rows = []
    site_id = 2000
    for mod in modules:
        for pres in presentations:
            for i in range(n_vle_resources):
                vle_rows.append({
                    "id_site": site_id,
                    "code_module": mod,
                    "code_presentation": pres,
                    "activity_type": np.random.choice(activity_types),
                    "week_from": np.random.randint(1, 15),
                    "week_to": np.random.randint(15, 35),
                })
                site_id += 1
    vle_df = pd.DataFrame(vle_rows)
    vle_df.to_csv(target_dir / "vle.csv", index=False)

    # ── studentVle.csv (interaction logs) ──
    sv_rows = []
    for _, student in student_info.sample(min(800, len(student_info))).iterrows():
        mod_vle = vle_df[
            (vle_df["code_module"] == student["code_module"]) &
            (vle_df["code_presentation"] == student["code_presentation"])
        ]
        n_interactions = np.random.randint(20, 200)
        engagement = {"Pass": 1.0, "Fail": 0.5, "Distinction": 1.5,
                       "Withdrawn": 0.3}[student["final_result"]]
        for _ in range(int(n_interactions * engagement)):
            if len(mod_vle) > 0:
                site = mod_vle.sample(1).iloc[0]
                sv_rows.append({
                    "code_module": student["code_module"],
                    "code_presentation": student["code_presentation"],
                    "id_student": student["id_student"],
                    "id_site": site["id_site"],
                    "date": np.random.randint(-20, 260),
                    "sum_click": int(np.random.exponential(5) + 1),
                })
    pd.DataFrame(sv_rows).to_csv(target_dir / "studentVle.csv", index=False)

    # ── studentRegistration.csv ──
    reg_rows = []
    for _, student in student_info.iterrows():
        unreg_date = None
        if student["final_result"] == "Withdrawn":
            unreg_date = np.random.randint(30, 200)
        reg_rows.append({
            "code_module": student["code_module"],
            "code_presentation": student["code_presentation"],
            "id_student": student["id_student"],
            "date_registration": np.random.randint(-25, 0),
            "date_unregistration": unreg_date,
        })
    pd.DataFrame(reg_rows).to_csv(target_dir / "studentRegistration.csv", index=False)

    logger.info(f"  ✓ Generated synthetic OULAD data ({n_students} students)")


def _generate_synthetic_uci(target_dir: Path) -> None:
    """Generate realistic synthetic UCI Student Performance data."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)

    columns = {
        "school": ["GP", "MS"],
        "sex": ["F", "M"],
        "age": list(range(15, 23)),
        "address": ["U", "R"],
        "famsize": ["LE3", "GT3"],
        "Pstatus": ["T", "A"],
        "Medu": [0, 1, 2, 3, 4],
        "Fedu": [0, 1, 2, 3, 4],
        "Mjob": ["teacher", "health", "services", "at_home", "other"],
        "Fjob": ["teacher", "health", "services", "at_home", "other"],
        "reason": ["home", "reputation", "course", "other"],
        "guardian": ["mother", "father", "other"],
    }

    for dataset_name, n in [("student-mat", 395), ("student-por", 649)]:
        rows = []
        for _ in range(n):
            study = np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.25, 0.15])
            failures = np.random.choice([0, 1, 2, 3], p=[0.65, 0.2, 0.1, 0.05])
            base = 7 + study * 2 - failures * 1.5 + np.random.normal(0, 2)
            g1 = int(np.clip(base + np.random.normal(0, 2), 0, 20))
            g2 = int(np.clip(g1 + np.random.normal(0, 1.5), 0, 20))
            g3 = int(np.clip(g2 + np.random.normal(0, 2), 0, 20))

            rows.append({
                "school": np.random.choice(columns["school"], p=[0.8, 0.2]),
                "sex": np.random.choice(columns["sex"]),
                "age": np.random.choice(columns["age"], p=[0.05, 0.2, 0.25, 0.25, 0.15, 0.05, 0.03, 0.02]),
                "address": np.random.choice(columns["address"], p=[0.7, 0.3]),
                "famsize": np.random.choice(columns["famsize"], p=[0.35, 0.65]),
                "Pstatus": np.random.choice(columns["Pstatus"], p=[0.85, 0.15]),
                "Medu": np.random.choice(columns["Medu"], p=[0.05, 0.15, 0.25, 0.3, 0.25]),
                "Fedu": np.random.choice(columns["Fedu"], p=[0.08, 0.2, 0.3, 0.25, 0.17]),
                "Mjob": np.random.choice(columns["Mjob"]),
                "Fjob": np.random.choice(columns["Fjob"]),
                "reason": np.random.choice(columns["reason"]),
                "guardian": np.random.choice(columns["guardian"], p=[0.6, 0.3, 0.1]),
                "traveltime": np.random.choice([1, 2, 3, 4], p=[0.45, 0.35, 0.15, 0.05]),
                "studytime": study,
                "failures": failures,
                "schoolsup": np.random.choice(["yes", "no"], p=[0.1, 0.9]),
                "famsup": np.random.choice(["yes", "no"], p=[0.6, 0.4]),
                "paid": np.random.choice(["yes", "no"], p=[0.5, 0.5]),
                "activities": np.random.choice(["yes", "no"]),
                "nursery": np.random.choice(["yes", "no"], p=[0.8, 0.2]),
                "higher": np.random.choice(["yes", "no"], p=[0.9, 0.1]),
                "internet": np.random.choice(["yes", "no"], p=[0.7, 0.3]),
                "romantic": np.random.choice(["yes", "no"], p=[0.35, 0.65]),
                "famrel": np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.08, 0.2, 0.4, 0.27]),
                "freetime": np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.4, 0.3, 0.1]),
                "goout": np.random.choice([1, 2, 3, 4, 5], p=[0.08, 0.2, 0.35, 0.25, 0.12]),
                "Dalc": np.random.choice([1, 2, 3, 4, 5], p=[0.6, 0.2, 0.1, 0.06, 0.04]),
                "Walc": np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.25, 0.25, 0.12, 0.08]),
                "health": np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.25, 0.25, 0.3]),
                "absences": int(np.clip(np.random.exponential(5), 0, 75)),
                "G1": g1,
                "G2": g2,
                "G3": g3,
            })
        pd.DataFrame(rows).to_csv(target_dir / f"{dataset_name}.csv", index=False, sep=";")

    logger.info("  ✓ Generated synthetic UCI data (395 math + 649 por)")


def download_all() -> dict[str, Path]:
    """Download all datasets. Returns dict of dataset name → path."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║     Personalized Learning Agent — Dataset Download      ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    paths = {}
    paths["oulad"] = download_oulad()
    paths["uci"] = download_uci()

    logger.info("━" * 60)
    logger.info("✓ All datasets ready!")
    logger.info("━" * 60)

    return paths


if __name__ == "__main__":
    download_all()
