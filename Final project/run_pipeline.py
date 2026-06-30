"""
═══════════════════════════════════════════════════════════════════════════════
  Pipeline Runner — Executes the full data → model → agent pipeline
  Run this once to download data, train models, and prepare the system.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LOG_FORMAT, LOG_DATE_FORMAT, PROCESSED_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("pipeline")


def run_pipeline():
    """Execute the complete pipeline: data → features → models → ready."""

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║    LearnFlow AI — Full Pipeline Execution               ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # ── Step 1: Download Datasets ────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("STEP 1: Download Datasets")
    logger.info("═" * 60)

    from data.download_datasets import download_all
    download_all()

    # ── Step 2: Load & Clean Data ────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("STEP 2: Load & Clean Data")
    logger.info("═" * 60)

    from src.data_processing.loader import OULADLoader, UCILoader
    from src.data_processing.cleaner import DataCleaner

    oulad = OULADLoader()
    tables = oulad.load_all()

    uci = UCILoader()
    uci_combined = uci.load_combined()

    cleaner = DataCleaner()
    cleaned_tables = cleaner.clean_oulad(tables)
    cleaned_uci = cleaner.clean_uci(uci_combined)

    # Print cleaning report
    report = cleaner.get_cleaning_report()
    logger.info(f"\n  Cleaning report: {len(report)} actions performed")

    # ── Step 3: Feature Engineering ──────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("STEP 3: Feature Engineering")
    logger.info("═" * 60)

    from src.data_processing.feature_engineer import FeatureEngineer

    fe = FeatureEngineer()

    # Build unified view from cleaned tables
    oulad_clean = OULADLoader()
    # We need to use the cleaned tables directly
    import pandas as pd
    # Save cleaned tables temporarily for the loader
    for name, df in cleaned_tables.items():
        csv_path = oulad.data_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)

    # Rebuild unified view
    unified = oulad.build_unified_student_view()

    # Build interaction matrix
    interaction_matrix = fe.build_interaction_matrix(
        cleaned_tables["studentAssessment"],
        cleaned_tables["assessments"],
        cleaned_tables["studentVle"],
        cleaned_tables["vle"],
    )

    # Build DKT sequences
    dkt_data = fe.build_dkt_sequences(
        cleaned_tables["studentAssessment"],
        cleaned_tables["assessments"],
    )

    # Build gap features
    gap_features = fe.build_gap_features(
        cleaned_tables["studentAssessment"],
        cleaned_tables["assessments"],
        cleaned_tables["studentInfo"],
        cleaned_tables["studentVle"],
    )

    # Build student profiles
    student_profiles = fe.build_student_profiles(unified)

    # Save all processed data
    paths = fe.save_processed_data(
        interaction_matrix, dkt_data, gap_features, student_profiles
    )
    logger.info(f"  ✓ Saved processed data to {PROCESSED_DIR}")

    # ── Step 4: Train DKT Model ─────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("STEP 4: Train DKT Model (LSTM)")
    logger.info("═" * 60)

    from src.models.dkt_trainer import DKTTrainer

    dkt_trainer = DKTTrainer(
        num_concepts=dkt_data["num_concepts"],
    )
    dkt_results = dkt_trainer.train(
        sequences=dkt_data["sequences"],
        save_dir=MODELS_DIR,
    )

    logger.info(f"  DKT Test AUC: {dkt_results['test_metrics']['auc']:.4f}")

    # ── Step 5: Train Gap Detector ──────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("STEP 5: Train Gap Detector (XGBoost)")
    logger.info("═" * 60)

    from src.models.gap_detector import GapDetector

    gap_detector = GapDetector()
    gap_results = gap_detector.train(gap_features)
    gap_detector.save()

    logger.info(f"  Gap Detector F1: {gap_results['cv_metrics']['f1']['mean']:.4f}")

    # ── Step 6: Train Recommender ───────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("STEP 6: Train Hybrid Recommender")
    logger.info("═" * 60)

    from src.models.recommender import HybridRecommender

    recommender = HybridRecommender()
    recommender.fit(interaction_matrix)
    recommender.save()

    # ── Step 7: Initialize Agent ────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("STEP 7: Initialize Agent")
    logger.info("═" * 60)

    from src.agent.agent import LearningAgent

    agent = LearningAgent()

    # Test with first available student
    student_ids = student_profiles["id_student"].unique()
    if len(student_ids) > 0:
        test_id = int(student_ids[0])
        logger.info(f"\n  Testing agent with student {test_id}...")
        result = agent.analyze_student(test_id)
        logger.info(f"\n{result['summary']}")

    # ── Done ────────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("═" * 60)
    logger.info("  All models trained and saved. System ready.")
    logger.info("  Run the dashboard: streamlit run app/streamlit_app.py")
    logger.info("  Run the API: uvicorn api.main:app --reload")


if __name__ == "__main__":
    run_pipeline()
