"""Train the AgriVision AI models and generate explainability artifacts."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_processor import DataProcessor
from explainability import ExplainabilityEngine
from model_trainer import ModelTrainer
from runtime import configure_logging

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "crop_modified.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

logger = configure_logging(__name__)


def main() -> None:
    """Run the end-to-end training pipeline."""

    logger.info("%s", "=" * 60)
    logger.info("AgriVision AI - Model Training Pipeline")
    logger.info("%s", "=" * 60)

    logger.info("[1/4] Loading and engineering features...")
    processor = DataProcessor(DATA_PATH)
    df = processor.full_pipeline()
    X, y, feature_names = processor.get_features_and_target(df)
    logger.info("Dataset: %s rows x %s features", f"{X.shape[0]:,}", X.shape[1])

    logger.info("[2/4] Training ML models...")
    trainer = ModelTrainer(models_dir=MODELS_DIR)
    results = trainer.train_all(X, y)
    logger.info("Model comparison:\n%s", results.to_string(index=False))

    logger.info("[3/4] Best model: %s", trainer.best_model_name)
    trainer.save_best_model(feature_names)

    logger.info("[4/4] Generating SHAP explanations (sample 500 rows)...")
    try:
        engine = ExplainabilityEngine(
            trainer.best_model, trainer.X_train, feature_names, output_dir=REPORTS_DIR
        )
        engine.build_explainer()
        engine.compute_shap_values(trainer.X_test, sample_size=500)
        engine.save_importance_bar_plot()
        engine.save_summary_plot()
        logger.info("SHAP plots saved to reports/")
    except Exception:
        logger.exception("SHAP generation skipped")

    logger.info("Training complete. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
