import logging
from pathlib import Path

from tqdm import tqdm

from graph_rag.config import MODELS_DIR, PROCESSED_DATA_DIR

def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger = logging.getLogger()
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.info("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    main()
