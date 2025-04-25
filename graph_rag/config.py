import logging
from sentence_transformers import SentenceTransformer
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
logger = logging.getLogger(__name__)
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
STATIC_DIR = PROJ_ROOT / "static"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

model_aml62 = SentenceTransformer("all-MiniLM-L6-v2")
model_amb2 = SentenceTransformer("all-mpnet-base-v2")
model_adr1 = SentenceTransformer("all-distilroberta-v1")

model_params = {
    "all-MiniLM-L6-v2": {"acronym": "aml62", "dims": 384, "max_seq_length": 256, "score": ["dot", "cos", "euc"],
                         "model": model_aml62},
    "all-mpnet-base-v2": {"acronym": "amb2", "dims": 768, "max_seq_len": 384, "score": ["dot", "cos", "euc"],
                          "model": model_amb2},
    "all-distilroberta-v1": {"acronym": "adr1", "dims": 768, "max_seq_length": 512, "score": ["dot", "cos", "euc"],
                             "model": model_adr1}
}
