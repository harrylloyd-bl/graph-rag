from pathlib import Path

import dotenv
import os
import pandas as pd
from loguru import logger
from neo4j import GraphDatabase
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
from tqdm import tqdm
import plotly.graph_objects as go
import typer

go.Figure.add_scatter3d()

tqdm.pandas()

GraphDatabase.driver


from graph_rag.config import PROCESSED_DATA_DIR

app = typer.Typer()

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
