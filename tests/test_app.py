import pandas as pd
import pytest

import graph_rag.app as app
from graph_rag.config import STATIC_DIR

MODEL = "all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def umap_data():
    most_common_lcsh_df, titles_df, embedded_df = app.load_umap_data(MODEL)
    return most_common_lcsh_df, titles_df, embedded_df


def test_load_umap_data(umap_data):
    most_common_lcsh_df, titles_df, embedded_df = umap_data

    assert most_common_lcsh_df.shape == (39589, 3)
    assert titles_df.shape == (50000, 1)
    assert embedded_df.shape == (39589, 1)


def test_create_duplicate_map(umap_data):
    _, titles_df, embedded_df = umap_data
    combined_df = titles_df.join(embedded_df).dropna()
    dup_map = app.create_duplicate_map(combined_df)

    assert len(dup_map) == 39589

    s_kw_df = pd.read_csv(STATIC_DIR / "shakespeare_kw.csv", encoding="utf-8-sig", index_col=0)
    s_df = pd.read_csv(STATIC_DIR / "shakespeare.csv", encoding="utf-8-sig", index_col=0)
    eic_df = pd.read_csv(STATIC_DIR / "east_india_company.csv", encoding="utf-8-sig", index_col=0)
    quran_df = pd.read_csv(STATIC_DIR / "quran.csv", encoding="utf-8-sig", index_col=0)

    assert not pd.Index(s_kw_df[MODEL + "_uri"].map(dup_map).values).hasnans
    assert not pd.Index(s_df[MODEL + "_uri"].map(dup_map).values).hasnans
    assert not pd.Index(eic_df[MODEL + "_uri"].map(dup_map).values).hasnans
    assert not pd.Index(quran_df[MODEL + "_uri"].map(dup_map).values).hasnans
