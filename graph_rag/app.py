import os
import pickle

import plotly.graph_objects
from huggingface_hub import InferenceClient
from matplotlib import colormaps
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import umap

from graph_rag.config import *


def kw_search(tx, query=None, limit=5):
    regex = f"(?i).*{query}.*"
    cypher = """
        MATCH (n:bibo__Book)
        WHERE n.dct__title[0] =~ $regex
        RETURN n.dct__title[0]
        LIMIT $limit
    """
    return tx.run(cypher, regex=regex, limit=limit).values()


def semantic_search(tx, model: SentenceTransformer, query: str, idx: str, nn: int) -> list[list[str]]:
    """
    Pass a cypher query on a vector index to a Neo4j driver session
    :param tx:
    :param model: SentenceTransformer
    :param query: str
    :param idx: str
    :param nn: int
    :return: list[list[str]]
    """
    embedded_query = str(model.encode(query).tolist())
    cypher = """
        CALL db.index.vector.queryNodes($idx, $nn, apoc.convert.fromJsonList($embedded_query)
        ) YIELD node, score
        RETURN node.dct__title, node.uri, score
    """
    return tx.run(cypher, idx=idx, nn=nn, embedded_query=embedded_query).values()


def rag(client: InferenceClient, titles: str, n_requests=1):
    """
    Pass a query to a HF Inference Client to generate a summary of book titles
    :param client: huggingface_hub.InferenceClient
    :param titles: str
    :param n_requests: int
    :return:
    """
    responses = []
    for i in range(n_requests):
        try:
            rag_content = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.3",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        Collectively summarise these book titles (separated by ';') in 40 words or less:
                        {titles}
                        """
                    }
                ],
                max_tokens=60,
            )
            responses.append(rag_content)

        except requests.exceptions.ConnectionError as e:
            print(f"Connection Error: {e}")
            print("Please ensure you are connected to the internet!")

    return responses


def create_umap_fig(df: pd.DataFrame) -> plotly.graph_objects.Figure:
    """
    Create a plotly 3d scatter plot from UMAP transformed data
    :param df: pd.DataFrame
    :return: plotly.graph_objects.Figure
    """
    umap_fig = px.scatter_3d(
        df, x="Feature 1", y="Feature 2", z="Feature 3", color="LCSH Class",
        hover_data=["LCSH Class Text", "Title", "Feature 1", "Feature 2", "Feature 3"], width=1000, height=1000
    )

    umap_fig.update_traces(marker_size=3, selector=dict(type='scatter3d'))
    umap_fig.update_layout(showlegend=False)
    return umap_fig


def load_umap_data(model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data required to run UMAP and visualise results
    :param model: str
    :return: tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    topic_path = os.path.join(PROCESSED_DATA_DIR, "most_common_topic.csv")
    titles_path = os.path.join(INTERIM_DATA_DIR, "titles.csv")
    acronym = model_params[model]["acronym"]
    embedded_path = os.path.join(PROCESSED_DATA_DIR, f"titles_lcsh_embedded_{acronym}.p")

    most_common_topic_df = pd.read_csv(topic_path, index_col=0, encoding="utf-8-sig")

    titles_df = pd.read_csv(titles_path, encoding="utf8", index_col=0)
    titles_df["title_clean"] = titles_df["title"].apply(lambda x: x[2:-2])

    embedded_df = pickle.load(open(embedded_path, "rb"))
    embedded_df[f"embedding_{acronym}"] = embedded_df[f"embedding_{acronym}"].apply(lambda x: np.array(x))

    return most_common_topic_df, titles_df, embedded_df


def create_umap(model: str) -> tuple[pd.DataFrame, umap.UMAP]:
    """
    Carry out the UMAP algorithm on a dataset of embedded BNB titles
    Return a plot ready dataframe and a fitted UMAP model
    :param model: str
    :return: tuple[pd.DataFrame, umap.UMAP]
    """
    most_common_topic_df, titles_df, embedded_df = load_umap_data(model)
    acronym = model_params[model]["acronym"]

    combined_df = titles_df.drop(columns="title").join(embedded_df).dropna()
    dd_df = combined_df.drop_duplicates(subset="title_clean").copy()
    topic_df = dd_df.join(most_common_topic_df["lcsh_t1"], on="uri").dropna()
    model_arr = np.vstack(topic_df[f"embedding_{acronym}"])

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3)
    reduced_data = reducer.fit_transform(model_arr)

    umap_df = topic_df.join(
        pd.DataFrame(data=reduced_data, columns=["f1", "f2", "f3"], index=topic_df.index)
    ).drop(
        columns=[f"embedding_{acronym}"]
    ).rename(
        columns={"title_clean": "Title", "lcsh_t1": "LCSH Class Text",
                 "f1": "Feature 1", "f2": "Feature 2", "f3": "Feature 3"}
    )

    classes = np.mod(np.arange(len(umap_df["LCSH Class Text"].unique())), 50) / 50
    class_map = pd.Series(
        data=colormaps["viridis"](classes)[:, :-1].tolist(),
        index=umap_df["LCSH Class Text"].unique()
    ).apply(lambda x: tuple(x))

    umap_df["LCSH Class"] = umap_df["LCSH Class Text"].map(class_map)

    return umap_df, reducer


def umap_transform_user_search(model: str, reducer: umap.UMAP):
    """
    Apply a UMAP transform generated by create_umap to user search term(s)
    The embedded term(s) are returned ready to be plotted
    :param model: str
    :param reducer: umap.UMAP
    :return: pd.DataFrame
    """
    acronym = model_params[model]["acronym"]
    user_search_path = os.path.join(INTERIM_DATA_DIR, "top_searches_to_date_2502.csv")
    top_searches_df = pd.read_csv(user_search_path, encoding="utf-8-sig", sep=",")

    informational_df = top_searches_df.query("Category == 'Informational'").copy()
    informational_df[f"embedding_{acronym}"] = informational_df["Search String"].progress_apply(
        lambda x: model_params[model]["model"].encode(x)
    )

    informational_embed = reducer.transform(np.vstack(informational_df[f"embedding_{acronym}"].values))

    informational_umap = informational_df.join(
        pd.DataFrame(data=informational_embed, columns=["Feature 1", "Feature 2", "Feature 3"],
                     index=informational_df.index)
    ).drop(columns=f"embedding_{acronym}")

    return informational_umap
