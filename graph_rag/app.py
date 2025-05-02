import os
import pickle

import plotly.graph_objects
import streamlit
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
    most_common_lcsh_path = os.path.join(STATIC_DIR, "most_common_topic.csv")
    titles_path = os.path.join(STATIC_DIR, "titles.csv")
    acronym = model_params[model]["acronym"]
    embedded_path = os.path.join(STATIC_DIR, f"titles_lcsh_embedded_{acronym}.p")

    most_common_lcsh_df = pd.read_csv(most_common_lcsh_path, index_col=0, encoding="utf-8-sig")
    titles_df = pd.read_csv(titles_path, encoding="utf8", index_col=0)

    embedded_df = pickle.load(open(embedded_path, "rb"))
    embedded_df[f"embedding_{acronym}"] = embedded_df[f"embedding_{acronym}"].apply(lambda x: np.array(x))

    return most_common_lcsh_df, titles_df, embedded_df


def apply_create_map(grp, dup_map: dict):
    map_to = grp.index[0]
    for uri in grp.index[1:]:
        dup_map[uri] = map_to
    return None


def create_duplicate_map(df: pd.DataFrame) -> dict[str, str]:
    """
    UMAP is an expensive operation. We de-dup the dataframe of works on "Title" before UMAP-ing to minimise compute time
    Searches on the database potentially return "Title" duplicates that were dropped.
    So create a map from dropped duplicate URIs to equivalent URIs represented in the UMAP-ed so we can plot
    duplicates later.
    :param df:
    :return:
    """
    duplicates_df = df[df.duplicated(subset="title", keep=False)]
    duplicates_map = {}
    duplicates_df.groupby(by="title", as_index=False).apply(apply_create_map, dup_map=duplicates_map,
                                                            include_groups=False)
    uris_retained_in_umap = df.drop_duplicates(subset="title", keep="first").index
    for uri in uris_retained_in_umap:
        duplicates_map[uri] = uri

    return duplicates_map


@streamlit.cache_data(show_spinner=False)
def create_umap(model: str) -> tuple[pd.DataFrame, umap.UMAP, dict[str: str]]:
    """
    Carry out the UMAP algorithm on a dataset of embedded BNB titles
    Return a plot ready dataframe and a fitted UMAP model
    :param model: str
    :return: tuple[pd.DataFrame, umap.UMAP]
    """
    most_common_lcsh_df, titles_df, embedded_df = load_umap_data(model)
    acronym = model_params[model]["acronym"]

    combined_df = titles_df.join(embedded_df).dropna()
    duplicates_map = create_duplicate_map(combined_df)
    drop_dup_df = combined_df.drop_duplicates(subset="title", keep="first").copy()
    topic_df = drop_dup_df.join(most_common_lcsh_df["lcsh_t1"], on="uri").dropna()
    umap_arr = np.vstack(topic_df[f"embedding_{acronym}"])

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3)
    reduced_data = reducer.fit_transform(umap_arr)

    umap_df = topic_df.join(
        pd.DataFrame(data=reduced_data, columns=["f1", "f2", "f3"], index=topic_df.index)
    ).drop(
        columns=[f"embedding_{acronym}"]
    ).rename(
        columns={"title": "Title", "lcsh_t1": "LCSH Class Text",
                 "f1": "Feature 1", "f2": "Feature 2", "f3": "Feature 3"}
    )

    classes = np.mod(np.arange(len(umap_df["LCSH Class Text"].unique())), 50) / 50
    class_map = pd.Series(
        data=colormaps["viridis"](classes)[:, :-1].tolist(),
        index=umap_df["LCSH Class Text"].unique()
    ).apply(lambda x: tuple(x))

    umap_df["LCSH Class"] = umap_df["LCSH Class Text"].map(class_map)

    return umap_df, reducer, duplicates_map


def umap_transform_user_search(model: str, reducer: umap.UMAP, use_live_data: bool = False):
    """
    Apply a UMAP transform generated by create_umap to user search term(s)
    The embedded term(s) are returned ready to be plotted
    Use the five first hardcoded values unless specified
    :param model: str
    :param reducer: umap.UMAP
    :param use_live_data: bool
    :return: pd.DataFrame
    """

    acronym = model_params[model]["acronym"]

    if use_live_data:
        user_search_path = os.path.join(STATIC_DIR, "top_searches_to_date_2502.csv")
        top_searches_df = pd.read_csv(user_search_path, encoding="utf-8-sig", sep=",")
        informational_df = top_searches_df.query("Category == 'Informational'").copy()
    else:
        informational_df = pd.DataFrame(
            data={"Search String": ["commerce", "shakespeare", "longford castle", "maps", "industry"],
                  "Searches": [1587, 1327, 1284, 1171, 1141],
                  "Results": [1390, 18646, 1284, 342210, 1974]},
            index=pd.RangeIndex(5)
        )
    informational_df[f"embedding_{acronym}"] = informational_df["Search String"].progress_apply(
        lambda x: model_params[model]["model"].encode(x)
    )

    informational_embed = reducer.transform(np.vstack(informational_df[f"embedding_{acronym}"].values))

    informational_umap = informational_df.join(
        pd.DataFrame(data=informational_embed, columns=["Feature 1", "Feature 2", "Feature 3"],
                     index=informational_df.index)
    ).drop(columns=f"embedding_{acronym}")

    return informational_umap
