import os
import pickle
import platform

import dotenv
from matplotlib import colormaps
from neo4j import GraphDatabase
import numpy as np
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
import streamlit as st
from tqdm import tqdm
import umap

from graph_rag import app_text as text
from graph_rag.config import model_params

tqdm.pandas()

GRAPH = True
VISUALISE = False

if platform.system() == "Linux":  # community cloud runs linux
    # Secrets stored in Streamlit in this case
    LOCAL = False
elif platform.system() == "Windows":
    LOCAL = True
    project_dir = os.path.join(os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

st.title("High Friction Graph Database Semantic Search")

st.markdown(text.title_text())

st.markdown("## The British National Bibliography")

st.markdown(text.bnb_text())

st.markdown("## Embedding")
emb_i, emb_a1, emb_a2, emb_models = text.embedding_text()
st.markdown(emb_i)
with st.expander("Analogy One: RGB Colour"):
    st.markdown(emb_a1, unsafe_allow_html=True)

with st.expander("Analogy Two: XKCD - Features of Adulthood"):
    st.markdown(emb_a2, unsafe_allow_html=True)

st.markdown(emb_models)
st.dataframe(pd.read_csv("data/processed/model_params.csv", index_col=0, encoding="utf-8-sig"))

st.markdown("## Search")
model = st.radio(
    label="Which model do you want to embed the query with?",
    options=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"]
)

q = st.text_input("What do you want to search for today?", help="The query will be embedded and compared a section of the BNB")

if GRAPH:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri=uri, auth=(user, pwd), database="neo4j")
    try:
        driver.verify_connectivity()
        st.success("Connected to Neo4j DB")
    except:
        st.error("Failed to connect to Neo4j DB")

    def basic_search(tx, query=None):
        regex = f"(?i).*{query}.*"
        cypher = """
            MATCH (n:bibo__Book)
            WHERE n.dct__title[0] =~ $regex
            RETURN n.dct__title[0]
            LIMIT 5
        """
        return tx.run(cypher, regex=regex).values()

    with driver.session() as session:
        result = session.execute_read(basic_search, query=q)
        st.write([x[0] for x in result])

st.markdown("## Retrieval Augmented Generation")

st.markdown("## Visualising Results")
if VISUALISE:
    # subject headings
    st.write("Loading subject headings")
    most_common_topic_df = pd.read_csv("data\\processed\\most_common_topic.csv", index_col=0, encoding="utf-8-sig")

    # embedded content
    st.write("Loading embedded content and UMAPing")
    titles_df = pd.read_csv("data\\interim\\titles.csv", encoding="utf8", index_col=0)
    titles_df["title_clean"] = titles_df["title"].apply(lambda x: x[2:-2])
    load_model = model
    st.write(f"Using model {model}")
    acronym = model_params[load_model]["acronym"]
    embedded_df = pickle.load(open(f"data\\processed\\titles_lcsh_embedded_{acronym}.p", "rb"))
    embedded_df[f"embedding_{acronym}"] = embedded_df[f"embedding_{acronym}"].apply(lambda x: np.array(x))
    combined_df = titles_df.drop(columns="title").join(embedded_df).dropna()
    dd_df = combined_df.drop_duplicates(subset="title_clean").copy()
    topic_df = dd_df.join(most_common_topic_df["lcsh_t1"], on="uri").dropna()
    model_arr = np.vstack(topic_df[f"embedding_{acronym}"])

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3)
    umap = reducer.fit_transform(model_arr)

    umap_df = topic_df.join(pd.DataFrame(data=umap, columns=["f1", "f2", "f3"], index=topic_df.index)).drop(columns=[f"embedding_{acronym}"])
    umap_df = umap_df.rename(columns={"title_clean":"Title", "lcsh_t1": "LCSH Class Text", "f1": "Feature 1", "f2": "Feature 2", "f3": "Feature 3"})
    classes = np.mod(np.arange(len(umap_df["LCSH Class Text"].unique())), 50) / 50
    class_map = pd.Series(colormaps["viridis"](classes)[:,:-1].tolist(), index=umap_df["LCSH Class Text"].unique()).apply(lambda x: tuple(x))
    umap_df["LCSH Class"] = umap_df["LCSH Class Text"].map(class_map)

    st.write("Plotly")
    umap_fig = px.scatter_3d(umap_df, x="Feature 1", y="Feature 2", z="Feature 3", color="LCSH Class", hover_data=["LCSH Class Text", "Title", "Feature 1", "Feature 2", "Feature 3"], width=1000, height=1000)
    umap_fig.update_traces(marker_size=3, selector=dict(type='scatter3d'))
    umap_fig.update_layout(showlegend=False)

    # user queries
    st.write("Embedding user searches")
    top_searches_df = pd.read_csv("data\\interim\\top_searches_to_date_2502.csv", encoding="utf-8-sig", sep=",")
    informational_df = top_searches_df.query("Category == 'Informational'").copy()
    informational_df[f"embedding_{acronym}"] = informational_df["Search String"].progress_apply(lambda x: model_params[load_model]["model"].encode(x))
    informational_embed = reducer.transform(np.vstack(informational_df[f"embedding_{acronym}"].values))
    informational_umap = informational_df.join(pd.DataFrame(data=informational_embed, columns=["Feature 1", "Feature 2", "Feature 3"], index=informational_df.index)).drop(columns=f"embedding_{acronym}")
    hovertext_df = informational_umap.loc[2:3, :"Results"].apply(lambda x: f"Search: {x['Search String']}<br>Total searches: {x['Searches']}<br>KW Hits {x['Results']}", axis=1)

    st.write("Adding user searches to plot")
    x,y,z = informational_umap.loc[2:3, ["Feature 1", "Feature 2", "Feature 3"]].values.T
    umap_fig.add_scatter3d(x=x,y=y,z=z, hoverinfo="text", hovertext=hovertext_df, showlegend=False, mode="markers", marker={"size":10, "symbol": "cross", "color":"black"})

    st.plotly_chart(umap_fig)

