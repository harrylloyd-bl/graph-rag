import pickle

from matplotlib import colormaps
import numpy as np
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
import streamlit as st
from tqdm import tqdm
import umap

from graph_rag import app_text as text

tqdm.pandas()

st.title("High Friction Graph Database Semantic Search")

st.markdown(text.title_text())

st.markdown("## The British National Bibliography")

st.markdown("## Embedding")

st.markdown("## Search")
model = st.radio(
    label="Which model do you want to embed the query with?",
    options=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"]
)

q = st.text_input("What do you want to search for today?", help="The query will be embedded and compared a section of the BNB")

st.markdown("## Retrieval Augmented Generation")

st.markdown("## Visualising Results")

# subject headings
st.write("Loading subject headings")
lcsh_df_v0 = pd.read_csv("data\\interim\\lcsh_topics.csv", encoding="utf8", names=["uri", "lcsh_topic"], skiprows=1, converters={"lcsh_topic": lambda x: x[2:-2]})
lcsh_df = lcsh_df_v0.copy()
lcsh_df["lcsh_t1"] = lcsh_df["lcsh_topic"].str.split("--").apply(lambda x: x[0])
lcsh_df = lcsh_df.join(lcsh_df["lcsh_t1"].value_counts(), on="lcsh_t1")
lcsh_df.set_index(["uri"], inplace=True)
most_common_topic_df = lcsh_df.groupby(level=0).max()

# embedded content
st.write("Loading embedded content and UMAPing")
load_model = "aml62"
embedded_df = pickle.load(open(f"data\\processed\\titles_embedded_{load_model}.p", "rb"))
embedded_df["embedding_aml62"] = embedded_df["embedding_aml62"].apply(lambda x: np.array(x))
dd_df = embedded_df.drop_duplicates(subset="title_clean").copy()
topic_df = dd_df.join(most_common_topic_df["lcsh_t1"], on="uri").dropna()
aml62_arr = np.vstack(topic_df["embedding_aml62"])

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3)
umap_aml62 = reducer.fit_transform(aml62_arr)

umap_df = topic_df.join(pd.DataFrame(data=umap_aml62, columns=["f1", "f2", "f3"], index=topic_df.index)).drop(columns=["uri", "embedding_aml62"])
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
model_aml62 = SentenceTransformer("all-MiniLM-L6-v2")
top_searches_df = pd.read_csv("data\\interim\\top_searches_to_date_2502.csv", encoding="utf-8-sig", sep=",")
informational_df = top_searches_df.query("Category == 'Informational'").copy()
informational_df["embedding_aml62"] = informational_df["Search String"].progress_apply(lambda x: model_aml62.encode(x))
informational_aml62 = reducer.transform(np.vstack(informational_df["embedding_aml62"].values))
informational_umap = informational_df.join(pd.DataFrame(data=informational_aml62, columns=["Feature 1", "Feature 2", "Feature 3"], index=informational_df.index)).drop(columns="embedding_aml62")
hovertext_df = informational_umap.loc[2:3, :"Results"].apply(lambda x: f"Search: {x['Search String']}<br>Total searches: {x['Searches']}<br>KW Hits {x['Results']}", axis=1)

st.write("Adding user searches to plot")
x,y,z = informational_umap.loc[2:3, ["Feature 1", "Feature 2", "Feature 3"]].values.T
umap_fig.add_scatter3d(x=x,y=y,z=z, hoverinfo="text", hovertext=hovertext_df, showlegend=False, mode="markers", marker={"size":10, "symbol": "cross", "color":"black"})

st.plotly_chart(umap_fig)

