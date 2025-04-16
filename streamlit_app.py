import os
import pickle
import platform

import dotenv
import requests.exceptions
from huggingface_hub import InferenceClient
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
VISUALISE = True
RAG = True

if platform.system() == "Linux":  # community cloud runs linux
    # Secrets stored in Streamlit in this case
    LOCAL = False
elif platform.system() == "Windows":
    LOCAL = True
    project_dir = os.path.join(os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

if LOCAL:
    uri = os.getenv("NEO4J_DESKTOP_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_DESKTOP_PASSWORD")
else:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")

st.title("High Friction Graph Database Semantic Search")
if GRAPH:
    driver = GraphDatabase.driver(uri=uri, auth=(user, pwd), database="neo4j")
    try:
        driver.verify_connectivity()
        st.success("Connected to Neo4j DB")
    except:
        st.error("Failed to connect to Neo4j DB. Please check Neo4j connection paramaters and try again")
        st.stop()

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
st.dataframe(pd.read_csv("static\\model_params.csv", index_col=0, encoding="utf-8-sig"))

st.markdown("## Search")

q = st.text_input("What do you want to search for today?", value="Enter search term here", help="The query will be embedded and compared a section of the BNB")

if GRAPH and q!= "Enter search term here":
    def basic_search(tx, query=None, limit=5):
        regex = f"(?i).*{query}.*"
        cypher = """
            MATCH (n:bibo__Book)
            WHERE n.dct__title[0] =~ $regex
            RETURN n.dct__title[0]
            LIMIT $limit
        """
        return tx.run(cypher, regex=regex, limit=limit).values()

    def semantic_search(tx, model: SentenceTransformer, query: str, idx: str, nn: int):
        embedded_query = str(model.encode(query).tolist())
        cypher = """
            CALL db.index.vector.queryNodes($idx, $nn, apoc.convert.fromJsonList($embedded_query)
            ) YIELD node, score
            RETURN node.dct__title, node.uri, score
        """
        return tx.run(cypher, idx=idx, nn=nn, embedded_query=embedded_query).values()

    with driver.session() as session:
        limit = 5
        basic_result = session.execute_read(basic_search, query=q, limit=limit)
        kw_res = ["" for x in range(5)]
        for i, x in enumerate(basic_result):
            kw_res[i] = x[0]

        results_df = pd.DataFrame(
            data={"Keyword Search": kw_res},
            index=pd.RangeIndex(limit)
        )

        for m in model_params.keys():
            embedder = model_params[m]["model"]
            idx = "titleLCSH" + model_params[m]["acronym"].capitalize()
            semantic_result = session.execute_read(semantic_search, query=q, model=embedder, idx=idx, nn=limit)
            semantic_res = [(x[0][0], x[1]) for x in semantic_result]
            results_df[m] = [x[0] for x in semantic_res]
            results_df[m + "_uri"] = [x[1] for x in semantic_res]
        st.success("Search complete")

        try:
            st.table(results_df[["Keyword Search", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"]])
        except:
            st.write("Oops that search term failed, please try another.")

st.markdown("## Retrieval Augmented Generation")

st.markdown(text.rag_text())

model = st.radio(
    label="Choose the embedding model output you'd like to summarise with Mistral-7B.",
    options=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"]
)

client = InferenceClient(provider="hf-inference", api_key=os.environ["HF_TOKEN"])
run_rag = st.button("Run RAG")
if run_rag and q != "Enter search term here":
    titles = "; ".join(results_df[model].values)
if RAG and run_rag:
    rag_content = None
    responses = []
    n_requests = 1
    with st.spinner(text="Get retrieval augmented generation content from HF API"):
        for i in range(n_requests):
            try:
                rag_content = client.chat.completions.create(
                    model="mistralai/Mistral-7B-Instruct-v0.3",
                    messages=[
                        # {
                        #     "role": "system",
                        #     "content": "you are a library assistant, answer any queries in 40 words or less"
                        # },
                        {
                            "role": "user",
                            "content": f"""
                            Collectively summarise these book titles in 40 words or less:
                            {titles}
                            """
                        }
                    ],
                    max_tokens=60,
                )
                responses.append(rag_content)
            except requests.exceptions.ConnectionError as e:
                st.error(f"Connection Error: {e}")
                st.write("Please ensure connected to the internet!")

    if responses:
        tabs = st.tabs([f"Response {x}" for x in range(1,n_requests + 1)])
        for r, t in zip(responses, tabs):
            t.write(f"LLM generated response to semantic search results:")
            t.markdown(f">{r.choices[0].message['content']}")

st.markdown("## Visualising Results")
if not VISUALISE:
    st.write("Visualisations toggled off")
if VISUALISE and not LOCAL:
    st.write("Due to storage constraints, the web version of this demo can't embed your searches at this time")
    st.write("We can discuss what the visualisation means, and how we might use it as a tool to understand issues with embedding.")

    umap_df = pickle.load(open(f"static\\{model_params[model]['acronym']}_umap.p", "rb"))
    umap_fig = px.scatter_3d(umap_df, x="Feature 1", y="Feature 2", z="Feature 3", color="LCSH Class", hover_data=["LCSH Class Text", "Title", "Feature 1", "Feature 2", "Feature 3"], width=1000, height=1000)
    umap_fig.update_traces(marker_size=3, selector=dict(type='scatter3d'))
    umap_fig.update_layout(showlegend=False)
    with st.spinner("Creating Plotly chart"):
        st.plotly_chart(umap_fig)

# Requires storage for the embedded files so that they can be umap-ed
if False:
    st.write("The visualisation process will take 1-2 minutes as the dimensionality reduction process is slow.")
    st.write(
        "You can choose to visualise the search term from above, or investigate some of the common search terms used in our live catalogue")
    vis_select = st.radio(
        label="Your search, or live catalogue searches from BL users?",
        options=["My search", "BL user searches"]
    )
    if vis_select == "My search":
        # subject headings
        st.write("Loading subject headings")
        most_common_topic_df = pd.read_csv("data\\processed\\most_common_topic.csv", index_col=0, encoding="utf-8-sig")

        # embedded content
        st.write("Loading embedded content and UMAPing")
        titles_df = pd.read_csv("data\\interim\\titles.csv", encoding="utf8", index_col=0)
        titles_df["title_clean"] = titles_df["title"].apply(lambda x: x[2:-2])
        st.write(f"Using model {model}")
        acronym = model_params[model]["acronym"]
        embedded_df = pickle.load(open(f"data\\processed\\titles_lcsh_embedded_{acronym}.p", "rb"))
        embedded_df[f"embedding_{acronym}"] = embedded_df[f"embedding_{acronym}"].apply(lambda x: np.array(x))
        combined_df = titles_df.drop(columns="title").join(embedded_df).dropna()
        dd_df = combined_df.drop_duplicates(subset="title_clean").copy()
        topic_df = dd_df.join(most_common_topic_df["lcsh_t1"], on="uri").dropna()
        model_arr = np.vstack(topic_df[f"embedding_{acronym}"])

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3)
        umap = reducer.fit_transform(model_arr)

        umap_df = topic_df.join(pd.DataFrame(data=umap, columns=["f1", "f2", "f3"], index=topic_df.index)).drop(
            columns=[f"embedding_{acronym}"])
        umap_df = umap_df.rename(
            columns={"title_clean": "Title", "lcsh_t1": "LCSH Class Text", "f1": "Feature 1", "f2": "Feature 2",
                     "f3": "Feature 3"})
        classes = np.mod(np.arange(len(umap_df["LCSH Class Text"].unique())), 50) / 50
        class_map = pd.Series(colormaps["viridis"](classes)[:, :-1].tolist(),
                              index=umap_df["LCSH Class Text"].unique()).apply(lambda x: tuple(x))
        umap_df["LCSH Class"] = umap_df["LCSH Class Text"].map(class_map)

        umap_df = pickle.load(f"static\\{model}_umap.p")
        st.write("Plotly")
        umap_fig = px.scatter_3d(umap_df, x="Feature 1", y="Feature 2", z="Feature 3", color="LCSH Class",
                                 hover_data=["LCSH Class Text", "Title", "Feature 1", "Feature 2", "Feature 3"],
                                 width=1000, height=1000)
        umap_fig.update_traces(marker_size=3, selector=dict(type='scatter3d'))
        umap_fig.update_layout(showlegend=False)

        hovertext_df = umap_df.loc[results_df[model + "_uri"], :"Results"].apply(
            lambda x: f"Search: {x['Search String']}<br>Total searches: {x['Searches']}<br>KW Hits {x['Results']}",
            axis=1)

        x, y, z = umap_df.loc[results_df[model + "_uri"].values, ["Feature 1", "Feature 2", "Feature 3"]].values.T
        umap_fig.add_scatter3d(x=x, y=y, z=z, hoverinfo="text", hovertext=hovertext_df, showlegend=False,
                               mode="markers", marker={"size": 8, "symbol": "cross", "color": "black"})

        embedded_search = model_params[model]["model"].encode(q)
        x, y, z = reducer.transform(embedded_search)
        umap_fig.add_scatter3d(x=x, y=y, z=z, hoverinfo="text", hovertext=f"Search term: {q}", showlegend=False,
                               mode="markers", marker={"size": 12, "symbol": "cross", "color": "gold"})

        st.plotly_chart(umap_fig)

    if vis_select == "BL user searches":
        # subject headings
        # st.write("Loading subject headings")
        most_common_topic_df = pd.read_csv("data\\processed\\most_common_topic.csv", index_col=0, encoding="utf-8-sig")

        # embedded content
        # st.write("Loading embedded content and UMAPing")
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

        umap_df = topic_df.join(pd.DataFrame(data=umap, columns=["f1", "f2", "f3"], index=topic_df.index)).drop(
            columns=[f"embedding_{acronym}"])
        umap_df = umap_df.rename(
            columns={"title_clean": "Title", "lcsh_t1": "LCSH Class Text", "f1": "Feature 1", "f2": "Feature 2",
                     "f3": "Feature 3"})
        classes = np.mod(np.arange(len(umap_df["LCSH Class Text"].unique())), 50) / 50
        class_map = pd.Series(colormaps["viridis"](classes)[:, :-1].tolist(),
                              index=umap_df["LCSH Class Text"].unique()).apply(lambda x: tuple(x))
        umap_df["LCSH Class"] = umap_df["LCSH Class Text"].map(class_map)

        st.write("Plotly")
        umap_fig = px.scatter_3d(umap_df, x="Feature 1", y="Feature 2", z="Feature 3", color="LCSH Class",
                                 hover_data=["LCSH Class Text", "Title", "Feature 1", "Feature 2", "Feature 3"],
                                 width=1000, height=1000)
        umap_fig.update_traces(marker_size=3, selector=dict(type='scatter3d'))
        umap_fig.update_layout(showlegend=False)

        # user queries
        st.write("Embedding user searches")
        top_searches_df = pd.read_csv("data\\interim\\top_searches_to_date_2502.csv", encoding="utf-8-sig", sep=",")
        informational_df = top_searches_df.query("Category == 'Informational'").copy()
        informational_df[f"embedding_{acronym}"] = informational_df["Search String"].progress_apply(
            lambda x: model_params[load_model]["model"].encode(x))
        informational_embed = reducer.transform(np.vstack(informational_df[f"embedding_{acronym}"].values))
        informational_umap = informational_df.join(
            pd.DataFrame(data=informational_embed, columns=["Feature 1", "Feature 2", "Feature 3"],
                         index=informational_df.index)).drop(columns=f"embedding_{acronym}")
        hovertext_df = informational_umap.loc[2:3, :"Results"].apply(
            lambda x: f"Search: {x['Search String']}<br>Total searches: {x['Searches']}<br>KW Hits {x['Results']}",
            axis=1)

        st.write("Adding user searches to plot")
        x, y, z = informational_umap.iloc[:5][["Feature 1", "Feature 2", "Feature 3"]].values.T
        umap_fig.add_scatter3d(x=x, y=y, z=z, hoverinfo="text", hovertext=hovertext_df, showlegend=False,
                               mode="markers", marker={"size": 10, "symbol": "cross", "color": "black"})

        st.plotly_chart(umap_fig)
