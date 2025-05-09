import streamlit as st
st.title("High Friction Graph Database Semantic Search")

with st.spinner("Loading packages"):
    import os
    import pickle
    import platform

    import dotenv
    from neo4j import GraphDatabase
    import pandas as pd
    from tqdm import tqdm

    from graph_rag import app_text as text
    from graph_rag import app
    from graph_rag.config import model_params, HOLDING_QUERY

    tqdm.pandas()

VISUALISE = True

if platform.system() == "Linux":  # community cloud runs linux
    # Secrets stored in Streamlit in this case
    LOCAL = False
elif platform.system() == "Windows":
    LOCAL = True
    project_dir = os.path.join(os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)
else:
    LOCAL = True

if LOCAL:
    uri = os.getenv("NEO4J_DESKTOP_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_DESKTOP_PASSWORD")
else:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")

with st.spinner("Connecting to Graph DB"):
    driver = GraphDatabase.driver(uri=uri, auth=(user, pwd), database="neo4j")

    try:
        driver.verify_connectivity()
        NEO4J_CONNECTION = True
        st.success("Connected to Neo4j DB")
    except:
        NEO4J_CONNECTION = False
        if LOCAL:
            st.warning("Failed to connect to Neo4j DB. Using pre-populated query results instead. Please check Neo4j connection paramaters and try again")
        else:
            st.warning("Database access isn't currently available remotely. There are several pre-populated search queries to choose from instead!")

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

if NEO4J_CONNECTION:
    q = st.text_input(
        "Search box",
        value=HOLDING_QUERY,
        help="The query will be embedded and compared to the embedded subset of the BNB",
        label_visibility="hidden"
    )

    if q != HOLDING_QUERY:

        with driver.session() as session:
            limit = 5
            kw_result = session.execute_read(app.kw_search, query=q, limit=limit)
            kw_res = ["" for x in range(5)]
            for i, x in enumerate(kw_result):
                kw_res[i] = x[0]

            results_df = pd.DataFrame(
                data={"Keyword Search": kw_res},
                index=pd.RangeIndex(limit)
            )

            for m in model_params.keys():
                embedder = model_params[m]["model"]
                idx = "titleLCSH" + model_params[m]["acronym"].capitalize()
                semantic_result = session.execute_read(app.semantic_search, query=q, model=embedder, idx=idx, nn=limit)
                semantic_res = [(x[0][0], x[1]) for x in semantic_result]
                results_df[m] = [x[0] for x in semantic_res]
                results_df[m + "_uri"] = [x[1] for x in semantic_res]
            st.success("Search complete")

    try:
        st.table(results_df[["Keyword Search", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"]])
        # dl = st.button("Download results df")
        # filename = st.text_input("Write file name")
        # if dl:
        #     results_df.to_csv(f"static\\{filename}.csv", encoding="utf-8-sig")
    except NameError:  # app init and results_df hasn't been created yet
        pass
    except:
        st.write("Oops that search term failed, please try another.")

elif not NEO4J_CONNECTION:
    st.write("The database connection is unavailable, but you can use these four example searches to compare results and for visualisations later")
    tab_shk_kw, tab_shk, tab_eic, tab_quran = st.tabs(["Shakespeare KW", "Shakespeare", "East India Company", "Quran"])

    tab_display_cols = ["Keyword Search", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"]

    with tab_shk_kw:
        st.write("Query: Shakespeare")
        st.write("This query searches 'Shakespeare' as a single key word, compared to the natural language phrase in the next tab")
        shk_kw_q = "Shakespeare"
        shk_kw_df = pd.read_csv("static\\shakespeare_kw.csv", encoding="utf-8-sig", index_col=0)
        st.dataframe(shk_kw_df[tab_display_cols], hide_index=True)

    with tab_shk:
        st.write("Query: What kind of works did William Shakespeare write?")
        shk_q = "What kind of works did William Shakespeare write?"
        shk_df = pd.read_csv("static\\shakespeare.csv", encoding="utf-8-sig", index_col=0)
        st.dataframe(shk_df[tab_display_cols], hide_index=True)

    with tab_eic:
        st.write("Query: How did the East India Company rise to power?")
        eic_q = "How did the East India Company rise to power?"
        eic_df = pd.read_csv("static\\east_india_company.csv", encoding="utf-8-sig", index_col=0)
        st.dataframe(eic_df[tab_display_cols], hide_index=True)

    with tab_quran:
        st.write("Query: Different versions of the quran")
        quran_q = "Different versions of the quran"
        quran_df = pd.read_csv("static\\quran.csv", encoding="utf-8-sig", index_col=0)
        st.dataframe(quran_df[tab_display_cols], hide_index=True)

    query_select = st.pills(
        label="Choose an offline query to work with:",
        options=["Shakespeare KW", "Shakespeare", "East India Company", "Quran"],
        default="Shakespeare KW"
    )

    result_df_map = {"Shakespeare KW": shk_kw_df, "Shakespeare": shk_df, "East India Company": eic_df, "Quran": quran_df}
    query_map = {"Shakespeare KW": shk_kw_q, "Shakespeare": shk_q, "East India Company": eic_q, "Quran": quran_q}
    results_df = result_df_map[query_select]
    q = query_map[query_select]


st.markdown("## Retrieval Augmented Generation")

st.markdown(text.rag_text())

with st.form(key="rag_form"):
    vis_model = st.radio(
        label="Choose the embedding model output you'd like to summarise with Mistral-7B.",
        options=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"]
    )

    rag_button = st.empty()
    rag_response = st.empty()

    run_rag = rag_button.form_submit_button(
        label="Run RAG",
        on_click=app.rag(query=q, titles=results_df[vis_model], n_requests=1, out_container=rag_response)
    )

st.markdown("## Ethics")
st.markdown(text.ethics_text())

st.markdown("## Visualising Results")
vis_intro, how_to_read_text = text.visualise_text()
st.write(vis_intro)

with st.expander(label="How to read the visualisation"):
    st.write(how_to_read_text)

if not VISUALISE:
    st.write("Visualisations toggled off")
    st.stop()

st.write(text.visualise_select())
vis_select = st.radio(
    label="Do you want to visualise your query from the search box above (or the pre-made query if offline), or searches of the interim catalogue generated by BL users?",
    options=["BL user searches", "My search (or offline search)"]
)

if vis_select == "My search" and q != HOLDING_QUERY:
    vis_model = st.pills(
        label="Choose the embedding model search results you'd like to visualise.",
        options=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"]
    )

# Requires storage for the embedded files so that they can be umap-ed
if LOCAL:
    # subject headings for plotting
    with st.spinner("Running UMAP"):
        umap_df, reducer, dup_map = app.create_umap(vis_model)
    umap_fig = app.create_umap_fig(umap_df)

    if vis_select == "My search" and q != HOLDING_QUERY:

        with st.spinner("Adding your search to figure"):
            uris = pd.Index(results_df[vis_model + "_uri"].map(dup_map).values)
            hovertext_df = umap_df.loc[uris, "Title"]

            x, y, z = umap_df.loc[uris, ["Feature 1", "Feature 2", "Feature 3"]].values.T
            umap_fig.add_scatter3d(x=x, y=y, z=z, hoverinfo="text", hovertext=hovertext_df, showlegend=False,
                                   mode="markers", marker={"size": 8, "symbol": "cross", "color": "black"})

            embedded_search = model_params[vis_model]["model"].encode(q).reshape(1, model_params[vis_model]["dims"])

            x, y, z = reducer.transform(embedded_search).T
            umap_fig.add_scatter3d(x=x, y=y, z=z, hoverinfo="text", hovertext=f"Search term: {q}", showlegend=False,
                                   mode="markers", marker={"size": 12, "symbol": "circle-open", "color": "red"})

    if vis_select == "BL user searches":
        # user queries
        with st.spinner("Adding BL user searches to figure"):
            informational_umap = app.umap_transform_user_search(vis_model, reducer, use_live_data=False)

            hovertext_df = informational_umap.iloc[:5].loc[:, :"Results"].apply(
                lambda x: f"Search: {x['Search String']}<br>Total searches: {x['Searches']}<br>KW Hits {x['Results']}",
                axis=1)

            x, y, z = informational_umap.iloc[:5][["Feature 1", "Feature 2", "Feature 3"]].values.T
            umap_fig.add_scatter3d(x=x, y=y, z=z, hoverinfo="text", hovertext=hovertext_df, showlegend=False,
                                   mode="markers", marker={"size": 10, "symbol": "cross", "color": "black"})

    with st.spinner("Creating UMAP scatter chart"):
        st.plotly_chart(umap_fig)

if not LOCAL:
        st.write(text.remote_warning())
        umap_df = pickle.load(open(f"static/{model_params[vis_model]['acronym']}_umap.p", "rb"))
        umap_fig = app.create_umap_fig(umap_df)

        hovertext_df = umap_df.loc[results_df[vis_model + "_uri"].values, "Title"]
        x, y, z = umap_df.loc[results_df[vis_model + "_uri"].values, ["Feature 1", "Feature 2", "Feature 3"]].values.T
        umap_fig.add_scatter3d(x=x, y=y, z=z, hoverinfo="text", hovertext=hovertext_df, showlegend=False,
                               mode="markers", marker={"size": 8, "symbol": "cross", "color": "black"})

        with st.spinner("Creating UMAP scatter chart"):
            st.plotly_chart(umap_fig)

