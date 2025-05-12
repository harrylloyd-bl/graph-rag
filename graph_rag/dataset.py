import pickle
from pathlib import Path
import pandas as pd
from graph_rag.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, MODEL_PARAMS
from graph_rag.app import load_umap_data


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


def validate_save_dup_map(model_params):
    dup_maps = []
    for m in model_params:
        print(m)
        most_common_lcsh_df, titles_df, embedded_df = load_umap_data(m)
        combined_df = titles_df.join(embedded_df).dropna()
        duplicates_map = create_duplicate_map(combined_df)
        dup_maps.append(duplicates_map)

    if dup_maps[0] == dup_maps[1] and dup_maps[1] == dup_maps[2]:
        pickle.dump(dup_maps[0], open(f"../static/duplicate_map.p", "wb"))

    return None


def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    validate_save_dup_map(MODEL_PARAMS)


if __name__ == "__main__":
    main()
