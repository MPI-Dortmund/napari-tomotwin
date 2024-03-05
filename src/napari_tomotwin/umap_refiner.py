import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import typing
from numpy.typing import ArrayLike


def calculate_umap(
    embeddings: pd.DataFrame,
    transform_chunk_size: int = 400000,
    reducer: "cuml.UMAP" = None,
    ncomponents=2,
    neighbors: int = 200,
    metric: str = "euclidean",
) -> typing.Tuple[ArrayLike, "cuml.UMAP"]:

    try:
        # Import has to be happen here, as otherwise cuml is not working from a seperate process
        # See: https://github.com/explosion/spaCy/issues/5507

        import cuml
        import cudf
    except ImportError:
        print("cuml can't be loaded")
    print("Prepare data")
    all_data = embeddings.drop(
        ["filepath", "Z", "Y", "X"], axis=1, errors="ignore"
    )
    indicis = np.arange(start=0, stop=len(all_data), dtype=int)
    np.random.shuffle(indicis)
    umap_embeddings = np.zeros(shape=(len(all_data), ncomponents))
    num_chunks = max(1, int(math.ceil(len(indicis) / transform_chunk_size)))
    print(
        f"Transform complete dataset in {num_chunks} chunks with a chunksize of ~{int(len(indicis) / num_chunks)}"
    )

    for chunk in tqdm(np.array_split(indicis, num_chunks), desc="Transform"):
        c = all_data.iloc[chunk]
        if reducer is None:
            reducer = cuml.UMAP(
                n_neighbors=neighbors,
                n_components=ncomponents,
                n_epochs=None,  # means automatic selection
                min_dist=0.0,
                random_state=19,
                metric=metric,
            )
            print(f" Fit umap on {len(chunk)} samples")
            umap_embeddings[chunk] = reducer.fit_transform(c)
        else:
            print(f" Fit umap on {len(chunk)} samples")
            umap_embeddings[chunk] = reducer.transform(c)
    del cuml
    return umap_embeddings, reducer


def refine(
    clusters, embeddings: pd.DataFrame, target_cluster: int = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    embeddings = embeddings.drop(columns=["level_0", "index"], errors="ignore")
    if target_cluster is None:
        clmask = (clusters > 0).to_numpy().squeeze()
    else:
        clmask = (clusters == target_cluster).to_numpy().squeeze()

    input_embeddings = embeddings.iloc[clmask, :]
    umap_embedding_np, _ = calculate_umap(input_embeddings)

    df_embeddings = pd.DataFrame(umap_embedding_np)
    df_embeddings.reset_index(drop=True, inplace=True)
    df_embeddings.columns = [
        f"umap_{i}" for i in range(umap_embedding_np.shape[1])
    ]

    input_embeddings.reset_index(drop=True, inplace=True)
    df_embeddings = pd.concat(
        [input_embeddings[["X", "Y", "Z"]], df_embeddings], axis=1
    )
    df_embeddings.attrs["embeddings_attrs"] = embeddings.attrs

    return df_embeddings, input_embeddings
