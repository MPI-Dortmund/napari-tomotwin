from typing import List, Tuple, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

global pbar


def get_non_numeric_column_titles(df: pd.DataFrame):

    return [l for l in df.columns if l.isnumeric() == False]


def _get_medoid_embedding(
    embeddings: pd.DataFrame, max_embeddings: int = 50000
) -> Tuple[pd.DataFrame, npt.ArrayLike]:
    """
    Calculates the medoid based of subset of the embeddings.
    """
    if len(embeddings) > max_embeddings:
        # For samples more than 50k it's way to slow and memory hungry.
        print(
            f"Your cluster size ({len(embeddings)}) is bigger then {max_embeddings}. Make a random sample to calculate medoid."
        )
        embeddings = embeddings.sample(max_embeddings, random_state=42)

    only_emb = embeddings.drop(
        columns=get_non_numeric_column_titles(embeddings), errors="ignore"
    ).astype(np.float32)
    distance_matrix = pairwise_distances(
        only_emb, metric="cosine", n_jobs=-1
    )  # its not the cosine similarity, rather a distance (its 0 in case of same embeddings)
    medoid_index = np.argmin(np.sum(distance_matrix, axis=0))
    medoid = only_emb.iloc[medoid_index, :]
    pos = embeddings.iloc[[medoid_index]][["X", "Y", "Z"]]
    return medoid, pos


def _make_targets(
    embeddings: pd.DataFrame,
    clusters: pd.DataFrame,
    avg_func: Callable[[pd.DataFrame], npt.ArrayLike],
    target_cluster: int = None,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], dict]:
    targets = []
    sub_embeddings = []
    target_names = []
    target_locations = {}
    for cluster in set(clusters):

        if cluster == 0:
            continue
        if target_cluster is not None and cluster != target_cluster:
            continue

        clmask = (clusters == cluster).to_numpy()

        cluster_embeddings = embeddings.loc[clmask, :]
        target, position = avg_func(cluster_embeddings)
        target_locations[cluster] = position
        sub_embeddings.append(embeddings.loc[clmask, :])
        target = target.to_frame().T
        targets.append(target)
        target_names.append(f"cluster_{cluster}")

    targets = pd.concat(targets, ignore_index=True)
    targets["filepath"] = target_names
    return targets, sub_embeddings, target_locations



