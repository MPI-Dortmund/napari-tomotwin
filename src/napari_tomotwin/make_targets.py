import os
import pathlib
from typing import List, Tuple, Literal, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from magicgui import magic_factory
from scipy.spatial.distance import cdist


def _get_medoid_embedding(embeddings: pd.DataFrame, max_embeddings: int = 50000) -> pd.DataFrame:
    """
    Calculates the medoid based of subset of the embeddings.
    """
    sample = embeddings
    if len(sample)>max_embeddings:
        # For samples more than 50k it's way to slow and memory hungry.
        sample = embeddings.sample(max_embeddings)
        print(f"Your cluster size ({len(embeddings)}) is bigger then {max_embeddings}. Make a random sample to calculate medoid.")
    distance_matrix=cdist(sample,sample,metric='cosine') # its not the cosine similarity, rather a distance (its 0 in case of same embeddings)
    medoid_index = np.argmin(np.sum(distance_matrix,axis=0))
    return embeddings.iloc[medoid_index,:]

def _get_avg_embedding(embeddings: pd.DataFrame) -> pd.DataFrame:
    target = embeddings.mean(axis=0)
    return target


def _make_targets(embeddings: pd.DataFrame, clusters: pd.DataFrame, avg_func: Callable[[pd.DataFrame], npt.ArrayLike]) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    targets = []
    sub_embeddings = []
    target_names = []
    for cluster in set(clusters):
        if cluster == 0:
            continue
        cluster_embeddings = embeddings.drop(columns=["X", "Y", "Z", "filepath"], errors="ignore").loc[clusters == cluster, :].astype(np.float32)
        target = avg_func(cluster_embeddings)
        sub_embeddings.append(embeddings.loc[clusters == cluster, :])
        target = target.to_frame().T
        targets.append(target)
        target_names.append(f"cluster_{cluster}")

    targets = pd.concat(targets, ignore_index=True)
    targets["filepath"] = target_names
    return targets, sub_embeddings

@magic_factory(
    call_button="Save",
    label_layer={'label': 'TomoTwin Label Mask:'},
    embeddings_filepath={'label': 'Path to embeddings file:',
              'filter': '*.temb'},
    average_method_name={'label': "Average method"},
    output_folder={
        'label': "Output folder",
        'mode': 'd'
    }
)
def make_targets(
        label_layer: "napari.layers.Labels",
        embeddings_filepath: pathlib.Path,
        output_folder: pathlib.Path,
        average_method_name: Literal["Average", "Medoid"] = "Medoid",
):
    print("Read embeddings")
    embeddings = pd.read_pickle(embeddings_filepath)

    print("Read clusters")
    clusters = label_layer.features['MANUAL_CLUSTER_ID']

    assert len(embeddings) == len(clusters), "Cluster and embedding file are not compatible."

    avg_method = _get_medoid_embedding
    if average_method_name == "Average":
        avg_method = _get_avg_embedding

    print("Make targets")
    embeddings = embeddings.reset_index()

    targets, sub_embeddings = _make_targets(embeddings, clusters,avg_func=avg_method)

    print("Write targets")
    os.makedirs(output_folder, exist_ok="True")
    pth_ref = os.path.join(output_folder, "cluster_targets.temb")

    targets.to_pickle(pth_ref)

    print("Write custer embeddings")
    for emb_i, emb in enumerate(sub_embeddings):
        pth_emb = os.path.join(output_folder, f"embeddings_cluster_{emb_i}.temb")
        emb.to_pickle(pth_emb)

    print("Done")