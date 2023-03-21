from magicgui import magic_factory, tqdm
import pathlib
import pandas as pd
import numpy as np
import os
from typing import List, Tuple


def _make_targets(embeddings: pd.DataFrame, clusters: pd.DataFrame) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    targets = []
    sub_embeddings = []
    target_names = []
    for cluster in set(clusters):
        if cluster == 0:
            continue
        target = embeddings.drop(columns=["X", "Y", "Z", "filepath"], errors="ignore").loc[clusters == cluster, :].astype(np.float32).mean(axis=0)
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
    output_folder={
        'label': "Output folder",
        'mode': 'd'
    }
)
def make_targets(
        label_layer: "napari.layers.Labels",
        embeddings_filepath: pathlib.Path,
        output_folder: pathlib.Path
):
    print("Read embeddings")
    embeddings = pd.read_pickle(embeddings_filepath)

    print("Read clusters")
    clusters = label_layer.features['MANUAL_CLUSTER_ID']

    assert len(embeddings) == len(clusters), "Cluster and embedding file are not compatible."

    print("Make targets")
    embeddings = embeddings.reset_index()

    targets, sub_embeddings = _make_targets(embeddings, clusters)

    print("Write targets")
    os.makedirs(output_folder, exist_ok="True")
    pth_ref = os.path.join(output_folder, "cluster_targets.temb")

    targets.to_pickle(pth_ref)

    print("Write custer embeddings")
    for emb_i, emb in enumerate(sub_embeddings):
        pth_emb = os.path.join(output_folder, f"embeddings_cluster_{emb_i}.temb")
        emb.to_pickle(pth_emb)

    print("Done")