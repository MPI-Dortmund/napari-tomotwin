import os
import pathlib
from typing import List, Tuple, Literal, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from magicgui import magic_factory
from magicgui.tqdm import tqdm
from napari.qt.threading import thread_worker
from napari.utils import notifications
from sklearn.metrics.pairwise import pairwise_distances

global pbar

def get_non_numeric_column_titles(df: pd.DataFrame):

    return [l for l in df.columns if l.isnumeric() == False]

def _get_medoid_embedding(embeddings: pd.DataFrame, max_embeddings: int = 50000) -> Tuple[pd.DataFrame, npt.ArrayLike]:
    """
    Calculates the medoid based of subset of the embeddings.
    """
    if len(embeddings)>max_embeddings:
        # For samples more than 50k it's way to slow and memory hungry.
        print(f"Your cluster size ({len(embeddings)}) is bigger then {max_embeddings}. Make a random sample to calculate medoid.")
        embeddings = embeddings.sample(max_embeddings)

    only_emb = embeddings.drop(columns=get_non_numeric_column_titles(embeddings), errors="ignore").astype(np.float32)
    distance_matrix=pairwise_distances(only_emb,metric='cosine', n_jobs=-1) # its not the cosine similarity, rather a distance (its 0 in case of same embeddings)
    medoid_index = np.argmin(np.sum(distance_matrix,axis=0))
    medoid = only_emb.iloc[medoid_index,:]
    pos = embeddings.iloc[[medoid_index]][['X','Y','Z']]
    return medoid, pos

def _get_avg_embedding(embeddings: pd.DataFrame) -> Tuple[pd.DataFrame, npt.ArrayLike]:
    only_emb = embeddings.drop(columns=get_non_numeric_column_titles(embeddings), errors="ignore").astype(np.float32)
    target = only_emb.mean(axis=0)
    return target, np.array([])

def _make_targets(embeddings: pd.DataFrame, clusters: pd.DataFrame, avg_func: Callable[[pd.DataFrame], npt.ArrayLike]) -> Tuple[pd.DataFrame, List[pd.DataFrame], dict]:
    targets = []
    sub_embeddings = []
    target_names = []
    target_locations = {

    }
    for cluster in set(clusters):

        if cluster == 0:
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



def _run(clusters,
                  embeddings: pd.DataFrame,
                  output_folder: pathlib.Path,
                  average_method_name: Literal["Average", "Medoid"] = "Medoid",
):
    assert len(embeddings) == len(clusters), "Cluster and embedding file are not compatible. They have a different number of embeddings"

    avg_method = _get_medoid_embedding
    if average_method_name == "Average":
        avg_method = _get_avg_embedding



    print("Make targets")
    #embeddings = embeddings.reset_index()
    embeddings = embeddings.drop(columns=["level_0","index"], errors="ignore")

    targets, sub_embeddings, target_locations = _make_targets(embeddings, clusters, avg_func=avg_method)

    print("Write targets")
    os.makedirs(output_folder, exist_ok="True")
    pth_ref = os.path.join(output_folder, "cluster_targets.temb")

    targets.to_pickle(pth_ref)
    print(target_locations)
    for cluster_id in target_locations:
        df_loc = target_locations[cluster_id]
        print(df_loc)
        if df_loc is not None and len(df_loc) > 0:
            pth_loc = os.path.join(output_folder, f"cluster_{cluster_id}_medoid.coords")
            df_loc[["X", "Y", "Z"]].to_csv(pth_loc, sep=" ", header=None, index=None)

    print("Write custer embeddings")
    for emb_i, emb in enumerate(sub_embeddings):
        pth_emb = os.path.join(output_folder, f"embeddings_cluster_{emb_i}.temb")
        emb.to_pickle(pth_emb)

    print("Done")


@thread_worker
def _run_worker(embeddings_filepath, label_layer, output_folder: str, average_method_name: str):
    print("Read clusters")
    clusters = label_layer.features['MANUAL_CLUSTER_ID']

    print("Read embeddings")
    embeddings = pd.read_pickle(embeddings_filepath)
    _run(clusters, embeddings, output_folder, average_method_name)

def on_init(widget):

    def update():
        try:
            if os.path.exists(widget.label_layer.value.metadata['tomotwin']['embeddings_path']):
                widget.embeddings_filepath.value = widget.label_layer.value.metadata['tomotwin']['embeddings_path']
                widget.embeddings_filepath.visible = False
            else:
                notifications.show_info("Can't find embeddings path in metadata. Please set it manually.")
                widget.embeddings_filepath.visible = True
        except:
            notifications.show_info("Can't find embeddings path in metadata. Please set it manually.")
            widget.embeddings_filepath.visible = True
            pass
    @widget.label_layer.changed.connect
    def _on_image_path_changed():
        update()

    widget.max_height = 200
    update()



@magic_factory(
    call_button="Save",
    label_layer={'label': 'TomoTwin Label Mask:'},
    embeddings_filepath={'label': 'Path to embeddings file:',
              'filter': '*.temb'},
    widget_init=on_init,
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
    global pbar
    pbar = tqdm()
    pbar.progressbar.label = "Save targets"
    worker = _run_worker(embeddings_filepath, label_layer, output_folder, average_method_name) # create "worker" object
    worker.finished.connect(lambda: pbar.progressbar.hide())
    worker.start()

