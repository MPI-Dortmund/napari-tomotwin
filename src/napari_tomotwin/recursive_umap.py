import os
import os.path
import pathlib
import shutil
import tempfile
import typing

import numpy as np
import pandas as pd
from magicgui.widgets import create_widget
from napari.layers import Labels
from napari.utils import notifications
from napari_tomotwin.load_umap import LoadUmapTool
from numpy.typing import ArrayLike
from qtpy.QtWidgets import (
    QFormLayout,
    QPushButton,
    QWidget,
    QLabel
)
from tqdm import tqdm

try:
    import cuml
except ImportError:
    print("cuml can't be loaded")


class UmapRefiner:

    @staticmethod
    def calculate_umap(
            embeddings: pd.DataFrame,
            transform_chunk_size: int = 400000,
            reducer: "cuml.UMAP" = None,
            ncomponents=2,
            neighbors: int = 200,
            metric: str = "euclidean") -> typing.Tuple[ArrayLike, "cuml.UMAP"]:

        print("Prepare data")
        all_data = embeddings.drop(['filepath', 'Z', 'Y', 'X'], axis=1, errors='ignore')
        indicis = np.arange(start=0, stop=len(all_data), dtype=int)
        np.random.shuffle(indicis)
        umap_embeddings = np.zeros(shape=(len(all_data),ncomponents))
        num_chunks = max(1, int(len(indicis) / transform_chunk_size))
        print(
            f"Transform complete dataset in {num_chunks} chunks with a chunksize of ~{int(len(indicis) / num_chunks)}")

        for chunk in tqdm(np.array_split(indicis, num_chunks), desc="Transform"):
            if reducer is None:
                reducer = cuml.UMAP(
                    n_neighbors=neighbors,
                    n_components=ncomponents,
                    n_epochs=None,  # means automatic selection
                    min_dist=0.0,
                    random_state=19,
                    metric=metric
                )
                print(f" Fit umap on {len(chunk)} samples")
                umap_embeddings[chunk]= reducer.fit_transform(all_data.iloc[chunk])
            else:
                umap_embeddings[chunk] = reducer.transform(all_data.iloc[chunk])

        return umap_embeddings, reducer

    @staticmethod
    def refine(clusters, embeddings: pd.DataFrame):
        embeddings = embeddings.drop(columns=["level_0", "index"], errors="ignore")
        clmask = (clusters > 0).to_numpy()
        cluster_embeddings = embeddings.loc[clmask, :]
        embedding, _ = UmapRefiner.calculate_umap(cluster_embeddings)
        return embedding, cluster_embeddings






class UmapRefinerQt(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()

        self.viewer = napari_viewer
        layout = QFormLayout()

        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.setLayout(layout)
        self._run_btn = QPushButton("Refine", self)
        self._run_btn.clicked.connect(self._on_refine_click)

        self.layer_select = create_widget(annotation=Labels, label="layer")
        self.select_path = create_widget(annotation=pathlib.Path, label="EmbeddingsPath")
        self.layer_select.native.currentIndexChanged.connect(self._on_layer_changed)
        self.select_path_label = QLabel("Embeddings path")
        self.layout().addRow("Label layer", self.layer_select.native)
        self.layout().addRow(self.select_path_label, self.select_path.native)
        self.layout().addRow("",self._run_btn)

        self.update_embeddings_file_selection()


    def _on_refine_click(self):
        self.reestimate_umap()

    def _on_layer_changed(self):
        self.update_embeddings_file_selection()

    def reestimate_umap(self):
        print("Read clusters")
        clusters = self.layer_select.value.features['MANUAL_CLUSTER_ID']
        print("Read embeddings")
        embeddings = pd.read_pickle(self.select_path.value)
        umap_embeddings, used_embeddings = UmapRefiner.refine(clusters=clusters,embeddings=embeddings)
        df_embeddings = pd.DataFrame(umap_embeddings)
        df_embeddings.reset_index(drop=True, inplace=True)
        used_embeddings.reset_index(drop=True, inplace=True)

        df_embeddings.columns = [f"umap_{i}" for i in range(umap_embeddings.shape[1])]

        df_embeddings = pd.concat([used_embeddings[['X', 'Y', 'Z']], df_embeddings], axis=1)
        df_embeddings.attrs['embeddings_attrs'] = embeddings.attrs
        df_embeddings.attrs['embeddings_path'] = None
        tmpdirname = tempfile.mkdtemp()
        tmp_umap_pth = os.path.join(tmpdirname, "temp_umap.tumap")
        df_embeddings.to_pickle(tmp_umap_pth)
        utool = LoadUmapTool()
        worker = utool.start_umap_worker(tmp_umap_pth)
        worker.returned.connect(lambda: shutil.rmtree(tmpdirname))
        worker.start()

    def update_embeddings_file_selection(self):

        def make_visible(visible: bool):
            self.select_path.visible = False
            self.select_path_label.setHidden(~visible)

        try:
            epth = self.layer_select.value.metadata['tomotwin']['embeddings_path']
            if os.path.exists(epth):
                self.select_path.value = self.layer_select.value.metadata['tomotwin']['embeddings_path']
                make_visible(False)
            else:
                notifications.show_info(f"Embeddings path in metadata ({epth}) does not exist. Please set it manually.")
                make_visible(True)
        except :
            notifications.show_info("Can't find embeddings path in metadata. Please set it manually.")
            make_visible(True)

