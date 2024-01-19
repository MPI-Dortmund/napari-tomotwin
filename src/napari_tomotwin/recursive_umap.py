import math
import os
import os.path
import shutil
import tempfile
import typing

import numpy as np
import pandas as pd

from napari.utils import notifications
from napari_tomotwin.load_umap import LoadUmapTool
from numpy.typing import ArrayLike
from napari_clusters_plotter._plotter import PlotterWidget
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QApplication
from multiprocessing import Process, Queue, Manager

from qtpy.QtWidgets import (
    QFormLayout,
    QPushButton,
    QWidget,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QLabel

)
from tqdm import tqdm


class UmapRefiner:

    @staticmethod
    def calculate_umap(
            embeddings: pd.DataFrame,
            transform_chunk_size: int = 400000,
            reducer: "cuml.UMAP" = None,
            ncomponents=2,
            neighbors: int = 200,
            metric: str = "euclidean") -> typing.Tuple[ArrayLike, "cuml.UMAP"]:

        try:
            # Import has to be happen here, as otherwise cuml is not working from a seperate process
            import cuml
            import cudf
        except ImportError:
            print("cuml can't be loaded")
        print("Prepare data")
        all_data = embeddings.drop(['filepath', 'Z', 'Y', 'X'], axis=1, errors='ignore')
        indicis = np.arange(start=0, stop=len(all_data), dtype=int)
        np.random.shuffle(indicis)
        umap_embeddings = np.zeros(shape=(len(all_data),ncomponents))
        num_chunks = max(1, int(math.ceil(len(indicis) / transform_chunk_size)))
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
                c = all_data.iloc[chunk]
                umap_embeddings[chunk]= reducer.fit_transform(c)
            else:
                umap_embeddings[chunk] = reducer.transform(all_data.iloc[chunk])
        del cuml
        return umap_embeddings, reducer



    @staticmethod
    def refine(clusters, embeddings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        embeddings = embeddings.drop(columns=["level_0", "index"], errors="ignore")
        clmask = (clusters > 0).to_numpy().squeeze()
        input_embeddings = embeddings.iloc[clmask, :]
        umap_embedding_np, _ = UmapRefiner.calculate_umap(input_embeddings)


        df_embeddings = pd.DataFrame(umap_embedding_np)
        df_embeddings.reset_index(drop=True, inplace=True)
        df_embeddings.columns = [f"umap_{i}" for i in range(umap_embedding_np.shape[1])]


        input_embeddings.reset_index(drop=True, inplace=True)
        df_embeddings = pd.concat([input_embeddings[['X', 'Y', 'Z']], df_embeddings], axis=1)
        df_embeddings.attrs['embeddings_attrs'] = embeddings.attrs

        return df_embeddings, input_embeddings

    @staticmethod
    @thread_worker
    def refine_worker(clusters, embeddings: pd.DataFrame):

        return UmapRefiner.refine(clusters, embeddings)

    @staticmethod
    def refine_worker_parallel(q: Queue, clusters, embeddings: pd.DataFrame):
        umap_embeddings, used_embeddings = UmapRefiner.refine(clusters, embeddings)
        q.put((umap_embeddings, used_embeddings))








class UmapRefinerQt(QWidget):


    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()

        self.viewer = napari_viewer
        layout = QFormLayout()

        app = QApplication.instance()
        app.lastWindowClosed.connect(self.on_close_callback)  # this line is connection to signal

        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.setLayout(layout)

        self._run_btn = QPushButton("Recalculate UMAP for selected clusters", self)
        self._run_btn.clicked.connect(self._on_refine_click)
        self.layout().addRow("",self._run_btn)

        self.progressBar = QProgressBar(self)
        self.pbar_label = QLabel("")
        self.progressBar.setRange(0, 0)
        self.layout().addRow(self.pbar_label,self.progressBar)



        self.plotter_widget: PlotterWidget
        self.umap_tool: LoadUmapTool
        self.tmp_dir_path: str
        self.setMaximumHeight(100)


    def cleanup(self):
        try:
            shutil.rmtree(self.tmp_dir_path)
        except AttributeError:
            # Means that the there was no recalculated UMAP
            pass
    def on_close_callback(self):
        self.cleanup()

    def set_plotter_widget(self, widget: PlotterWidget):
        self.plotter_widget = widget

    def set_umap_tool(self, tool: LoadUmapTool):
        self.umap_tool = tool


    def _on_refine_click(self):
       self.reestimate_umap()

    @staticmethod
    def random_filename() -> str:
        return next(tempfile._get_candidate_names())

    def show_umap(self, res):
        (umap_embeddings, used_embeddings) = res
        self.tmp_dir_path = tempfile.mkdtemp()
        tmp_embed_pth = os.path.join(self.tmp_dir_path, UmapRefinerQt.random_filename())
        used_embeddings.to_pickle(tmp_embed_pth)
        umap_embeddings.attrs['embeddings_path'] = tmp_embed_pth
        tmp_umap_pth = os.path.join(self.tmp_dir_path, UmapRefinerQt.random_filename())
        umap_embeddings.to_pickle(tmp_umap_pth)

        # Visualizse in
        self.umap_tool.set_new_label_layer_name("UMAP Refined")
        worker = self.umap_tool.start_umap_worker(tmp_umap_pth)
        worker.start()

    def reestimate_umap(self):


        try:
            print("Read clusters")
            clusters = self.plotter_widget.layer_select.value.features['MANUAL_CLUSTER_ID']
            if not np.any(clusters>0):
                raise KeyError
        except KeyError:
            notifications.show_info(f"Not cluster selected. Can't refine.")
            return
        print("Read embeddings")

        def get_embedding_path(pth: str) -> str:
            '''
            Checks if the embedding path exists. If it does not exist, it opens a file selection dialogue. Otherwise it returns the path.
            '''
            if not os.path.exists(pth):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Can't open embedding file")
                msg.setText("Can't open embedding file")
                msg.setInformativeText("Embedding path in metadata data (see below) does not exist or can't be accesst. Please click OK and select the path to the embedding file.")
                msg.setDetailedText(pth)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                pth = QFileDialog.getOpenFileName(self, 'Open embedding file',
                                                    os.getcwd(),
                                                    "Embedding file (*.temb)")[0]

            return pth

        emb_pth = get_embedding_path(self.plotter_widget.layer_select.value.metadata['tomotwin']['embeddings_path'])

        if emb_pth == "":
            print("Not path selected.")
            return

        embeddings = pd.read_pickle(emb_pth)
        manager = Manager()
        q = manager.Queue() # Had to use a managed Queue isntead of "normal", as otherwise the process did not return.
        p = Process(target=UmapRefiner.refine_worker_parallel, args=(q, clusters,embeddings))
        p.start()
        print("WAIT")
        p.join()
        print("DONE")
        (umap_embeddings, used_embeddings) = q.get()
        print("Got it...")
        #umap_embeddings, used_embeddings = UmapRefiner.refine(clusters=clusters,embeddings=embeddings)

        self.tmp_dir_path = tempfile.mkdtemp()
        tmp_embed_pth = os.path.join(self.tmp_dir_path, UmapRefinerQt.random_filename())
        used_embeddings.to_pickle(tmp_embed_pth)
        umap_embeddings.attrs['embeddings_path'] = tmp_embed_pth
        tmp_umap_pth = os.path.join(self.tmp_dir_path, UmapRefinerQt.random_filename())
        umap_embeddings.to_pickle(tmp_umap_pth)

        # Visualizse in
        self.umap_tool.set_new_label_layer_name("UMAP Refined")
        worker = self.umap_tool.start_umap_worker(tmp_umap_pth)
        worker.start()





