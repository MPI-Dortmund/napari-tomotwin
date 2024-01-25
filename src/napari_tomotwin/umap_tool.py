import math
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import os
import os.path
import shutil
import tempfile
import typing
from concurrent import futures


import numpy as np
import pandas as pd
from napari.utils import notifications
from napari_clusters_plotter._plotter import PlotterWidget
from napari_clusters_plotter._utilities import get_nice_colormap
from napari_tomotwin.make_targets import _make_targets, _get_medoid_embedding
from napari_tomotwin.load_umap import LoadUmapTool
from numpy.typing import ArrayLike
from qtpy.QtWidgets import QApplication


from qtpy.QtWidgets import (
    QFormLayout,
    QPushButton,
    QWidget,
    QFileDialog,
    QMessageBox,
    QLabel,
    QHBoxLayout,
    QLineEdit
)
from qtpy.QtCore import Signal
from tqdm import tqdm
from napari_tomotwin._qt.labeled_progress_bar import LabeledProgressBar


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
            # See: https://github.com/explosion/spaCy/issues/5507

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
            c = all_data.iloc[chunk]
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
                umap_embeddings[chunk]= reducer.fit_transform(c)
            else:
                print(f" Fit umap on {len(chunk)} samples")
                umap_embeddings[chunk] = reducer.transform(c)
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


class UmapToolQt(QWidget):

    refinement_done = Signal("PyQt_PyObject")
    target_calc_done = Signal("PyQt_PyObject")

    @staticmethod
    def check_if_gpu_is_available() -> bool:
        try:
            import cudf
            import cuml
        except:
            return False
        return True

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()

        self.viewer = napari_viewer


        #######
        # UI Setup
        ######
        layout = QFormLayout()
        app = QApplication.instance()
        app.lastWindowClosed.connect(self.on_close_callback)  # this line is connection to signal
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.setLayout(layout)

        ###
        # Select UMAP UI Elements
        ###
        umap_pth_layout = QHBoxLayout()
        self._select_umap_pth_btn = QPushButton("Select file", self)
        self._selected_umap_pth = QLineEdit()

        def select_file_clicked():
            pth = QFileDialog.getOpenFileName(self, 'Open UMAP file',
                                              os.getcwd(),
                                              "UMAP file (*.tumap)")[0]
            self._selected_umap_pth.setText(pth)


        self._select_umap_pth_btn.clicked.connect(select_file_clicked)
        self._load_umap_btn = QPushButton("Load")
        umap_pth_layout.addWidget(self._selected_umap_pth)
        umap_pth_layout.addWidget(self._select_umap_pth_btn)
        self.layout().addRow("Path to UMAP:", umap_pth_layout)
        self.layout().addRow("", self._load_umap_btn)
        self.plotter_widget: PlotterWidget = None
        self.plotter_widget_run_func = None
        self.plotter_Widget_dock = None
        self.nvidia_available=True
        self.load_umap_tool: LoadUmapTool

        def load_umap_btn_clicked():
            if self.plotter_widget is not None:
                ret = QMessageBox.question(self, '', "Do you really want to close the current UMAP and load another?", QMessageBox.Yes | QMessageBox.No)
                if ret == QMessageBox.No:
                    return
                self.viewer.window.remove_dock_widget(self.plotter_Widget_dock)
                for l in self.load_umap_tool.get_created_layers():
                    self.plotter_widget.layer_select.changed.disconnect() # otherwise I get an emit loop error
                    self.viewer.layers.remove(l)



            self.plotter_Widget_dock, self.plotter_widget = self.viewer.window.add_plugin_dock_widget('napari-clusters-plotter',
                                                                               widget_name='Plotter Widget',
                                                                               tabify=False)



            self.load_umap_tool = LoadUmapTool(pbar=self.progressBar, plotter_widget=self.plotter_widget)

            self.progressBar.setHidden(False)
            self.plotter_widget_run_func = self.plotter_widget.run
            self.plotter_widget.run=self.patched_run
            self.load_umap_tool.set_new_label_layer_name("UMAP")
            worker = self.load_umap_tool.start_umap_worker(self._selected_umap_pth.text())
            worker.start()

        self._load_umap_btn.clicked.connect(load_umap_btn_clicked)



        ####
        # Recalculate UMAP UI Elements
        ####
        self._run_umap_recalc_btn = QPushButton("Recalculate UMAP for selected clusters", self)
        self._run_umap_recalc_btn.clicked.connect(self._on_refine_click)
        self._run_umap_recalc_btn.setEnabled(False)
        self._run_umap_recalc_btn.setToolTip("Takes the embeddings assigned to a cluster and calculates a new UMAP based on these embeddings. This can be helpful to pinpoint the region that encodes the center of the protein or to clean clusters from unwanted embeddings.")
        if not self.check_if_gpu_is_available():
            self.nvidia_available = False
            self._run_umap_recalc_btn.setEnabled(False)
            self._run_umap_recalc_btn.setToolTip("No NVIDIA GPU available")
        self.layout().addRow("", self._run_umap_recalc_btn)


        self.refinement_done.connect(self.show_umap_callback)


        ####
        # Show target embedding position
        ####
        self._run_show_targets= QPushButton("Show target embedding positions", self)
        self._run_show_targets.clicked.connect(self._on_show_target_clicked)
        self._run_show_targets.setEnabled(False)
        self._run_show_targets.setToolTip("For each cluster, it estimates the target embedding (medoid) and visualizes its position in the tomogram with the same edge color as the cluster.")
        self.layout().addRow("", self._run_show_targets)
        self.target_calc_done.connect(self.show_targets_callback)


        ###
        # Other
        ###
        self.pbar_label = QLabel("")
        self.progressBar = LabeledProgressBar(self.pbar_label)

        self.progressBar.setRange(0, 0)
        self.progressBar.hide()
        self.layout().addRow(self.pbar_label,self.progressBar)
        self.tmp_dir_path: str
        self.setMaximumHeight(150)
        self._target_point_layer = None



    def cleanup(self):
        try:
            shutil.rmtree(self.tmp_dir_path)
        except AttributeError:
            # Means that the there was no recalculated UMAP
            pass

    def patched_run(self, *args, **kwargs):
        result = self.plotter_widget_run_func(*args, **kwargs)
        try:
            # The target points layer should get deleted when clusters are reseted
            # Furthermore, the button to calculate the targest should get disabled
            clusters = self.plotter_widget.layer_select.value.features['MANUAL_CLUSTER_ID']
            if len(np.unique(clusters)) == 1: # 1=only background cluster
                self.delete_points_layer()
                self._run_show_targets.setEnabled(False)
                self._run_umap_recalc_btn.setEnabled(False)
            else:
                self._run_show_targets.setEnabled(True)
                if self.nvidia_available:
                    self._run_umap_recalc_btn.setEnabled(True)
        except:
            pass
        return result

    @staticmethod
    def calc_targets(embedding_path: str , clusters: np.array) -> pd.DataFrame:
        # get embeddings
        embeddings = pd.read_pickle(embedding_path)
        embeddings = embeddings.drop(columns=["level_0", "index"], errors="ignore")

        # get clusters


        # calculate target positions
        _, _, target_locations = _make_targets(embeddings=embeddings, clusters=clusters, avg_func=_get_medoid_embedding)

        # Create points coords

        points = []
        for c in np.unique(clusters):
            c = int(c)
            if c == 0:
                continue
            points.append(target_locations[c][["Z", "Y", "X"]].drop(columns=["level_0", "index"], errors="ignore"))

        points = pd.concat(points)
        return points


    def _on_show_target_clicked(self):
        emb_pth = self.plotter_widget.layer_select.value.metadata['tomotwin']['embeddings_path']
        clusters = self.plotter_widget.layer_select.value.features['MANUAL_CLUSTER_ID']

        self.progressBar.setHidden(False)
        self.progressBar.set_label_text("Calculate target positions")

        ppe = futures.ProcessPoolExecutor(max_workers=1)
        f= ppe.submit(self.calc_targets, emb_pth, clusters)
        ppe.shutdown(wait=False)
        f.add_done_callback(self.target_calc_done.emit)


    def delete_points_layer(self):
        if self._target_point_layer is not None:
            self.viewer.layers.remove(self._target_point_layer)
            self._target_point_layer = None



       # pass
    def on_close_callback(self):
        self.cleanup()

    def set_plotter_widget(self, widget: PlotterWidget):
        self.plotter_widget = widget

    def set_umap_tool(self, tool: LoadUmapTool):
        self.load_umap_tool = tool

    def _on_refine_click(self):
        self.viewer.window._qt_window.setEnabled(False)
        self.delete_points_layer()
        self.reestimate_umap()

    @staticmethod
    def random_filename() -> str:
        return next(tempfile._get_candidate_names())

    def napari_update_umap(self, umap_embeddings, used_embeddings):

        self.tmp_dir_path = tempfile.mkdtemp()
        tmp_embed_pth = os.path.join(self.tmp_dir_path, UmapToolQt.random_filename())
        used_embeddings.to_pickle(tmp_embed_pth)
        umap_embeddings.attrs['embeddings_path'] = tmp_embed_pth
        tmp_umap_pth = os.path.join(self.tmp_dir_path, UmapToolQt.random_filename())
        umap_embeddings.to_pickle(tmp_umap_pth)

        # Visualizse it
        self.load_umap_tool.set_new_label_layer_name("UMAP Refined")
        worker = self.load_umap_tool.start_umap_worker(tmp_umap_pth)
        worker.start()
        #worker.finished.connect(lambda x,y: self.viewer.window._qt_window.setEnabled(True))


    def show_targets_callback(self, future: futures.Future):
        points: pd.DataFrame = future.result()
        point_colors = []
        colors = get_nice_colormap()
        import PIL.ImageColor as ImageColor

        for c in range(1,len(points)+1):
            rgba = [float(v) / 255
                    for v in list(
                    ImageColor.getcolor(colors[c % len(colors)], "RGB")
                )
                    ]
            rgba.append(0.9)
            point_colors.append(rgba)

        self.delete_points_layer()

        self._target_point_layer = self.viewer.add_points(points,
                                                          symbol='o',
                                                          size=37,
                                                          edge_color=point_colors,
                                                          face_color="transparent",
                                                          edge_width=0.10,
                                                          out_of_slice_display=True,
                                                          name="Targets")

        self.viewer.window._qt_window.setEnabled(True)
        self.progressBar.setHidden(True)
        self.progressBar.set_label_text("")


    def show_umap_callback(self, future: futures.Future):
        (umap_embeddings, used_embeddings) = future.result()
        self.viewer.window._qt_window.setEnabled(True)
        self.napari_update_umap(umap_embeddings, used_embeddings)

    def reestimate_umap(self):
        try:
            print("Read clusters")
            clusters = self.plotter_widget.layer_select.value.features['MANUAL_CLUSTER_ID']
            if not np.any(clusters>0):
                raise KeyError
        except KeyError:
            notifications.show_info(f"No cluster selected. Can't refine.")
            return

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

        print("Read embeddings")
        emb_pth = get_embedding_path(self.plotter_widget.layer_select.value.metadata['tomotwin']['embeddings_path'])


        if emb_pth == "":
            print("No path selected.")
            return

        self.plotter_widget.layer_select.value.metadata['tomotwin']['embeddings_path'] = emb_pth
        embeddings = pd.read_pickle(emb_pth)

        self.progressBar.setHidden(False)
        self.progressBar.set_label_text("Recalculate umap")

        # this workaround using signal is necessary, as "add_done_callback" starts the method
        # in a separate thread, but to change Qt elements, it must be run in the same thread as the main program.
        ppe = futures.ProcessPoolExecutor(max_workers=1)
        f= ppe.submit(UmapRefiner.refine, clusters,embeddings)
        ppe.shutdown(wait=False)
        f.add_done_callback(self.refinement_done.emit)



