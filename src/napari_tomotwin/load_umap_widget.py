import math
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import os
import os.path
import shutil
import tempfile
from concurrent import futures
import numpy as np
import pandas as pd
from napari.utils import notifications
from napari_clusters_plotter._plotter import PlotterWidget
from napari_clusters_plotter._utilities import get_nice_colormap
from napari_tomotwin.make_targets_widget import _make_targets, _get_medoid_embedding
from napari_tomotwin.load_umap import LoadUmapTool
from qtpy.QtWidgets import QApplication
from . import umap_refiner as urefine


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
from napari_tomotwin._qt.labeled_progress_bar import LabeledProgressBar



class UmapToolQt(QWidget):

    target_calc_done = Signal("PyQt_PyObject")

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()

        self.viewer = napari_viewer


        #######
        # UI Setup
        ######
        layout = QFormLayout()
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

            if self._selected_umap_pth.text() == None or self._selected_umap_pth.text() == "":
                return

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

            self.load_umap_tool = LoadUmapTool(plotter_widget=self.plotter_widget)
            self.load_umap_tool.set_progressbar(self.progressBar)

            self.cluster_widget_dock, self.cluster_widget = self.viewer.window.add_plugin_dock_widget(
                'napari-tomotwin',
                widget_name='ClusterTool',
                tabify=True)
            self.cluster_widget.set_plotter_widget(self.plotter_widget)
            #self.cluster_widget.set_umap_tool(self.load_umap_tool)

            self.progressBar.setHidden(False)
            self.plotter_widget_run_func = self.plotter_widget.run
            self.plotter_widget.run=self.patched_run
            self.load_umap_tool.set_new_label_layer_name("UMAP")
            worker = self.load_umap_tool.start_umap_worker(self._selected_umap_pth.text())
            worker.start()

        self._load_umap_btn.clicked.connect(load_umap_btn_clicked)



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
        except Exception as e:
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
            try:
                self.viewer.layers.remove(self._target_point_layer)
            except ValueError:
                # Then it somehow got deleted
                pass
            self._target_point_layer = None



    def set_plotter_widget(self, widget: PlotterWidget):
        self.plotter_widget = widget

    def set_umap_tool(self, tool: LoadUmapTool):
        self.load_umap_tool = tool


    @staticmethod
    def random_filename() -> str:
        return next(tempfile._get_candidate_names())



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




