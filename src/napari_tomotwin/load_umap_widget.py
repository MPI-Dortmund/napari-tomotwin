import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
import os
import os.path
import tempfile
from concurrent import futures
from napari_clusters_plotter._plotter import PlotterWidget
from napari_tomotwin.load_umap import LoadUmapTool
from napari.utils import notifications

from qtpy.QtWidgets import (
    QFormLayout,
    QPushButton,
    QWidget,
    QFileDialog,
    QMessageBox,
    QLabel,
    QHBoxLayout,
    QLineEdit,
)
from napari_tomotwin._qt.labeled_progress_bar import LabeledProgressBar


class UmapToolQt(QWidget):

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
        self._select_umap_pth_btn = QPushButton("Select", self)
        self._selected_umap_pth = QLineEdit()

        def select_file_clicked():
            pth = QFileDialog.getOpenFileName(
                self, "Open UMAP file", os.getcwd(), "UMAP file (*.tumap)"
            )[0]
            self._selected_umap_pth.setText(pth)

        self._select_umap_pth_btn.clicked.connect(select_file_clicked)
        self._load_umap_btn = QPushButton("Load")
        umap_pth_layout.addWidget(self._selected_umap_pth)
        umap_pth_layout.addWidget(self._select_umap_pth_btn)
        umap_pth_layout.addWidget(self._load_umap_btn)
        self.layout().addRow("Path to UMAP:", umap_pth_layout)
        self.plotter_widget: PlotterWidget = None
        self.plotter_widget_run_func = None
        self.plotter_Widget_dock = None
        self.cluster_widget_dock = None
        self.nvidia_available = True
        self.load_umap_tool: LoadUmapTool

        def load_umap_btn_clicked():

            if (
                self._selected_umap_pth.text() == None
                or self._selected_umap_pth.text() == ""
            ):
                notifications.show_error("No file selected")
                return

            if os.path.exists(self._selected_umap_pth.text()) == False:
                notifications.show_error("Path does not exist.")
                return

            if self.plotter_widget is not None:
                ret = QMessageBox.question(
                    self,
                    "",
                    "Do you really want to close the current UMAP and load another?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if ret == QMessageBox.No:
                    return
                self.viewer.window.remove_dock_widget(self.plotter_Widget_dock)
                self.viewer.window.remove_dock_widget(self.cluster_widget_dock)
                for l in self.load_umap_tool.get_created_layers():
                    self.plotter_widget.layer_select.changed.disconnect()  # otherwise I get an emit loop error
                    self.viewer.layers.remove(l)

            self.plotter_Widget_dock, self.plotter_widget = (
                self.viewer.window.add_plugin_dock_widget(
                    "napari-clusters-plotter",
                    widget_name="Plotter Widget",
                    tabify=False,
                )
            )

            from .cluster_widget import ClusteringWidgetQt

            self.cluster_widget = ClusteringWidgetQt(self.viewer)
            self.cluster_widget_dock = self.viewer.window.add_dock_widget(
                self.cluster_widget, area="right", name="UMAP Tools"
            )

            self.load_umap_tool = LoadUmapTool(
                plotter_widget=self.plotter_widget
            )
            self.load_umap_tool.set_progressbar(self.progressBar)

            self.cluster_widget.set_plotter_widget(self.plotter_widget)
            self.progressBar.setHidden(False)

            self.load_umap_tool.set_new_label_layer_name("UMAP")
            worker = self.load_umap_tool.start_umap_worker(
                self._selected_umap_pth.text()
            )
            worker.start()

        self._load_umap_btn.clicked.connect(load_umap_btn_clicked)

        ###
        # Other
        ###
        self.pbar_label = QLabel("")
        self.progressBar = LabeledProgressBar(self.pbar_label)

        self.progressBar.setRange(0, 0)
        self.progressBar.hide()
        self.layout().addRow(self.pbar_label, self.progressBar)
        self.tmp_dir_path: str
        self.setMaximumHeight(100)
        self._target_point_layer = None

    def set_plotter_widget(self, widget: PlotterWidget):
        self.plotter_widget = widget

    def set_umap_tool(self, tool: LoadUmapTool):
        self.load_umap_tool = tool

    @staticmethod
    def random_filename() -> str:
        return next(tempfile._get_candidate_names())

    def show_umap_callback(self, future: futures.Future):
        (umap_embeddings, used_embeddings) = future.result()
        self.viewer.window._qt_window.setEnabled(True)
        self.napari_update_umap(umap_embeddings, used_embeddings)
