import os
import shutil
import tempfile
from concurrent import futures

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QComboBox, QStyledItemDelegate
from napari.utils import notifications
from napari_clusters_plotter._plotter import PlotterWidget
from napari_clusters_plotter._utilities import get_nice_colormap
from napari_tomotwin._qt.labeled_progress_bar import LabeledProgressBar
from napari_tomotwin.make_targets_widget import _make_targets, _get_medoid_embedding
from napari_tomotwin.load_umap import LoadUmapTool
from qtpy.QtCore import Qt
from qtpy.QtCore import Signal
from qtpy.QtGui import QColor, QPixmap
from qtpy.QtWidgets import QApplication
from qtpy.QtWidgets import (
    QFormLayout,
    QPushButton,
    QWidget,
    QFileDialog,
    QMessageBox,
    QLabel,
    QHBoxLayout,
    QListView,
    QTableWidgetItem,
    QTableWidget
)

from . import umap_refiner as urefine


class ColorItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # Draw a colored rectangle in the item
        color = index.data(Qt.UserRole)
        if color is not None and isinstance(color, QColor):
            painter.fillRect(option.rect, color)

    def sizeHint(self, option, index):
        # Set the size of the item
        return index.data(Qt.SizeHintRole) if index.data(Qt.SizeHintRole) else super().sizeHint(option, index)


class ColorComboBox(QComboBox):
    def __init__(self):
        super().__init__()

        # Set the custom item delegate
        delegate = ColorItemDelegate(self)
        self.setItemDelegate(delegate)

        # Set the combo box to only display icons
        self.setView(QListView(self))


class ClusteringWidgetQt(QWidget):
    refinement_done = Signal("PyQt_PyObject")
    target_calc_done = Signal("PyQt_PyObject")

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()

        self.viewer = napari_viewer
        self.plotter_widget: PlotterWidget
        self._load_umap_tool = None
        self.tmp_dir_path: str = None
        self.pbar_label = QLabel("")
        self.progressbar = LabeledProgressBar(self.pbar_label)
        self.progressbar.setRange(0, 0)
        self.progressbar.setHidden(True)
        self.added_canditates: int = 0
        self._target_point_layer = None

        #######
        # UI Setup
        ######
        layout = QFormLayout()
        app = QApplication.instance()
        app.lastWindowClosed.connect(self._on_close_callback)  # this line is connection to signal
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.setLayout(layout)

        ####
        recalc_layout = QHBoxLayout()
        self._recalc_umap = QPushButton("Recalculate UMAP", self)
        self._recalc_umap.clicked.connect(self._on_refine_click)
        self._recalc_umap.setEnabled(False)
        self._recalc_umap.setToolTip(
            "Takes the embeddings assigned to a cluster and calculates a new UMAP based on these embeddings. This can be helpful to pinpoint the region that encodes the center of the protein or to clean clusters from unwanted embeddings.")
        self.nvidia_available = self.check_if_gpu_is_available()
        if not self.nvidia_available:
            self.nvidia_available = False
            self._recalc_umap.setEnabled(False)
            self._recalc_umap.setToolTip("No NVIDIA GPU available")
        self.refinement_done.connect(self.show_umap_callback)
        self._cluster_dropdown = self.get_current_cluster_dropdown()
        self._show_targets = QPushButton("Show target", self)
        self._show_targets.clicked.connect(self._on_show_target_clicked)
        self._show_targets.setToolTip("For the selected cluster it estimates the target embedding (medoid) and visualizes its position in the tomogram with the same edge color as the cluster.")

        self.target_calc_done.connect(self.show_targets_callback)

        self._add_candidate = QPushButton("Add candidate", self)
        self._add_candidate.setEnabled(False)
        self._add_candidate.clicked.connect(self._on_add_candidate_clicked)

        recalc_layout.addWidget(self._cluster_dropdown)
        recalc_layout.addWidget(self._show_targets)
        recalc_layout.addWidget(self._recalc_umap)
        recalc_layout.addWidget(self._add_candidate)

        self.layout().addRow("", recalc_layout)

        ## Now q table widget
        candlabl = QLabel("Candidates:")
        self.layout().addWidget(candlabl)
        self.tableWidget = QTableWidget(self)
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setHorizontalHeaderLabels(["", "Color", "UMAP", "Description"])
        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)
        self.layout().addWidget(self.tableWidget)

        ## Save and delete
        save_delete_layout = QHBoxLayout()
        self.save = QPushButton("Save candidates", self)
        self.save.clicked.connect(self.update_all)
        self.delete = QPushButton("Delete candidates", self)
        self.delete.clicked.connect(self.delete_candidate)
        save_delete_layout.addWidget(self.delete)
        save_delete_layout.addWidget(self.save)
        self.layout().addRow("", save_delete_layout)
        pbar_layout = QHBoxLayout()
        pbar_layout.addWidget(self.pbar_label)
        pbar_layout.addWidget(self.progressbar)
        self.layout().addRow("",pbar_layout)
        self.setMinimumHeight(300)

    def get_umap_tool(self) -> LoadUmapTool:
        if self._load_umap_tool is None:
            self._load_umap_tool = LoadUmapTool(self.plotter_widget)
            self._load_umap_tool.set_progressbar(self.progressbar)
        return self._load_umap_tool

    @staticmethod
    def check_if_gpu_is_available() -> bool:
        try:
            import cudf
            import cuml
        except:
            return False
        return True

    def set_plotter_widget(self, plotter_widget: PlotterWidget):
        self.plotter_widget = plotter_widget
        self.plotter_widget_run_func = self.plotter_widget.run
        self.plotter_widget.graphics_widget.mpl_connect('draw_event', lambda _: self.after_draw_event())

    def delete_candidate(self):
        if self.tableWidget.currentItem() is None:
            return
        selected_row = self.tableWidget.currentRow()
        if selected_row >= 0:
            self.tableWidget.removeRow(selected_row)
            self.tableWidget.clearSelection()
            self.tableWidget.setCurrentItem(None)

    def delete_points_layer(self):
        if self._target_point_layer is not None:
            try:
                self.viewer.layers.remove(self._target_point_layer)
            except ValueError:
                # Then it somehow got deleted
                pass
            self._target_point_layer = None

    def after_draw_event(self):
        self.update_all()
        try:
            # The target points layer should get deleted when clusters are reseted
            # Furthermore, the button to calculate the targest should get disabled
            no_clusters = True
            if 'MANUAL_CLUSTER_ID' in self.plotter_widget.layer_select.value.features:
                clusters = self.plotter_widget.layer_select.value.features['MANUAL_CLUSTER_ID']
                ucl = np.unique(clusters)
                no_clusters = len(ucl)==1

            if no_clusters:  # 1=only background cluster
                self.delete_points_layer()
                self._recalc_umap.setEnabled(False)
                self._add_candidate.setEnabled(False)
                self._show_targets.setEnabled(False)
            else:
                if self.nvidia_available:
                    self._recalc_umap.setEnabled(True)
                self._add_candidate.setEnabled(True)
                self._show_targets.setEnabled(True)

        except Exception as e:
            print(e)
            pass

    def cleanup(self):
        if self.tmp_dir_path is None:
            return
        try:
            shutil.rmtree(self.tmp_dir_path)
        except AttributeError:
            # Means that the there was no recalculated UMAP
            pass

    @staticmethod
    def calc_targets(embedding_path: str, clusters: np.array, target_cluster: int) -> pd.DataFrame:
        # get embeddings
        embeddings = pd.read_pickle(embedding_path)
        embeddings = embeddings.drop(columns=["level_0", "index"], errors="ignore")

        # get clusters

        # calculate target positions
        _, _, target_locations = _make_targets(embeddings=embeddings, clusters=clusters, avg_func=_get_medoid_embedding, target_cluster=target_cluster)

        # Create points coords

        points = []
        for c in np.unique(clusters):
            c = int(c)
            if c == 0:
                continue
            if target_cluster is not None and target_cluster != c:
                continue

            points.append(target_locations[c][["Z", "Y", "X"]].drop(columns=["level_0", "index"], errors="ignore"))

        points = pd.concat(points)
        return points

    def _on_show_target_clicked(self):
        emb_pth = self.plotter_widget.layer_select.value.metadata['tomotwin']['embeddings_path']
        clusters = self.plotter_widget.layer_select.value.features['MANUAL_CLUSTER_ID']

        self.progressbar.setHidden(False)
        self.progressbar.set_label_text("Calculate target positions")

        target_cluster = self._cluster_dropdown.currentIndex() + 1

        ppe = futures.ProcessPoolExecutor(max_workers=1)

        f= ppe.submit(self.calc_targets, emb_pth, clusters, target_cluster)
        ppe.shutdown(wait=False)
        f.add_done_callback(self.target_calc_done.emit)


    def show_targets_callback(self, future: futures.Future):
        points: pd.DataFrame = future.result()
        point_colors = []
        colors = get_nice_colormap()
        import PIL.ImageColor as ImageColor

        c = self._cluster_dropdown.currentIndex() + 1

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

        self.viewer.dims.set_current_step(0, int(points[['Z']].to_numpy()[0,0]))
        self.viewer.window._qt_window.setEnabled(True)
        self.progressbar.setHidden(True)
        self.progressbar.set_label_text("")


    def _on_close_callback(self):
        self.cleanup()

    def _on_add_candidate_clicked(self):

        current_row_count = self.tableWidget.rowCount()
        self.tableWidget.setRowCount(current_row_count + 1)
        self.added_canditates = self.added_canditates + 1
        c = self._cluster_dropdown.currentIndex() + 1
        entry = ["", "", self.plotter_widget.layer_select.value.name, f"target-{self.added_canditates}"]
        for col, value in enumerate(entry):
            item = QTableWidgetItem(value)
            if col == 0:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
            if col == 1:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                pixmap = QPixmap(10, 10)
                pixmap.fill(QColor(*self.index_to_rgba(c)))
                item.setData(Qt.DecorationRole, pixmap)
            if col == 2:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

            self.tableWidget.setItem(current_row_count, col, item)

    def napari_update_umap(self, umap_embeddings, used_embeddings):

        self.tmp_dir_path = tempfile.mkdtemp()
        tmp_embed_pth = os.path.join(self.tmp_dir_path, next(tempfile._get_candidate_names()))
        used_embeddings.to_pickle(tmp_embed_pth)
        umap_embeddings.attrs['embeddings_path'] = tmp_embed_pth
        tmp_umap_pth = os.path.join(self.tmp_dir_path, next(tempfile._get_candidate_names()))
        umap_embeddings.to_pickle(tmp_umap_pth)

        # Visualizse it
        self.get_umap_tool().set_new_label_layer_name("UMAP Refined")
        worker = self.get_umap_tool().start_umap_worker(tmp_umap_pth)
        worker.start()
        # worker.returned.connect(self.update_all)

    def show_umap_callback(self, future: futures.Future):
        print("SHOW")
        (umap_embeddings, used_embeddings) = future.result()
        self.viewer.window._qt_window.setEnabled(True)
        self.napari_update_umap(umap_embeddings, used_embeddings)

    @staticmethod
    def index_to_rgba(index: int) -> list[int]:
        colors = get_nice_colormap()
        import PIL.ImageColor as ImageColor
        rgba = [int(v)
                for v in list(
                ImageColor.getcolor(colors[index % len(colors)], "RGB")
            )
                ]
        rgba.append(int(255 * 0.9))
        return rgba

    def update_all(self):
        cls = []
        if hasattr(self.plotter_widget.layer_select.value, 'features') and 'MANUAL_CLUSTER_ID' in self.plotter_widget.layer_select.value.features:
            cls = self.plotter_widget.layer_select.value.features['MANUAL_CLUSTER_ID']
        self.update_items_cluster_dropdown(self._cluster_dropdown, cls)


    def update_items_cluster_dropdown(self, dropdown: QComboBox, cluster_ids: list[int]):

        dropdown.clear()
        for c in np.unique(cluster_ids):
            if c == 0:
                continue
            rgba = self.index_to_rgba(c)
            dropdown.addItem("")
            pixmap = QPixmap(10, 10)
            pixmap.fill(QColor(*rgba))
            dropdown.setItemData(dropdown.count() - 1, pixmap, Qt.DecorationRole)

    def _on_refine_click(self):
        self.viewer.window._qt_window.setEnabled(False)
        self.delete_points_layer()
        self.reestimate_umap()

    def get_current_cluster_dropdown(self):
        color_dropdown = QComboBox(self)
        color_dropdown.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        return color_dropdown

    def reestimate_umap(self):
        try:
            print("Read clusters")
            clusters = self.plotter_widget.layer_select.value.features['MANUAL_CLUSTER_ID']
            if not np.any(clusters > 0):
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
                msg.setInformativeText(
                    "Embedding path in metadata data (see below) does not exist or can't be accesst. Please click OK and select the path to the embedding file.")
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

        self.progressbar.setHidden(False)
        self.progressbar.set_label_text("Recalculate umap")

        # this workaround using signal is necessary, as "add_done_callback" starts the method
        # in a separate thread, but to change Qt elements, it must be run in the same thread as the main program.
        ppe = futures.ProcessPoolExecutor(max_workers=1)
        target_cluster = self._cluster_dropdown.currentIndex() + 1
        f = ppe.submit(urefine.refine, clusters, embeddings, target_cluster)
        ppe.shutdown(wait=False)
        f.add_done_callback(self.refinement_done.emit)