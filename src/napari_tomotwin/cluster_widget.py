import os
import shutil
import tempfile
from concurrent import futures

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QComboBox,
    QStyledItemDelegate,
    QHeaderView,
    QMenu,
    QAction,
)
from napari.qt.threading import thread_worker
from napari.utils import notifications
from napari_clusters_plotter import PlotterWidget
from nap_plot_tools.cmap import cat10_mod_cmap
from napari_tomotwin._qt.labeled_progress_bar import LabeledProgressBar
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
    QTableWidgetItem,
    QTableWidget,
)

from . import umap_refiner as urefine
from .make_targets import (
    _make_targets,
    _get_medoid_embedding,
)
from .target_manager import TargetManager, Target


def _get_active_layer(plotter_widget):
    """Helper to get the active layer from PlotterWidget.
    
    In napari-clusters-plotter 0.10+, layers is a list of selected layers.
    Returns the first layer or None if no layers are selected.
    """
    if plotter_widget.layers and len(plotter_widget.layers) > 0:
        return plotter_widget.layers[0]
    return None


class ColorItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # Draw a colored rectangle in the item
        color = index.data(Qt.UserRole)
        if color is not None and isinstance(color, QColor):
            painter.fillRect(option.rect, color)

    def sizeHint(self, option, index):
        # Set the size of the item
        return (
            index.data(Qt.SizeHintRole)
            if index.data(Qt.SizeHintRole)
            else super().sizeHint(option, index)
        )


class QClusterPixmap(QPixmap):

    # def __init__(self):
    #    super().__init__(10,10)
    #    #print("INIT")
    #    #self.cluster = None

    def set_cluster(self, cluster: int):
        pass
        # self.cluster = cluster

    def get_cluster(self) -> int:
        return self.cluster


class ClusteringWidgetQt(QWidget):
    refinement_done = Signal("PyQt_PyObject")
    target_calc_done = Signal("PyQt_PyObject")

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()

        self.viewer = napari_viewer
        self.plotter_widget: PlotterWidget
        self._load_umap_tool = None
        self.tmp_dir_path: str = None
        self.base_umap_features: pd.DataFrame = None  # Store the original UMAP features
        self.base_umap_metadata: dict = None  # Store the original metadata
        self.pbar_label = QLabel("")
        self.progressbar = LabeledProgressBar(self.pbar_label)
        self.progressbar.setRange(0, 0)
        self.progressbar.setHidden(True)
        self.added_canditates: int = 0
        self._target_point_layer = None
        self.target_manger = TargetManager()

        #######
        # UI Setup
        ######
        layout = QFormLayout()
        app = QApplication.instance()
        app.lastWindowClosed.connect(
            self._on_close_callback
        )  # this line is connection to signal
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.setLayout(layout)

        ####
        recalc_layout = QHBoxLayout()
        self._recalc_umap = QPushButton("Remap", self)
        self._recalc_umap.clicked.connect(self._on_refine_click)
        self._recalc_umap.setEnabled(False)
        self._recalc_umap.setToolTip(
            "Takes the embeddings assigned to a cluster and calculates a new UMAP based on these embeddings. This can be helpful to pinpoint the region that encodes the center of the protein or to clean clusters from unwanted embeddings."
        )
        self.nvidia_available = self.check_if_gpu_is_available()
        if not self.nvidia_available:
            self.nvidia_available = False
            self._recalc_umap.setEnabled(False)
            self._recalc_umap.setToolTip("No NVIDIA GPU available")
        self.refinement_done.connect(self.show_umap_callback)
        self._cluster_dropdown = self.get_current_cluster_dropdown()
        self._show_targets = QPushButton("Show target", self)
        self._show_targets.clicked.connect(self._on_show_target_clicked)
        self._show_targets.setToolTip(
            "For the selected cluster it estimates the target embedding (medoid) and visualizes its position in the tomogram with the same edge color as the cluster."
        )

        self.target_calc_done.connect(self.show_targets_callback)

        self._add_candidate = QPushButton("Add candidate", self)
        self._add_candidate.setEnabled(False)
        self._add_candidate.clicked.connect(self._on_add_candidate_clicked)

        recalc_layout.addWidget(self._cluster_dropdown)
        recalc_layout.addWidget(self._show_targets)
        recalc_layout.addWidget(self._recalc_umap)
        recalc_layout.addWidget(self._add_candidate)

        self.layout().addRow("", recalc_layout)

        # Show Base UMAP button
        self._show_base_umap = QPushButton("Show Base UMAP", self)
        self._show_base_umap.clicked.connect(self._on_show_base_umap_click)
        self._show_base_umap.setEnabled(False)
        self._show_base_umap.setToolTip(
            "Return to the original base UMAP after remapping."
        )
        self.layout().addWidget(self._show_base_umap)

        ## Now q table widget
        candlabl = QLabel("Candidates:")
        self.layout().addWidget(candlabl)
        self.tableWidget = QTableWidget(self)
        self.tableWidgetHeaders = ["ID", "Color", "UMAP", "Label"]
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setHorizontalHeaderLabels(self.tableWidgetHeaders)
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)

        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)
        self.tableWidget.setSelectionMode(QTableWidget.SingleSelection)
        self.tableWidget.itemChanged.connect(self._table_item_name_changed)

        self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)

        self.tableWidget.customContextMenuRequested.connect(
            self.show_table_context_menu
        )

        self.layout().addWidget(self.tableWidget)

        ## Save
        self.save = QPushButton("Save candidates", self)
        self.save.clicked.connect(self._save_candidate_click)
        self.layout().addWidget(self.save)

        ## Progressbar layout
        pbar_layout = QHBoxLayout()
        pbar_layout.addWidget(self.pbar_label)
        pbar_layout.addWidget(self.progressbar)
        layout.addRow("", pbar_layout)
        self.setMinimumHeight(300)

    def set_umap_tool(self, tool: LoadUmapTool):
        self._load_umap_tool = tool

    def get_umap_tool(self) -> LoadUmapTool:
        if self._load_umap_tool is None:
            self._load_umap_tool = LoadUmapTool(self.plotter_widget)
            self._load_umap_tool.set_progressbar(self.progressbar)
        return self._load_umap_tool

    def show_table_context_menu(self, pos):
        context_menu = QMenu(self)
        action_show = QAction("Show", self)
        action_delete = QAction("Delete", self)
        action_delete.triggered.connect(self.delete_candidate)
        action_show.triggered.connect(self.show_candidate)
        context_menu.addAction(action_show)
        context_menu.addAction(action_delete)
        context_menu.exec_(self.tableWidget.mapToGlobal(pos))

    @staticmethod
    def check_if_gpu_is_available() -> bool:
        try:
            import cudf
            import cuml
        except:
            return False
        return True

    def replot_cluster_plotter(self):
        # In napari-clusters-plotter 0.10+, just emit the update signal
        # The widget handles axis settings via properties
        try:
            self.plotter_widget.x_axis = "umap_0"
            self.plotter_widget.y_axis = "umap_1"
            self.plotter_widget.plot_needs_update.emit()
        except AttributeError:
            # In this case, replotting is not yet possible
            pass

    def show_candidate(self):

        if self.tableWidget.currentItem() is None:
            return
        selected_row = self.tableWidget.currentRow()
        if selected_row >= 0:
            first_column_item = self.tableWidget.item(selected_row, 0)

            id = int(first_column_item.text())

            target = self.target_manger.get_target_by_id(id)
            # In napari-clusters-plotter 0.10+, select layer via napari's layer selection
            self.viewer.layers.selection.active = target.layer
            clids = np.zeros(
                shape=target.embeddings_mask.shape, dtype=np.int64
            )
            clids[target.embeddings_mask] = target.cluster_id
            target.layer.features[
                "MANUAL_CLUSTER_ID"
            ] = clids
            self.replot_cluster_plotter()
            self.update_all()
            # Update label colors to make non-selected points transparent
            self._update_label_colors_with_transparent_background(target.layer)

    def set_plotter_widget(self, plotter_widget: PlotterWidget):
        self.plotter_widget = plotter_widget
        # In napari-clusters-plotter 0.10+, use signals from biaplotter instead of mpl_connect
        # Connect to artist_changed_signal and selector_changed_signal for UI updates
        self.plotter_widget.plotting_widget.artist_changed_signal.connect(
            lambda _: self.after_draw_event()
        )
        self.plotter_widget.plotting_widget.selector_changed_signal.connect(
            lambda _: self.after_draw_event()
        )
        # Connect to selection_applied_signal from all selectors to update UI after lasso/ellipse/rectangle selection
        # and auto-increment the class so next selection gets a different color (only when Ctrl is held)
        for selector in self.plotter_widget.plotting_widget.selectors.values():
            selector.selection_applied_signal.connect(
                lambda _: self._on_selection_applied()
            )
        
        # Connect to canvas button_press_event to reset features BEFORE selection starts (if Ctrl not held)
        self.plotter_widget.plotting_widget.canvas.mpl_connect(
            'button_press_event', self._on_canvas_button_press
        )

    def _on_canvas_button_press(self, event):
        """Called when mouse button is pressed on canvas. Resets features if Ctrl is not held."""
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QGuiApplication
        import pandas as pd
        
        print(f"DEBUG _on_canvas_button_press: button={event.button}")
        
        # Only handle left mouse button (button 1)
        if event.button != 1:
            print("DEBUG _on_canvas_button_press: ignoring non-left button")
            return
        
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl_held = modifiers == Qt.ControlModifier
        # Store ctrl_held state for use in _on_selection_applied
        # (by the time selection_applied_signal fires, user may have released Ctrl)
        self._ctrl_held_at_selection_start = ctrl_held
        print(f"DEBUG _on_canvas_button_press: ctrl_held={ctrl_held}")
        
        if not ctrl_held:
            print("DEBUG _on_canvas_button_press: resetting all MANUAL_CLUSTER_ID to 0")
            # No Ctrl: reset all previous selections before the new one is made
            for layer in self.plotter_widget.layers:
                if "MANUAL_CLUSTER_ID" in layer.features.columns:
                    # Reset all cluster IDs to 0
                    cluster_ids = layer.features["MANUAL_CLUSTER_ID"].to_numpy().copy()
                    sum_before = pd.to_numeric(cluster_ids, errors='coerce').sum()
                    non_zero_before = (pd.to_numeric(cluster_ids, errors='coerce') != 0).sum()
                    print(f"DEBUG _on_canvas_button_press: BEFORE reset - sum={sum_before}, non_zero_count={non_zero_before}")
                    cluster_ids[:] = 0
                    layer.features["MANUAL_CLUSTER_ID"] = pd.Series(cluster_ids).astype("category")
                    # Verify after reset
                    cluster_ids_after = layer.features["MANUAL_CLUSTER_ID"].to_numpy()
                    sum_after = pd.to_numeric(cluster_ids_after, errors='coerce').sum()
                    non_zero_after = (pd.to_numeric(cluster_ids_after, errors='coerce') != 0).sum()
                    print(f"DEBUG _on_canvas_button_press: AFTER reset - sum={sum_after}, non_zero_count={non_zero_after}")
                    print(f"DEBUG _on_canvas_button_press: reset layer {layer.name}")
            
            # CRITICAL: Also reset the artist's color_indices to 0!
            # The biaplotter selector reads color_indices from the artist (not layer.features)
            # and ADDS the new selection to it. If we don't reset color_indices, the old
            # selections accumulate even though layer.features was reset.
            try:
                active_artist = self.plotter_widget.plotting_widget.active_artist
                if active_artist is not None and hasattr(active_artist, 'color_indices'):
                    num_points = len(active_artist.color_indices) if active_artist.color_indices is not None else 0
                    print(f"DEBUG _on_canvas_button_press: resetting artist color_indices (length={num_points})")
                    if num_points > 0:
                        active_artist.color_indices = np.zeros(num_points, dtype=int)
                        print(f"DEBUG _on_canvas_button_press: artist color_indices reset to zeros")
            except Exception as e:
                print(f"DEBUG _on_canvas_button_press: error resetting artist color_indices: {e}")
            
            # Update the tomogram highlight immediately after reset
            # This ensures that a single click (without dragging) clears the highlight
            self.after_draw_event()
        else:
            print("DEBUG _on_canvas_button_press: Ctrl held, keeping previous selections")
            # Ctrl held: increment class BEFORE the selection is made so the new selection
            # gets a different color than the previous one
            current_class = self.plotter_widget.plotting_widget.class_spinbox.value
            new_class = current_class + 1
            self.plotter_widget.plotting_widget.class_spinbox.value = new_class
            print(f"DEBUG _on_canvas_button_press: Ctrl held, incremented class from {current_class} to {new_class}")

    def _on_selection_applied(self):
        """Called after a selection is applied. Updates UI and handles class value based on Ctrl key."""
        import pandas as pd
        
        print("DEBUG _on_selection_applied: called")
        
        # Show cluster IDs after selection was applied
        for layer in self.plotter_widget.layers:
            if "MANUAL_CLUSTER_ID" in layer.features.columns:
                cluster_ids = layer.features["MANUAL_CLUSTER_ID"].to_numpy()
                sum_ids = pd.to_numeric(cluster_ids, errors='coerce').sum()
                non_zero_count = (pd.to_numeric(cluster_ids, errors='coerce') != 0).sum()
                print(f"DEBUG _on_selection_applied: layer {layer.name} - sum={sum_ids}, non_zero_count={non_zero_count}")
        
        # Use the ctrl_held state that was saved at button press time
        # (by the time this signal fires, user may have released Ctrl already)
        ctrl_held = getattr(self, '_ctrl_held_at_selection_start', False)
        current_class = self.plotter_widget.plotting_widget.class_spinbox.value
        
        print(f"DEBUG _on_selection_applied: ctrl_held={ctrl_held}, current_class={current_class}")
        
        if ctrl_held:
            # Ctrl held: class was already incremented in _on_canvas_button_press BEFORE
            # the selection was made, so don't increment again here
            print(f"DEBUG _on_selection_applied: Ctrl held, class already incremented to {current_class}")
        else:
            # No Ctrl: reset class to number of candidates + 1 for next selection
            num_candidates = self.tableWidget.rowCount()
            new_class = num_candidates + 1
            self.plotter_widget.plotting_widget.class_spinbox.value = new_class
            print(f"DEBUG _on_selection_applied: No Ctrl, reset class to {new_class}")
        
        self.after_draw_event()

    def delete_candidate(self):
        if self.tableWidget.currentItem() is None:
            return
        selected_row = self.tableWidget.currentRow()
        if selected_row >= 0:

            first_column_item = self.tableWidget.item(selected_row, 0)

            id = int(first_column_item.text())
            self.target_manger.remove_target([id])

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
            active_layer = self.plotter_widget.layers[0] if self.plotter_widget.layers else None
            if active_layer is not None and (
                "MANUAL_CLUSTER_ID"
                in active_layer.features
            ):
                clusters = active_layer.features[
                    "MANUAL_CLUSTER_ID"
                ]
                ucl = np.unique(clusters)
                no_clusters = len(ucl) == 1

            if no_clusters:  # 1=only background cluster
                self.delete_points_layer()
                self._recalc_umap.setEnabled(False)
                self._add_candidate.setEnabled(False)
                self._show_targets.setEnabled(False)
                if active_layer is not None:
                    active_layer.opacity = 0  # Hide when no clusters
            else:
                if self.nvidia_available:
                    self._recalc_umap.setEnabled(True)
                self._add_candidate.setEnabled(True)
                self._show_targets.setEnabled(True)
                if active_layer is not None:
                    active_layer.opacity = 1  # Show when clusters are selected
                    # Make non-selected labels (MANUAL_CLUSTER_ID = 0) transparent
                    self._update_label_colors_with_transparent_background(active_layer)

        except Exception as e:
            print(e)
            pass

    def _update_label_colors_with_transparent_background(self, layer):
        """Update the label colormap to make non-selected labels transparent."""
        try:
            from napari.utils import DirectLabelColormap
            
            features = layer.features
            if "MANUAL_CLUSTER_ID" not in features.columns:
                return
            if "label" not in features.columns:
                return
                    
            cluster_ids = features["MANUAL_CLUSTER_ID"].values
            label_values = features["label"].values
            
            # Build color dict: transparent for MANUAL_CLUSTER_ID=0, colored for others
            color_dict = {0: np.array([0, 0, 0, 0])}  # Background always transparent
            
            for label_val, cluster_id in zip(label_values, cluster_ids):
                label_int = int(label_val)
                if cluster_id == 0:
                    # Non-selected: transparent
                    color_dict[label_int] = np.array([0, 0, 0, 0])
                else:
                    # Selected: use cluster color
                    rgba = self.index_to_rgba(int(cluster_id))
                    color_dict[label_int] = np.array([c / 255.0 for c in rgba])
            
            layer.colormap = DirectLabelColormap(color_dict=color_dict)
            layer.refresh()
        except Exception as e:
            print(f"Error updating label colors: {e}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        if self.tmp_dir_path is None:
            return
        try:
            shutil.rmtree(self.tmp_dir_path)
        except AttributeError:
            # Means that the there was no recalculated UMAP
            pass

    @staticmethod
    def calc_targets(
        embedding_path: str, clusters: np.array, target_cluster: int
    ) -> pd.DataFrame:
        # get embeddings
        embeddings = pd.read_pickle(embedding_path)
        embeddings = embeddings.drop(
            columns=["level_0", "index"], errors="ignore"
        )

        # get clusters

        # calculate target positions
        _, _, target_locations = _make_targets(
            embeddings=embeddings,
            clusters=clusters,
            avg_func=_get_medoid_embedding,
            target_cluster=target_cluster,
        )

        # Create points coords

        points = []
        for c in np.unique(clusters):
            c = int(c)
            if c == 0:
                continue
            if target_cluster is not None and target_cluster != c:
                continue

            points.append(
                target_locations[c][["Z", "Y", "X"]].drop(
                    columns=["level_0", "index"], errors="ignore"
                )
            )

        points = pd.concat(points)
        return points

    @thread_worker
    def save_worker(self, output_path):
        self.target_manger.save_to_disk(output_path)

    def _save_candidate_click(self):
        pth = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if pth is not None and pth != "":
            self.progressbar.setHidden(False)
            wsave = self.save_worker(pth)
            wsave.finished.connect(lambda: self.progressbar.hide())
            wsave.start()

    def _on_show_target_clicked(self):
        active_layer = self.plotter_widget.layers[0] if self.plotter_widget.layers else None
        if active_layer is None:
            return
        emb_pth = active_layer.metadata["tomotwin"][
            "embeddings_path"
        ]
        clusters = active_layer.features[
            "MANUAL_CLUSTER_ID"
        ]

        self.progressbar.setHidden(False)
        self.progressbar.set_label_text("Calculate target positions")

        target_cluster = self._cluster_dropdown.currentData(Qt.UserRole)

        ppe = futures.ProcessPoolExecutor(max_workers=1)

        f = ppe.submit(self.calc_targets, emb_pth, clusters, target_cluster)
        ppe.shutdown(wait=False)
        f.add_done_callback(self.target_calc_done.emit)

    def show_targets_callback(self, future: futures.Future):
        points: pd.DataFrame = future.result()
        point_colors = []

        c = self._cluster_dropdown.currentData(Qt.UserRole)

        # In napari-clusters-plotter 0.10+, use cat10_mod_cmap
        # Get the color from the colormap for the given index
        rgba = list(cat10_mod_cmap(c % 10))  # cat10 has 10 colors
        rgba[3] = 0.9  # Set alpha
        point_colors.append(rgba)

        self.delete_points_layer()

        self._target_point_layer = self.viewer.add_points(
            points,
            symbol="o",
            size=37,
            edge_color=point_colors,
            face_color="transparent",
            edge_width=0.10,
            out_of_slice_display=True,
            name="Targets",
        )

        self.viewer.dims.set_current_step(
            0, int(points[["Z"]].to_numpy()[0, 0])
        )
        self.viewer.window._qt_window.setEnabled(True)
        self.progressbar.setHidden(True)
        self.progressbar.set_label_text("")

    def _on_close_callback(self):
        self.cleanup()

    def _table_item_name_changed(self, item: QTableWidgetItem):
        self.tableWidget.itemChanged.disconnect(self._table_item_name_changed)

        all_names = []
        for r in range(self.tableWidget.rowCount()):
            if r == int(item.row()):
                continue
            lbl = self.tableWidget.item(r, 3).text()
            if lbl != "None":
                all_names.append(lbl)

        if self.tableWidgetHeaders[item.column()] == "Label":
            import re

            new_target_name = re.sub(
                "[^\d\w\-_]", "_", str(item.text()), flags=re.ASCII
            )
            if new_target_name in all_names:
                new_target_name = "None"
                notifications.show_error("Label already exists")
            item.setText(new_target_name)
            id_item = self.tableWidget.item(item.row(), 0)
            target = self.target_manger.get_target_by_id(int(id_item.text()))
            target.target_name = new_target_name

        self.tableWidget.itemChanged.connect(self._table_item_name_changed)

    def make_target(self, cluster_id):
        active_layer = self.plotter_widget.layers[0] if self.plotter_widget.layers else None
        if active_layer is None:
            return None
        embeddings_mask = (
            active_layer.features[
                "MANUAL_CLUSTER_ID"
            ]
            == cluster_id
        )
        embeddings_path = active_layer.metadata[
            "tomotwin"
        ]["embeddings_path"]
        layer = active_layer
        color = self.index_to_rgba(cluster_id)
        target = Target(
            embeddings_path, embeddings_mask, layer, cluster_id, color
        )
        return target

    def _on_add_candidate_clicked(self):
        active_layer = self.plotter_widget.layers[0] if self.plotter_widget.layers else None
        if active_layer is None:
            return

        c = self._cluster_dropdown.currentData(Qt.UserRole)
        target = self.make_target(c)

        target_added_succesfull = self.target_manger.add_target(target)
        if not target_added_succesfull:
            return
        current_row_count = self.tableWidget.rowCount()
        self.tableWidget.setRowCount(current_row_count + 1)
        self.added_canditates = self.added_canditates + 1

        entry = [
            f"{target.target_id}",
            "",
            active_layer.name,
            target.target_name,
        ]
        for col, value in enumerate(entry):
            item = QTableWidgetItem(value)
            if col == 0:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if col == 1:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                pixmap = QPixmap(10, 10)
                pixmap.fill(QColor(*self.index_to_rgba(c)))
                item.setData(Qt.DecorationRole, pixmap)
            if col == 2:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

            self.tableWidget.setItem(current_row_count, col, item)
        
        # Auto-increment the class spinbox to prepare for the next cluster selection
        # This ensures the next cluster gets a different color
        current_class = self.plotter_widget.plotting_widget.class_spinbox.value
        self.plotter_widget.plotting_widget.class_spinbox.value = current_class + 1

    def napari_update_umap(self, umap_embeddings, used_embeddings):

        self.tmp_dir_path = tempfile.mkdtemp()
        tmp_embed_pth = os.path.join(
            self.tmp_dir_path, next(tempfile._get_candidate_names())
        )
        used_embeddings.to_pickle(tmp_embed_pth)
        umap_embeddings.attrs["embeddings_path"] = tmp_embed_pth
        tmp_umap_pth = os.path.join(
            self.tmp_dir_path, next(tempfile._get_candidate_names())
        )
        umap_embeddings.to_pickle(tmp_umap_pth)

        # Visualizse it
        self.get_umap_tool().set_new_label_layer_name("UMAP Refined")
        worker = self.get_umap_tool().start_umap_worker(tmp_umap_pth)
        worker.start()
        # worker.returned.connect(self.update_all)

    def show_umap_callback(self, future: futures.Future):
        (umap_embeddings, used_embeddings) = future.result()
        self.viewer.window._qt_window.setEnabled(True)
        self.napari_update_umap(umap_embeddings, used_embeddings)
        self.progressbar.setHidden(True)
        self.progressbar.set_label_text("Remap")
        # Enable the Show Base UMAP button after remapping
        if self.base_umap_features is not None:
            self._show_base_umap.setEnabled(True)

    def _on_show_base_umap_click(self):
        """Restore the original base UMAP display."""
        if self.base_umap_features is None:
            notifications.show_info("Base UMAP data not available.")
            return
        
        active_layer = self.plotter_widget.layers[0] if self.plotter_widget.layers else None
        if active_layer is None:
            notifications.show_info("No layer selected.")
            return
        
        # Restore the original features and metadata
        active_layer.features = self.base_umap_features.copy()
        active_layer.metadata["tomotwin"] = self.base_umap_metadata.copy()
        
        # Replot
        self.replot_cluster_plotter()
        self.update_all()
        
        # Disable the button and clear stored data
        self._show_base_umap.setEnabled(False)
        self.base_umap_features = None
        self.base_umap_metadata = None

    @staticmethod
    def index_to_rgba(index: int) -> list[int]:
        # In napari-clusters-plotter 0.10+, use cat10_mod_cmap
        rgba_float = list(cat10_mod_cmap(index % 10))  # cat10 has 10 colors
        rgba = [int(v * 255) for v in rgba_float[:3]]
        rgba.append(int(255 * 0.9))
        return rgba

    def update_all(self):
        cls = []
        active_layer = self.plotter_widget.layers[0] if self.plotter_widget.layers else None
        if (
            active_layer is not None
            and hasattr(active_layer, "features")
            and "MANUAL_CLUSTER_ID"
            in active_layer.features
        ):
            cls = active_layer.features[
                "MANUAL_CLUSTER_ID"
            ]
        self.update_items_cluster_dropdown(self._cluster_dropdown, cls)

    def update_items_cluster_dropdown(
        self, dropdown: QComboBox, cluster_ids: list[int]
    ):

        dropdown.clear()
        # Convert to numeric to handle categorical dtype from napari-clusters-plotter
        unique_ids = np.unique(pd.to_numeric(cluster_ids, errors='coerce'))
        for c in unique_ids:

            if c <= 0:
                continue
            rgba = self.index_to_rgba(c)
            dropdown.addItem("")
            pixmap = QPixmap(10, 10)
            pixmap.fill(QColor(*rgba))
            dropdown.setItemData(
                dropdown.count() - 1, pixmap, Qt.DecorationRole
            )
            dropdown.setItemData(dropdown.count() - 1, c, Qt.UserRole)

    def _on_refine_click(self):
        self.viewer.window._qt_window.setEnabled(False)
        self.delete_points_layer()
        self.reestimate_umap()

    def get_current_cluster_dropdown(self):
        color_dropdown = QComboBox(self)
        color_dropdown.setSizeAdjustPolicy(
            QComboBox.AdjustToContentsOnFirstShow
        )
        return color_dropdown

    def reestimate_umap(self):
        active_layer = self.plotter_widget.layers[0] if self.plotter_widget.layers else None
        if active_layer is None:
            notifications.show_info(f"No layer selected. Can't refine.")
            return
        
        # Store the base UMAP features before remapping (only if not already stored)
        if self.base_umap_features is None:
            self.base_umap_features = active_layer.features.copy()
            self.base_umap_metadata = active_layer.metadata["tomotwin"].copy()
        
        try:
            print("Read clusters")
            clusters = active_layer.features[
                "MANUAL_CLUSTER_ID"
            ]
            # Convert to numeric to handle categorical dtype from napari-clusters-plotter
            clusters_numeric = pd.to_numeric(clusters, errors='coerce')
            if not np.any(clusters_numeric > 0):
                raise KeyError
        except KeyError:
            notifications.show_info(f"No cluster selected. Can't refine.")
            return

        def get_embedding_path(pth: str) -> str:
            """
            Checks if the embedding path exists. If it does not exist, it opens a file selection dialogue. Otherwise it returns the path.
            """
            if not os.path.exists(pth):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Can't open embedding file")
                msg.setText("Can't open embedding file")
                msg.setInformativeText(
                    "Embedding path in metadata data (see below) does not exist or can't be accesst. Please click OK and select the path to the embedding file."
                )
                msg.setDetailedText(pth)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                pth = QFileDialog.getOpenFileName(
                    self,
                    "Open embedding file",
                    os.getcwd(),
                    "Embedding file (*.temb)",
                )[0]

            return pth

        print("Read embeddings")
        emb_pth = get_embedding_path(
            active_layer.metadata["tomotwin"][
                "embeddings_path"
            ]
        )

        if emb_pth == "":
            print("No path selected.")
            return

        active_layer.metadata["tomotwin"][
            "embeddings_path"
        ] = emb_pth
        embeddings = pd.read_pickle(emb_pth)

        self.progressbar.setHidden(False)
        self.progressbar.set_label_text("Remap")

        # this workaround using signal is necessary, as "add_done_callback" starts the method
        # in a separate thread, but to change Qt elements, it must be run in the same thread as the main program.
        ppe = futures.ProcessPoolExecutor(max_workers=1)
        target_cluster = self._cluster_dropdown.currentData(Qt.UserRole)
        umap_neighbors = self.get_umap_tool().get_umap_neighbors()
        umap_metric = self.get_umap_tool().get_umap_metric()
        f = ppe.submit(urefine.refine, clusters, embeddings, target_cluster, umap_neighbors, umap_metric)
        ppe.shutdown(wait=False)
        f.add_done_callback(self.refinement_done.emit)
