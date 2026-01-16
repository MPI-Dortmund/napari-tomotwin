import os
import pathlib
from functools import partial
from typing import List

import napari
import numpy as np
import pandas as pd
from magicgui.tqdm import tqdm as mtqdm
from matplotlib.patches import Circle
from napari.qt.threading import thread_worker
from napari.utils import notifications
from napari_tomotwin.anchor_tool import drag_circle_callback
from qtpy.QtWidgets import (
    QFileDialog,
    QMessageBox,
)


def estimate_number_bins(data) -> int:
    """
    Estimates number of bins according Freedman–Diaconis rule.
    
    This function was previously from napari_clusters_plotter._plotter_utilities
    but is now included locally for compatibility with napari-clusters-plotter 0.10+.
    """
    from scipy.stats import iqr as scipy_iqr

    est_a = (np.max(data) - np.min(data)) / (
        2 * scipy_iqr(data) / np.cbrt(len(data))
    )
    if np.isnan(est_a):
        return 256
    return int(est_a)


class LoadUmapTool:

    def __init__(self, plotter_widget=None):

        self.umap = None
        self.plotter_widget = plotter_widget
        self.pbar = None
        self.circles: List[Circle] = []
        self.viewer = napari.current_viewer()
        self.label_layer_name: str = "Label layer"
        self.viewer.mouse_drag_callbacks.append(
            partial(drag_circle_callback, self.plotter_widget)
        )
        self.created_layers = []

    def set_progressbar(self, pbar):
        self.pbar = pbar

    def set_new_label_layer_name(self, name: str):
        self.label_layer_name = name

    def update_progress_bar(self, text: str) -> None:
        try:
            if self.pbar is not None:
                self.pbar.set_label_text(text)
        except AttributeError:
            print("Can't initialize progress bar")

    def hide_progress_bar(self) -> None:
        try:
            self.pbar.setHidden(True)
        except AttributeError:
            print("Can't hide progress bar. Not initialized")

    # Note: run_clusters_plotter was removed in napari-clusters-plotter 0.10+
    # The plotter now uses plot_needs_update.emit() instead of run()

    def show_umap(self, label_layer):

        if label_layer is None:
            self.viewer.window._qt_window.setEnabled(True)

            notifications.show_error("Can't load umap")

        valid = self.check_umap_metadata()

        if not valid:
            if self.pbar is not None:
                self.pbar.hide()
            self.viewer.window._qt_window.setEnabled(True)
            return

        label_layer.metadata["tomotwin"]["embeddings_path"] = self.umap.attrs[
            "embeddings_path"
        ]  # might have been updated while checking umap metadata

        self.update_progress_bar("Visualize umap")
        self.viewer.add_layer(label_layer)
        self.created_layers.append(label_layer)

        try:
            # napari-clusters-plotter 0.10+
            label_layer.opacity = 0
            label_layer.visible = True
            # Select the layer via napari's layer selection mechanism
            self.viewer.layers.selection.active = label_layer
        except Exception as e:
            print(f"ERROR: {e}")
            pass

        # Determine plotting type based on embedding mode
        # COORDS mode uses SCATTER (0), sliding window uses HISTOGRAM2D (1)
        use_scatter = False
        try:
            mode = self.umap.attrs["embeddings_attrs"]["mode"]
            print(f"DEBUG: embeddings_attrs mode = {mode}")
            if mode == "COORDS":
                use_scatter = True
        except KeyError:
            print("Old Embedding file detected. Assuming sliding window data.")
            pass
        print(f"DEBUG: use_scatter = {use_scatter}")

        # Set properties using the new 0.10 API
        # Block plot_needs_update signal to prevent premature replotting
        self.plotter_widget.plot_needs_update.disconnect(self.plotter_widget._replot)
        
        # Set plotting_type FIRST to ensure correct active_artist
        plot_type = "SCATTER" if use_scatter else "HISTOGRAM2D"
        print(f"DEBUG: Setting plot_type to {plot_type}")
        self.plotter_widget.control_widget.plot_type_box.setCurrentText(plot_type)
        self.plotter_widget._on_plot_type_changed()
        print(f"DEBUG: After _on_plot_type_changed, active_artist type: {type(self.plotter_widget.plotting_widget.active_artist).__name__}")
        print(f"DEBUG: plotting_type property: {self.plotter_widget.plotting_type}")
        
        # Now set axes (these would normally trigger replot, but we disconnected it)
        self.plotter_widget.control_widget.x_axis_box.setCurrentText("umap_0")
        self.plotter_widget.control_widget.y_axis_box.setCurrentText("umap_1")
        self.plotter_widget.automatic_bins = not use_scatter
        self.plotter_widget.hide_non_selected = True
        
        # Reconnect the signal
        self.plotter_widget.plot_needs_update.connect(self.plotter_widget._replot)
        self.plotter_widget.setDisabled(True)

        try:
            # In napari-clusters-plotter 0.10+, just emit the update signal
            self.plotter_widget.plot_needs_update.emit()
            self.plotter_widget.setEnabled(True)
            self.hide_progress_bar()
            napari.current_viewer().window._qt_window.setEnabled(True)

        except Exception as e:
            print(f"Error updating plot: {e}")
            notifications.show_error("Can't load umap")
            self.hide_progress_bar()
            napari.current_viewer().window._qt_window.setEnabled(True)
            pass

    def get_created_layers(self) -> List[any]:
        return self.created_layers

    def create_embedding_mask(
        self, umap: pd.DataFrame, values: np.array
    ) -> np.array:
        """
        Creates mask where each individual subvolume of the running windows gets an individual ID
        """
        print("Create embedding mask")
        Z = umap.attrs["embeddings_attrs"]["tomogram_input_shape"][0]
        Y = umap.attrs["embeddings_attrs"]["tomogram_input_shape"][1]
        X = umap.attrs["embeddings_attrs"]["tomogram_input_shape"][2]

        stride = umap.attrs["embeddings_attrs"]["stride"]
        stride = stride[0] if stride is not None else 10 # in this case coords were embedded
        segmentation_array = np.zeros(shape=(Z, Y, X), dtype=np.float32)
        z = np.array(umap["Z"], dtype=int)
        y = np.array(umap["Y"], dtype=int)
        x = np.array(umap["X"], dtype=int)

        iscoords = False
        try:
            iscoords = umap.attrs["embeddings_attrs"]["mode"] == "COORDS"
        except KeyError:
            print("Old Embedding file detected. Assuming sliding window data.")
            pass

        # values = np.array(range(1, len(x) + 1))
        for stride_x in mtqdm(list(range(-stride,stride))):
            for stride_y in range(-stride,stride):
                for stride_z in range(-stride,stride):
                    if iscoords:
                        if stride_x ** 2 + stride_y ** 2 + stride_z ** 2 > stride ** 2:
                            continue
                    index = (z + stride_z, y + stride_y, x + stride_x)
                    segmentation_array[index] = values

        return segmentation_array

    def relabel_and_update(self):
        """
        Here I reduce the number of labels according the histogram bins. This is only for speed reasons.
        """
        print("Relabel")
        nbins = np.max(
            [
                estimate_number_bins(self.umap["umap_0"]),
                estimate_number_bins(self.umap["umap_1"]),
            ]
        )
        print(f"Number of bins: {nbins}")

        h, xedges, yedges = np.histogram2d(
            self.umap["umap_0"], self.umap["umap_1"], bins=nbins
        )
        xbins = np.digitize(self.umap["umap_0"], xedges)
        ybins = np.digitize(self.umap["umap_1"], yedges)
        new_lbl = xbins * h.shape[0] + ybins
        if "label" not in self.umap.keys().tolist():
            self.umap["label"] = new_lbl
        lbl_data = self.create_embedding_mask(self.umap, new_lbl).astype(
            np.int64
        )
        return lbl_data

    def check_umap_metadata(self) -> bool:
        def get_embedding_path(pth: str) -> str:
            """
            Checks if the embedding path exists. If it does not exist, it opens a file selection dialogue. Otherwise, it returns the path.
            """
            if not os.path.exists(pth):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Can't open embedding file")
                msg.setText("Can't open embedding file")
                msg.setInformativeText(
                    "The embedding path in the metadata (see below) doesn't exist or can't be accessed, click OK and select the path to the embedding file."
                )
                msg.setDetailedText(pth)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                pth = QFileDialog.getOpenFileName(
                    napari.current_viewer().window._qt_window,
                    "Open embedding file",
                    os.getcwd(),
                    "Embedding file (*.temb)",
                )[0]

            return pth

        if "embeddings_attrs" not in self.umap.attrs:
            napari.utils.notifications.show_error(
                "The umap was calculated with an old version of TomoTwin. Please update TomoTwin and re-estimate the umap."
            )
            return False

        emb_path = get_embedding_path(self.umap.attrs["embeddings_path"])

        if emb_path == "":
            return False

        self.umap.attrs["embeddings_path"] = (
            emb_path  # overwrite in case it was updated
        )

        return True

    def get_umap_metric(self) -> str:
        return self.umap.attrs.get("umap_metric","euclidean")

    def get_umap_mode(self) -> str:
        return self.umap.attrs["embeddings_attrs"]["mode"]

    def get_umap_neighbors(self) -> int:
        try:
            return self.umap.attrs["umap_neighbors"]
        except KeyError:
            return 200

    def load_umap(self, filename: pathlib.Path):
        self.update_progress_bar("Read umap")
        self.umap = pd.read_pickle(filename)
        self.update_progress_bar("Generate label layer")

        if "umap_0" not in self.umap:
            return None

        lbl_data = self.relabel_and_update()
        from napari.layers import Layer

        lbl_layer = Layer.create(
            lbl_data, {"name": self.label_layer_name}, layer_type="Labels"
        )
        lbl_layer.features = self.umap
        lbl_layer.properties = self.umap
        lbl_layer.metadata["tomotwin"] = {
            "umap_path": filename,
            "embeddings_path": self.umap.attrs["embeddings_path"],
        }

        return lbl_layer

    @thread_worker
    def _load_umap_worker(self, filename: pathlib.Path):
        return self.load_umap(filename)

    def start_umap_worker(self, filename: pathlib.Path):
        napari.current_viewer().window._qt_window.setEnabled(False)
        worker = self._load_umap_worker(filename)
        worker.returned.connect(self.show_umap)

        return worker
