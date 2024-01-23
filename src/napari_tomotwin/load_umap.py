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
from napari_clusters_plotter._plotter_utilities import estimate_number_bins
from napari_tomotwin._qt.labeled_progress_bar import LabeledProgressBar
from napari_tomotwin.anchor_tool import drag_circle_callback
from qtpy.QtWidgets import (
    QFileDialog,
    QMessageBox,

)


class LoadUmapTool:

    def __init__(self, pbar: LabeledProgressBar, plotter_widget = None):

        self.umap = None
        self.plotter_widget = plotter_widget
        self.pbar = pbar
        self.circles: List[Circle] = []
        self.viewer = napari.current_viewer()
        self.label_layer_name: str = "Label layer"
        self.viewer.mouse_drag_callbacks.append(partial(drag_circle_callback, self.plotter_widget))
        self.created_layers = []

    def set_new_label_layer_name(self, name: str):
        self.label_layer_name = name

    def update_progress_bar(self,text: str) -> None:
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



    @thread_worker()
    def run_clusters_plotter(self, plotter_widget,
                             features,
                             plot_x_axis_name,
                             plot_y_axis_name,
                             plot_cluster_name,
                             force_redraw):
        '''
        Wrapper function to run clusters plotter in a seperate thead.
        '''
        plotter_widget.run(features=features, plot_x_axis_name=plot_x_axis_name, plot_y_axis_name=plot_y_axis_name,
                           plot_cluster_name=plot_cluster_name, force_redraw=force_redraw)


    def show_umap(self, label_layer):

        valid = self.check_umap_metadata()

        if not valid:
            if self.pbar is not None:
                self.pbar.hide()
            import sys
            sys.exit(1)

        label_layer.metadata['tomotwin']["embeddings_path"] = self.umap.attrs['embeddings_path'] #might have been updated while checking umap metadata

        self.update_progress_bar("Visualize umap")
        self.viewer.add_layer(label_layer)

        label_layer.opacity = 0
        label_layer.visible = True
        self.created_layers.append(label_layer)


        try:
            # napari-clusters-plotter > 0.7.4
            self.plotter_widget.layer_select.value = label_layer
        except:
            # napari-clusters-plotter < 0.7.4
            pass

        self.plotter_widget.plot_x_axis.setCurrentIndex(3)
        self.plotter_widget.plot_y_axis.setCurrentIndex(4)
        self.plotter_widget.bin_auto.setChecked(True)
        self.plotter_widget.plotting_type.setCurrentIndex(1)
        self.plotter_widget.plot_hide_non_selected.setChecked(True)
        self.plotter_widget.setDisabled(True)

        try:
            # Needs to run in a separate thread, otherwise it freezes when it is loading the umap
            worker = self.run_clusters_plotter(self.plotter_widget, features=self.umap, plot_x_axis_name="umap_0",
                                          plot_y_axis_name="umap_1", plot_cluster_name=None,
                                          force_redraw=True)  # create "worker" object
            worker.returned.connect(lambda x: self.plotter_widget.setEnabled(True))
            worker.finished.connect(self.hide_progress_bar)
            worker.finished.connect(lambda: napari.current_viewer().window._qt_window.setEnabled(True))
            worker.start()

        except:
            pass

    def get_created_layers(self) -> List[any]:
        return self.created_layers

    def create_embedding_mask(self, umap: pd.DataFrame, values: np.array) -> np.array:
        """
        Creates mask where each individual subvolume of the running windows gets an individual ID
        """
        print("Create embedding mask")
        Z = umap.attrs['embeddings_attrs']["tomogram_input_shape"][0]
        Y = umap.attrs['embeddings_attrs']["tomogram_input_shape"][1]
        X = umap.attrs['embeddings_attrs']["tomogram_input_shape"][2]
        stride = umap.attrs['embeddings_attrs']["stride"][0]
        segmentation_array = np.zeros(shape=(Z, Y, X), dtype=np.float32)
        z = np.array(umap["Z"], dtype=int)
        y = np.array(umap["Y"], dtype=int)
        x = np.array(umap["X"], dtype=int)

        # values = np.array(range(1, len(x) + 1))
        for stride_x in mtqdm(list(range(stride))):
            for stride_y in range(stride):
                for stride_z in range(stride):
                    index = (z + stride_z, y + stride_y, x + stride_x)
                    segmentation_array[index] = values

        return segmentation_array

    def relabel_and_update(self):
        '''
        Here I reduce the number of labels according the histogram bins. This is only for speed reasons.
        '''
        print("Relabel")
        nbins = np.max([estimate_number_bins(self.umap['umap_0']), estimate_number_bins(self.umap['umap_1'])])
        h, xedges, yedges = np.histogram2d(self.umap['umap_0'], self.umap['umap_1'], bins=nbins)
        xbins = np.digitize(self.umap['umap_0'], xedges)
        ybins = np.digitize(self.umap['umap_1'], yedges)
        new_lbl = xbins * h.shape[0] + ybins
        if "label" not in self.umap.keys().tolist():
            self.umap['label'] = new_lbl
        lbl_data = self.create_embedding_mask(self.umap, new_lbl).astype(np.int64)
        return lbl_data

    def check_umap_metadata(self) -> bool:
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
                    "The embedding path in the metadata (see below) doesn't exist or can't be accessed, click OK and select the path to the embedding file.")
                msg.setDetailedText(pth)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                pth = QFileDialog.getOpenFileName(napari.current_viewer().window._qt_window, 'Open embedding file',
                                                  os.getcwd(),
                                                  "Embedding file (*.temb)")[0]

            return pth

        if 'embeddings_attrs' not in self.umap.attrs:
            napari.utils.notifications.show_error(
                "The umap was calculated with an old version of TomoTwin. Please update TomoTwin and re-estimate the umap.")
            return False

        emb_path = get_embedding_path(self.umap.attrs['embeddings_path'])

        if emb_path == "":
            return False

        self.umap.attrs['embeddings_path'] = emb_path # overwrite in case it was updated

        return True


    def load_umap(self, filename: pathlib.Path):
        self.update_progress_bar("Read umap")
        self.umap = pd.read_pickle(filename)
        self.update_progress_bar("Generate label layer")


        lbl_data = self.relabel_and_update()
        from napari.layers import Layer
        lbl_layer = Layer.create(lbl_data, {
            "name": self.label_layer_name}, layer_type="Labels")
        lbl_layer.features = self.umap
        lbl_layer.properties = self.umap
        lbl_layer.metadata['tomotwin'] = {
            "umap_path": filename,
            "embeddings_path": self.umap.attrs['embeddings_path']
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









