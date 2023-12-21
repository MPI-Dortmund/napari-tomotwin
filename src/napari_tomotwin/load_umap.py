import pathlib
from typing import List

import napari
import numpy as np
import pandas as pd
from magicgui import magic_factory
from magicgui.tqdm import tqdm
from matplotlib.patches import Circle
from napari.qt.threading import thread_worker
from napari.utils import notifications
from napari_clusters_plotter._plotter_utilities import estimate_number_bins
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication  # pylint: disable=E0611


class LoadUmapTool:

    def __init__(self):

        self.umap = None
        self.plotter_widget = None
        self.pbar = None
        self.circles: List[Circle] = []
        self.viewer = napari.current_viewer()

    def _draw_circle(self, data_coordinates, label_layer, umap):
        '''
        Adds a circle on the umap when you click on the image
        '''
        label_layer.visible = True
        val = label_layer._get_value(data_coordinates)

        umap_coordinates = umap.loc[
            umap['label'] == val, [self.plotter_widget.plot_x_axis.currentText(), self.plotter_widget.plot_y_axis.currentText()]]

        try:
            center = umap_coordinates.values.tolist()[0]
        except IndexError:
            return
        modifiers = QGuiApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            pass
        else:
            for c in self.circles[::-1]:
                c.remove()
            self.circles = []
        col = '#40d5aa'
        if self.plotter_widget.log_scale.isChecked():
            col = '#79abfd'
        circle = Circle(tuple(center), 0.5, fill=False, color=col)
        self.circles.append(circle)
        self.plotter_widget.graphics_widget.axes.add_patch(circle)
        self.plotter_widget.graphics_widget.draw_idle()

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
        self.pbar.progressbar.label = "Visualize umap"
        self.viewer.add_layer(label_layer)
        widget, self.plotter_widget = self.viewer.window.add_plugin_dock_widget('napari-clusters-plotter',
                                                                      widget_name='Plotter Widget',
                                                                      tabify=False)

        #label_layer = self.viewer.add_labels(lbl_data, name='Label layer', features=self.umap, properties=self.umap)

        label_layer.opacity = 0
        label_layer.visible = True

        @self.viewer.mouse_drag_callbacks.append
        def get_event(viewer, event):
            data_coordinates = label_layer.world_to_data(event.position)
            self._draw_circle(data_coordinates, label_layer, self.umap)

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

        def activate_plotter_widget():
            self.plotter_widget.setEnabled(True)

        try:
            # Needs to run in a seperate thread, otherweise it freezes when it is loading the umap

            worker = self.run_clusters_plotter(self.plotter_widget, features=self.umap, plot_x_axis_name="umap_0",
                                          plot_y_axis_name="umap_1", plot_cluster_name=None,
                                          force_redraw=True)  # create "worker" object
            worker.returned.connect(activate_plotter_widget)
            worker.finished.connect(lambda: self.pbar.progressbar.hide())
            worker.finished.connect(lambda: napari.current_viewer().window._qt_window.setEnabled(True))
            worker.start()

        except:
            pass

    def create_embedding_mask(self, umap: pd.DataFrame, values: np.array):
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
        for stride_x in tqdm(list(range(stride))):
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

    def load_umap(self, filename: pathlib.Path):
        if self.pbar is not None:
            self.pbar.progressbar.label = "Read umap"
        self.umap = pd.read_pickle(filename)
        if 'embeddings_attrs' not in self.umap.attrs:
            napari.utils.notifications.show_error(
                "The umap was calculated with an old version of TomoTwin. Please update TomoTwin and re-estimate the umap.")
            if self.pbar is not None:
                self.pbar.progressbar.hide()
            import sys
            sys.exit(1)
        if self.pbar is not None:
            self.pbar.progressbar.label = "Generate label layer"
        lbl_data = self.relabel_and_update()
        from napari.layers import Layer
        lbl_layer = Layer.create(lbl_data, {
            "name": "Label layer"}, layer_type="Labels")
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
        self.pbar = tqdm()

        napari.current_viewer().window._qt_window.setEnabled(False)
        worker = self._load_umap_worker(filename)
        worker.returned.connect(self.show_umap)

        return worker


def set_width(widget):
    widget.max_height= 100

@magic_factory(
    call_button="Load",
    filename={'label': 'Path to UMAP:',
              'filter': '*.tumap'},
    widget_init=set_width
)
def load_umap_magic(
        filename: pathlib.Path
):

    if filename.suffix not in ['.tumap']:
        notifications.show_error("UMAP is not specificed")
        return
    tool = LoadUmapTool()
    worker = tool.start_umap_worker(filename)
    worker.start()







