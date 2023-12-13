import napari
import pathlib
from magicgui import magic_factory
from napari_clusters_plotter._plotter import PlotterWidget
import pandas as pd
import numpy as np
from matplotlib.patches import Circle
from napari.utils import notifications
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication # pylint: disable=E0611
from typing import List
from napari.qt.threading import thread_worker
from magicgui.tqdm import tqdm

plotter_widget: PlotterWidget = None
circles: List[Circle] = []
umap: pd.DataFrame
pbar = None

def _draw_circle(data_coordinates, label_layer, umap):
    global circles
    global plotter_widget

    label_layer.visible = 1

    val = label_layer._get_value(data_coordinates)

    umap_coordinates = umap.loc[
        umap['label'] == val, [plotter_widget.plot_x_axis.currentText(), plotter_widget.plot_y_axis.currentText()]]

    try:
        center = umap_coordinates.values.tolist()[0]
    except IndexError:
        return
    modifiers = QGuiApplication.keyboardModifiers()
    if modifiers == Qt.ShiftModifier:
        pass
    else:
        for c in circles[::-1]:
            c.remove()
        circles = []
    col = '#40d5aa'
    if plotter_widget.log_scale.isChecked():
        col = '#79abfd'
    circle = Circle(tuple(center), 0.5, fill=False, color=col)
    circles.append(circle)
    plotter_widget.graphics_widget.axes.add_patch(circle)
    plotter_widget.graphics_widget.draw_idle()


@thread_worker()
def run_clusters_plotter(plotter_widget,
                         features,
                         plot_x_axis_name,
                         plot_y_axis_name,
                         plot_cluster_name,
                         force_redraw):
    plotter_widget.run(features = features, plot_x_axis_name = plot_x_axis_name, plot_y_axis_name = plot_y_axis_name, plot_cluster_name = plot_cluster_name, force_redraw = force_redraw)

def show_umap(label_layer):
    global plotter_widget
    label_layer.opacity = 0
    label_layer.visible = True

    viewer = napari.current_viewer()

    @viewer.mouse_drag_callbacks.append
    def get_event(viewer, event):
        data_coordinates = label_layer.world_to_data(event.position)
        _draw_circle(data_coordinates, label_layer, umap)


    widget, plotter_widget = viewer.window.add_plugin_dock_widget('napari-clusters-plotter',
                                                                  widget_name='Plotter Widget')


    plotter_widget.plot_x_axis.setCurrentIndex(1)
    plotter_widget.plot_y_axis.setCurrentIndex(2)

    plotter_widget.bin_auto.setChecked(True)
    plotter_widget.plotting_type.setCurrentIndex(1)
    plotter_widget.plot_hide_non_selected.setChecked(True)
    plotter_widget.setDisabled(True)

    def activate_plotter_widget():
        plotter_widget.setEnabled(True)


    try:
        # Needs to run in a seperate thread, otherweise it freezes when it is loading the umap

        worker = run_clusters_plotter(plotter_widget,features=umap, plot_x_axis_name="umap_0",plot_y_axis_name="umap_1",plot_cluster_name=None,force_redraw=True)  # create "worker" object
        worker.returned.connect(activate_plotter_widget)
        worker.finished.connect(lambda: pbar.progressbar.hide())
        worker.finished.connect(lambda: napari.current_viewer().window._qt_window.setEnabled(True))
        worker.start()
    except:
        pass


@thread_worker
def _load_umap(filename: pathlib.Path, label_layer):
    global umap
    umap = pd.read_pickle(filename)
    if "label" not in umap.keys().tolist():
        lbls = np.arange(1,len(umap)+1,dtype=int)

        label_column = pd.DataFrame(
            {"label": lbls}
        )
        umap = pd.concat([label_column, umap], axis=1)


    if hasattr(label_layer, "properties"):
        label_layer.properties = umap
    if hasattr(label_layer, "features"):
        label_layer.features = umap


    return label_layer



def load_umap(label_layer: "napari.layers.Labels",
        filename: pathlib.Path):
    global umap
    global plotter_widget
    global pbar

    pbar = tqdm()

    napari.current_viewer().window._qt_window.setEnabled(False)
    worker = _load_umap(filename, label_layer=label_layer)
    worker.returned.connect(show_umap)
    return worker



@magic_factory(
    call_button="Load",
    label_layer={'label': 'TomoTwin Label Mask:'},
    filename={'label': 'Path to UMAP:',
              'filter': '*.tumap'},
)
def load_umap_magic(
        label_layer: "napari.layers.Labels",
        filename: pathlib.Path
):
    if label_layer == None:
        notifications.show_error("Label mask is not specificed")
        return

    if filename.suffix not in ['.tumap']:
        notifications.show_error("UMAP is not specificed")
        return


    worker = load_umap(label_layer, filename)
    worker.start()







