import napari
import pathlib
from magicgui import magic_factory
from napari_clusters_plotter._plotter import PlotterWidget
from napari_clusters_plotter._plotter_utilities import estimate_number_bins
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

def show_umap(map_and_lbl_data):


    global plotter_widget

    viewer = napari.current_viewer()

    widget, plotter_widget = viewer.window.add_plugin_dock_widget('napari-clusters-plotter',
                                                                  widget_name='Plotter Widget',
                                                                  tabify=False)
    (umap, lbl_data) = map_and_lbl_data


    label_layer = viewer.add_labels(lbl_data, name='Label layer', features=umap,properties=umap)

    label_layer.opacity = 0.5
    label_layer.visible = False

    @viewer.mouse_drag_callbacks.append
    def get_event(viewer, event):
        data_coordinates = label_layer.world_to_data(event.position)
        _draw_circle(data_coordinates, label_layer, umap)



    def findDataIndex(combo, data):
        for index in range(combo.count()):
            print(combo.itemData(index), type(combo.itemData(index)))
            if combo.itemData(index) == data:
                return index
        return -1


    try:
        # napari-clusters-plotter > 0.7.4
        plotter_widget.layer_select.value = label_layer
    except:
        # napari-clusters-plotter < 0.7.4
        pass

    plotter_widget.plot_x_axis.setCurrentIndex(3)
    plotter_widget.plot_y_axis.setCurrentIndex(4)
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

def create_embedding_mask(umap: pd.DataFrame, values: np.array):
    """
    Creates mask where each individual subvolume of the running windows gets an individual ID
    """
    print("Create embedding mask")
    Z = umap.attrs["tomogram_input_shape"][0]
    Y = umap.attrs["tomogram_input_shape"][1]
    X = umap.attrs["tomogram_input_shape"][2]
    stride = umap.attrs["stride"][0]
    segmentation_array = np.zeros(shape=(Z, Y, X), dtype=np.float32)
    z = np.array(umap["Z"], dtype=int)
    y = np.array(umap["Y"], dtype=int)
    x = np.array(umap["X"], dtype=int)

    #values = np.array(range(1, len(x) + 1))
    for stride_x in tqdm(list(range(stride))):
        for stride_y in range(stride):
            for stride_z in range(stride):
                index = (z + stride_z, y + stride_y, x + stride_x)
                segmentation_array[index] = values

    return segmentation_array

def relabel_and_update(umap):
    print("Relabel")
    nbins = np.max([estimate_number_bins(umap['umap_0']), estimate_number_bins(umap['umap_1'])])
    h, xedges, yedges = np.histogram2d(umap['umap_0'], umap['umap_1'], bins=nbins)
    xbins = np.digitize(umap['umap_0'], xedges)
    ybins = np.digitize(umap['umap_1'], yedges)
    new_lbl= xbins*h.shape[0]+ybins
    if "label" not in umap.keys().tolist():
        umap['label']= new_lbl
    return create_embedding_mask(umap,new_lbl).astype(np.int64)

@thread_worker
def _load_umap(filename: pathlib.Path):
    global umap
    umap = pd.read_pickle(filename)
    lbl_data = relabel_and_update(umap)

    return umap, lbl_data



def load_umap(filename: pathlib.Path):
    global umap
    global plotter_widget
    global pbar

    pbar = tqdm()

    napari.current_viewer().window._qt_window.setEnabled(False)
    worker = _load_umap(filename)
    worker.returned.connect(show_umap)
    return worker



@magic_factory(
    call_button="Load",
    filename={'label': 'Path to UMAP:',
              'filter': '*.tumap'},
)
def load_umap_magic(
        filename: pathlib.Path
):

    if filename.suffix not in ['.tumap']:
        notifications.show_error("UMAP is not specificed")
        return


    worker = load_umap(filename)
    worker.start()







