from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtWidgets import (
    QPushButton,

    QVBoxLayout,
    QWidget,
)
import napari
import pathlib
from magicgui import magic_factory, tqdm
from napari_clusters_plotter._plotter import PlotterWidget
import pandas as pd
import numpy as np
from matplotlib.patches import Circle

circle: Circle = None
class LoadUmapWidget(QWidget):

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.napari_viewer = napari_viewer

        btn = QPushButton("Load umap!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(btn)

    def _load_umap(self):
        print("napari has", len(self.viewer.layers), "layers")

@magic_factory(
    call_button="Load",
    label_layer={'label': 'TomoTwin Label Mask:'},
    filename={'label': 'Path to UMAP:',
              'filter': '*.tumap'},
)
def load_umap_magic(
        label_layer: "napari.layers.Labels",
        filename=pathlib.Path('/some/path.tumap')
):
    umap = pd.read_pickle(filename)
    if "label" not in umap.keys().tolist():
        lbls = [int(l+1) for l,_ in enumerate(umap[['umap_1', 'umap_0']].itertuples(index=True, name='Pandas'))]
        label_column = pd.DataFrame(
            {"label": np.array(lbls)}
        )
        umap = pd.concat([label_column, umap], axis=1)

    if hasattr(label_layer, "properties"):
        label_layer.properties = umap
    if hasattr(label_layer, "features"):
        label_layer.features = umap
    label_layer.opacity = 0


    viewer = napari.current_viewer()
    plotter_widget: PlotterWidget = None
    widget, plotter_widget = viewer.window.add_plugin_dock_widget('napari-clusters-plotter', widget_name='Plotter Widget')
    plotter_widget.plot_x_axis.setCurrentIndex(1)
    plotter_widget.plot_y_axis.setCurrentIndex(2)

    plotter_widget.bin_auto.setChecked(True)
    plotter_widget.plotting_type.setCurrentIndex(1)
    plotter_widget.plot_hide_non_selected.setChecked(True)
    plotter_widget.run(
                umap,
                "umap_0",
                "umap_1",
                plot_cluster_name=None,
                force_redraw=True
            )

    @viewer.mouse_drag_callbacks.append
    def get_event(viewer, event):
        global circle
        data_coordinates = label_layer.world_to_data(event.position)
        val = label_layer._get_value(data_coordinates)
        umap_coordinates = umap.loc[umap['label']==val,['umap_0','umap_1']]

        center = umap_coordinates.values.tolist()[0]
        if circle is not None:
            circle.remove()
        circle = Circle(tuple(center), 0.5, fill=False, color='r')
        plotter_widget.graphics_widget.axes.add_patch(circle)
        plotter_widget.graphics_widget.draw_idle()



def create_embedding_mask(embeddings: pd.DataFrame):
    print("Create embedding mask")
    embeddings = embeddings.reset_index()
    Z = embeddings.attrs['tomogram_input_shape'][0]
    Y = embeddings.attrs['tomogram_input_shape'][1]
    X = embeddings.attrs['tomogram_input_shape'][2]

    segmentation_array = np.zeros(shape=(Z, Y, X),dtype=np.int64)
    label = 0
    for row in tqdm.tqdm(embeddings[['Z', 'Y', 'X']].itertuples(index=True, name='Pandas'), total=len(embeddings)-1):
        X = int(row.X)
        Y = int(row.Y)
        Z = int(row.Z)
        segmentation_array[(Z):(Z + 2), (Y):(Y + 2), (X):(X + 2)] = label + 1
        label = label+1

    return segmentation_array

@magic_factory(
    call_button="Create label mask",
    filename={'label': 'Path to embeddings file:',
              'filter': '*.temb'},
)
def make_label_mask(
        label_layer: "napari.layers.Labels",
        filename=pathlib.Path('/some/path.tumap')
):
    embeddings = pd.read_pickle(filename)
    segmentation_data = create_embedding_mask(embeddings=embeddings)
    viewer = napari.current_viewer()
    viewer.add_labels(segmentation_data, name='TomoTwin Label Mask')



@magic_factory(
    call_button="Load",
    label_layer={'label': 'TomoTwin Label Mask:'},
)
def save_clustering(
        label_layer: "napari.layers.Labels"):

    print(f"you have selected {label_layer}")
    print(label_layer.features)
    print("HIDDEN?", plotter_widget.isHidden())