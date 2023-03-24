import napari
import pathlib
from magicgui import magic_factory, tqdm
from napari_clusters_plotter._plotter import PlotterWidget
import pandas as pd
import numpy as np
from matplotlib.patches import Circle

circle: Circle = None
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
    try:
        plotter_widget.run(
                    umap,
                    "umap_0",
                    "umap_1",
                    plot_cluster_name=None,
                    force_redraw=True
                )
    except:
        pass

    @viewer.mouse_drag_callbacks.append
    def get_event(viewer, event):
        global circle
        data_coordinates = label_layer.world_to_data(event.position)
        val = label_layer._get_value(data_coordinates)
        umap_coordinates = umap.loc[umap['label']==val,[plotter_widget.plot_x_axis.currentText(),plotter_widget.plot_y_axis.currentText()]]

        try:
            center = umap_coordinates.values.tolist()[0]
        except IndexError:
            return

        if circle is not None:
            circle.remove()
        circle = Circle(tuple(center), 0.5, fill=False, color='r')
        plotter_widget.graphics_widget.axes.add_patch(circle)
        plotter_widget.graphics_widget.draw_idle()



