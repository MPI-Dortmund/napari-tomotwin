from typing import List

from matplotlib.patches import Circle
from napari_clusters_plotter import PlotterWidget
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication

circles: List[Circle] = []


def _draw_circle(
    plotter_widget: PlotterWidget, data_coordinates, label_layer, umap
):
    global circles
    """
    Adds a circle on the umap when you click on the image
    """
    label_layer.visible = True
    val = label_layer._get_value(data_coordinates)

    umap_coordinates = umap.loc[
        umap["label"] == val,
        [
            plotter_widget.x_axis,
            plotter_widget.y_axis,
        ],
    ]

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
    col = "#40d5aa"
    if plotter_widget.log_scale:
        col = "#79abfd"
    circle = Circle(tuple(center), 0.5, fill=False, color=col)
    circles.append(circle)
    plotter_widget.plotting_widget.axes.add_patch(circle)
    plotter_widget.plotting_widget.draw_idle()


def _get_active_layer(plotter_widget: PlotterWidget):
    """Helper to get the active layer from the plotter widget."""
    if len(plotter_widget.layers) > 0:
        return plotter_widget.layers[0]
    return None


def drag_circle_callback(plotter_widget, viewer, event):
    layer = _get_active_layer(plotter_widget)
    if layer is None:
        return
    data_coordinates = layer.world_to_data(event.position)
    _draw_circle(
        plotter_widget,
        data_coordinates,
        layer,
        layer.features,
    )
