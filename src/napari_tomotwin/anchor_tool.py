from typing import List

from matplotlib.patches import Circle
from napari_clusters_plotter._plotter import PlotterWidget
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication

circles: List[Circle] = []

def _draw_circle(
        plotter_widget: PlotterWidget,
        data_coordinates,
        label_layer,
        umap):
    global circles
    '''
    Adds a circle on the umap when you click on the image
    '''
    label_layer.visible = True
    val = label_layer._get_value(data_coordinates)

    umap_coordinates = umap.loc[
        umap['label'] == val, [plotter_widget.plot_x_axis.currentText(), plotter_widget.plot_y_axis.currentText()]]

    try:
        center = umap_coordinates.values.tolist()[0]
    except IndexError:
        print("ERROR!!! UAMP")
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

def drag_circle_callback(plotter_widget, viewer, event):
    data_coordinates = plotter_widget.layer_select.value.world_to_data(event.position)
    _draw_circle(plotter_widget, data_coordinates, plotter_widget.layer_select.value, plotter_widget.layer_select.value.features)
