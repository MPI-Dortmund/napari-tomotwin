
import napari
from qtpy.QtGui import QGuiApplication
from qtpy.QtCore import Qt
from matplotlib.patches import Circle
from typing import List
from napari_clusters_plotter._plotter import PlotterWidget
import pandas as pd

class AnchorTool:
    def __init__(self,
                 plotting_widget: PlotterWidget,
                 umap: pd.DataFrame,
                 label_layer):
        self.plotter_widget = plotting_widget
        self.umap = umap
        self.circles: List[Circle] = []
        self.label_layer = label_layer

        def drag_event(viewer, event):
            data_coordinates = self.plotter_widget.layer_select.value.world_to_data(event.position)
            self._draw_circle(data_coordinates, self.plotter_widget.layer_select.value, self.plotter_widget.layer_select.value.features)
        print("LEN CALLBACKS", len(napari.current_viewer().mouse_drag_callbacks))
        napari.current_viewer().mouse_drag_callbacks.append(drag_event)

    def _draw_circle(self, data_coordinates, label_layer, umap):
        '''
        Adds a circle on the umap when you click on the image
        '''
        print("DRAW!")
        label_layer.visible = True
        val = label_layer._get_value(data_coordinates)

        umap_coordinates = umap.loc[
            umap['label'] == val, [self.plotter_widget.plot_x_axis.currentText(), self.plotter_widget.plot_y_axis.currentText()]]

        try:
            center = umap_coordinates.values.tolist()[0]
        except IndexError:
            print("ERROR!!! UAMP")
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