import numpy as np
from PyQt5.QtWidgets import QComboBox, QStyledItemDelegate, QFrame
from qtpy.QtWidgets import (
    QFormLayout,
    QPushButton,
    QWidget,
    QFileDialog,
    QMessageBox,
    QLabel,
    QHBoxLayout,
    QLineEdit,
    QListView,
    QTableWidgetItem,
    QTableWidget,
    QHeaderView,
    QLayout
)
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QColor, QPixmap
from qtpy.QtCore import Qt
from napari_clusters_plotter._plotter import PlotterWidget
from napari_clusters_plotter._utilities import get_nice_colormap

class ColorItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # Draw a colored rectangle in the item
        color = index.data(Qt.UserRole)
        if color is not None and isinstance(color, QColor):
            painter.fillRect(option.rect, color)

    def sizeHint(self, option, index):
        # Set the size of the item
        return index.data(Qt.SizeHintRole) if index.data(Qt.SizeHintRole) else super().sizeHint(option, index)

class ColorComboBox(QComboBox):
    def __init__(self):
        super().__init__()

        # Set the custom item delegate
        delegate = ColorItemDelegate(self)
        self.setItemDelegate(delegate)

        # Set the combo box to only display icons
        self.setView(QListView(self))

class ClusteringWidgetQt(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()

        self.viewer = napari_viewer
        self.plotter_widget: PlotterWidget


        #######
        # UI Setup
        ######
        layout = QFormLayout()
        #app = QApplication.instance()
        #app.lastWindowClosed.connect(self.on_close_callback)  # this line is connection to signal
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.setLayout(layout)

        ####
        recalc_layout = QHBoxLayout()
        self._recalc_umap = QPushButton("Recalculate UMAP", self)
        self._color_dropdown = self.get_current_cluster_dropdown()
        self._show_targets = QPushButton("Show targets", self)



        recalc_layout.addWidget(self._color_dropdown)
        recalc_layout.addWidget(self._show_targets)
        recalc_layout.addWidget(self._recalc_umap)

        self.layout().addRow("", recalc_layout)




        # Add a horizontal line
        horizontal_line = QFrame(self)
        horizontal_line.setFrameShape(QFrame.HLine)
        horizontal_line.setFrameShadow(QFrame.Sunken)

        self.layout().addWidget(horizontal_line)

        ## Now q table widget
        candlabl = QLabel("Candidates:")
        self.layout().addWidget(candlabl)
        self.tableWidget = QTableWidget(self)
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setHorizontalHeaderLabels(["", "Color", "UMAP", "Description"])
        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)
        self.layout().addWidget(self.tableWidget)

        ## Add candidate buttons
        add_candidiate_layout = QHBoxLayout()
        self._add_candidate = QPushButton("Add candidate", self)
        self._add_candidate.clicked.connect(self.add_candidate)
        self._candidate_dropdown = self.get_current_cluster_dropdown()
        add_candidiate_layout.addWidget(self._candidate_dropdown)
        add_candidiate_layout.addWidget(self._add_candidate)
        self.layout().addRow("", add_candidiate_layout)

        ## Save and delete

        save_delete_layout = QHBoxLayout()
        self.save = QPushButton("Save candidates", self)
        self.save.clicked.connect(self.update_all)
        self.delete = QPushButton("Delete candidates", self)
        save_delete_layout.addWidget(self.delete)
        save_delete_layout.addWidget(self.save)
        self.layout().addRow("", save_delete_layout)

    def add_candidate(self):

        current_row_count = self.tableWidget.rowCount()
        self.tableWidget.setRowCount(current_row_count + 1)

        c = self._candidate_dropdown.currentIndex()+1

        entry = ["","",self.plotter_widget.layer_select.value.name, "Lorem ipsum"]
        for col, value in enumerate(entry):
            item = QTableWidgetItem(value)
            if col == 0:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
            if col == 1:
                pixmap = QPixmap(10, 10)
                pixmap.fill(QColor(*self.index_to_rgba(c)))
                item.setData(Qt.DecorationRole, pixmap)

            self.tableWidget.setItem(current_row_count, col, item)

    @staticmethod
    def index_to_rgba(index: int) -> list[int]:
        colors = get_nice_colormap()
        import PIL.ImageColor as ImageColor
        rgba = [int(v)
                for v in list(
                ImageColor.getcolor(colors[index % len(colors)], "RGB")
            )
                ]
        rgba.append(int(255 * 0.9))
        return rgba
    def update_all(self):

        self.update_items_cluster_dropdown(self._color_dropdown, self.plotter_widget.layer_select.value.features['MANUAL_CLUSTER_ID'])
        self.update_items_cluster_dropdown(self._candidate_dropdown,
                                           self.plotter_widget.layer_select.value.features['MANUAL_CLUSTER_ID'])

    def update_items_cluster_dropdown(self, dropdown : QComboBox, cluster_ids: list[int]):

        dropdown.clear()
        for c in np.unique(cluster_ids):
            if c == 0:
                continue
            rgba = self.index_to_rgba(c)
            dropdown.addItem("")
            pixmap = QPixmap(10, 10)
            pixmap.fill(QColor(*rgba))
            dropdown.setItemData(dropdown.count()-1, pixmap, Qt.DecorationRole)



    def set_plotter_widget(self, plotter_widget: PlotterWidget):
        self.plotter_widget = plotter_widget

    def get_current_cluster_dropdown(self):
        color_dropdown = QComboBox(self)
        color_dropdown.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)

        color_dropdown.addItem("")
        color_dropdown.addItem("")
        color_dropdown.addItem("")

        pixmap = QPixmap(10, 10)
        pixmap.fill(QColor(Qt.black))
        color_dropdown.setItemData(0, pixmap, Qt.DecorationRole)

        pixmap = QPixmap(10, 10)
        pixmap.fill(QColor(Qt.green))
        color_dropdown.setItemData(1, pixmap, Qt.DecorationRole)

        pixmap = QPixmap(10, 10)
        pixmap.fill(QColor(Qt.red))
        color_dropdown.setItemData(2, pixmap, Qt.DecorationRole)
        return color_dropdown