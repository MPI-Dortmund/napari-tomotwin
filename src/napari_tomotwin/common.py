import napari
from pyqtspinner import WaitingSpinner
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor  # pylint: disable=E0611


def make_spinner():
    return WaitingSpinner(napari.current_viewer().window._qt_window,
                             True,
                             True, Qt.ApplicationModal,
                             color=QColor(255, 255, 255),
                             fade=60,
                             line_width=5,
                             line_length=15,
                             )