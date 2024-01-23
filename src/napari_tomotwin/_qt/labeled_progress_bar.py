from qtpy.QtWidgets import (
    QProgressBar,
    QLabel,
)
class LabeledProgressBar(QProgressBar):
    def __init__(self, label: QLabel):
        super().__init__()
        self.label = label

    def set_label_text(self, text: str):
        self.label.setText(text)

    def hide(self):
        super().hide()
        self.label.hide()

    def setHidden(self, hidden):
        super().setHidden(hidden)
        self.label.setHidden(hidden)