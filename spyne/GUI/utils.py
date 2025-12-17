""" Created on Wed Jun 25 14:49:00 2025
    @author: dcupolillo """

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QAction, QMenu, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QPushButton, QApplication, QWidget)
from PyQt5.QtCore import pyqtSignal
from pyqtspinner import WaitingSpinner


class KernelDialog(QDialog):

    kernel_selected = pyqtSignal(tuple)

    def __init__(
            self,
            parent: QMainWindow = None,
            kernel: tuple = (3, 3, 3),
    ) -> None:

        super().__init__(parent)
        self.setWindowTitle("Select Kernel (x, y, t)")

        layout = QVBoxLayout()

        self.kernel_x = QSpinBox()
        self.kernel_x.setRange(1, 99)
        self.kernel_x.setValue(kernel[0])

        self.kernel_y = QSpinBox()
        self.kernel_y.setRange(1, 99)
        self.kernel_y.setValue(kernel[1])

        self.kernel_t = QSpinBox()
        self.kernel_t.setRange(1, 99)
        self.kernel_t.setValue(kernel[2])

        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("X : "))
        kernel_layout.addWidget(self.kernel_x)
        kernel_layout.addWidget(QLabel("Y : "))
        kernel_layout.addWidget(self.kernel_y)
        kernel_layout.addWidget(QLabel("t : "))
        kernel_layout.addWidget(self.kernel_t)

        layout.addLayout(kernel_layout)

        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self._emit_kernel_and_accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)

        layout.addLayout(button_box)
        self.setLayout(layout)

    def _emit_kernel_and_accept(self):
        kernel = (
            self.kernel_x.value(),
            self.kernel_y.value(),
            self.kernel_t.value(),
        )
        self.kernel_selected.emit(kernel)
        self.accept()
