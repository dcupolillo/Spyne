""" Created on Wed Jun 25 15:22:26 2025
    @author: dcupolillo """

from spyne.core.imagingdataset import ImagingDataset
from PyQt5.QtWidgets import (
    QMainWindow, QFrame, QLabel, QGridLayout,
    QSlider, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal


class RoiPanel(QFrame):

    roi_n_signal = pyqtSignal(int)
    sweep_n_signal = pyqtSignal(int)

    def __init__(self, parent: QMainWindow) -> None:

        super().__init__(parent)

        self.dataset = None
        self.current_roi_n = None
        self.current_sweep_n = None

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        for row in [0, 7]:
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            self.layout.addWidget(separator, row, 0, 1, 3)

        title = QLabel("ROI")
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title, 1, 0, 1, 2)

        description = QLabel("Select a ROI and a sweep using the sliders:")
        self.layout.addWidget(description, 2, 0, 1, 2)

        self.roi_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.roi_slider.setEnabled(False)
        self.layout.addWidget(self.roi_slider, 3, 0, 1, 2)

        self.current_roi_label = QLabel("Current Roi: ", self)
        self.layout.addWidget(self.current_roi_label, 4, 0, 1, 2)

        self.sweep_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.sweep_slider.setEnabled(False)
        self.layout.addWidget(self.sweep_slider, 5, 0, 1, 2)

        self.current_sweep_label = QLabel("Current sweep: ", self)
        self.layout.addWidget(self.current_sweep_label, 6, 0, 1, 2)

        self._roi_metadata_summary()

    def _roi_metadata_summary(self) -> None:

        title = QLabel("ROI properties")
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title, 8, 0, 1, 2)

        labels = [
            "Roi index",
            "Sweep index",
            "Roi size (pixel)",
            "Roi size (µm)",
            "Z index",
            "Branch degree",
            "Branch ID",
            "Frame rate (Hz)",
            "3D median filter (kernel)"
        ]

        self.roi_metadata_entries = []

        for row, text in enumerate(labels):

            label = QLabel(f"{text} : ")
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.layout.addWidget(label, row + 9, 0, 1, 1)

            entry = QLineEdit(str(None))
            entry.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            entry.setReadOnly(True)

            self.layout.addWidget(entry, row + 9, 1, 1, 1)

            self.roi_metadata_entries.append(entry)

    def _update_roi_metadata(self) -> None:

        attributes = [
             self.dataset[self.current_roi_n].roi_index,
             self.dataset[self.current_roi_n][
                 self.current_sweep_n].sweep_index,
             self.dataset[self.current_roi_n].pixel_resolution_xy,
             self.dataset[self.current_roi_n].size_xy,
             self.dataset[self.current_roi_n].z_ind,
             self.dataset[self.current_roi_n].branch_degree,
             self.dataset[self.current_roi_n].branch_id,
             self.dataset[self.current_roi_n].frame_rate,
             self.dataset[self.current_roi_n].median_filter_kernel_size,
        ]

        for n, entry in enumerate(self.roi_metadata_entries):

            if isinstance(attributes[n], list):
                attr = [f"{attribute}:.2f" for attribute in attributes[n]]

            if isinstance(attributes[n], float):
                attr = f"{attributes[n]}:.2f"

            else:
                attr = attributes[n]

            entry.setText(str(attr))

    def set_dataset(
            self,
            dataset: ImagingDataset
    ) -> None:

        self.dataset = dataset
        self.enable_roi_slider()
        self.enable_sweep_slider()

    def enable_roi_slider(self) -> None:

        self.roi_slider.setEnabled(True)
        self.roi_slider.setRange(0, len(self.dataset) - 1)
        self.roi_slider.setSingleStep(1)
        self.roi_slider.setValue(0)
        self.roi_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.roi_slider.valueChanged.connect(self.update_roi_slider)

        self._roi_metadata_summary()

    def update_roi_slider(self, roi_n: int) -> None:

        self.current_roi_n = roi_n

        if self.current_sweep_n is None:
            self.current_sweep_n = 0

        self.current_roi_label.setText(f'Current Roi: {self.current_roi_n}')

        self._update_roi_metadata()
        self.roi_n_signal.emit(self.current_roi_n)

    def enable_sweep_slider(self) -> None:

        self.sweep_slider.setEnabled(True)
        self.sweep_slider.setRange(0, self.dataset[0].n_sweeps - 1)
        self.sweep_slider.setSingleStep(1)
        self.sweep_slider.setValue(0)
        self.sweep_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sweep_slider.valueChanged.connect(self.update_sweep_slider)

    def update_sweep_slider(self, sweep_n: int) -> None:

        self.current_sweep_n = sweep_n
        self.current_sweep_label.setText(
            f'Current sweep: {self.current_sweep_n}')

        self._update_roi_metadata()
        self.sweep_n_signal.emit(self.current_sweep_n)
