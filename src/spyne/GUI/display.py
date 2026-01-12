""" Created on Thu Jun 26 09:56:37 2025
    @author: dcupolillo """

import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QGridLayout, QWidget, QPushButton, QSlider,
    QAction, QInputDialog)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt


class ImageView(pg.ImageView):

    def __init__(
            self,
            parent,
            frame_rate: float = 1.0,
            *args,
            **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)

        image_window_ratio = 4
        window_width = parent.width()
        window_height = parent.height()
        base_size = min(window_width, window_height) * image_window_ratio

        self.frame_rate = frame_rate

        self.setMinimumSize(int(base_size), int(base_size))

        self.getHistogramWidget().setFixedWidth(80)

        self.ui.roiPlot.hide()
        self.ui.roiBtn.setVisible(False)
        self.ui.menuBtn.setVisible(False)

    def play(self, rate=None):
        if rate is None:
            rate = self.frame_rate
        super().play(rate)
        self._is_playing = rate > 0

    def pause(self):
        self.play(rate=0)
        self._is_playing = False

    def resume(self):
        self.play(rate=self.frame_rate)
        self._is_playing = True

    def is_playing(self):
        return self._is_playing

    def toggle_play_pause(self):
        if self._is_playing:
            self.pause()
        else:
            self.resume()

    def setLookupTable(self, lut):
        self.getImageItem().setLookupTable(lut)


class VideoPlayerWindow(QMainWindow):

    def __init__(
            self,
            frames: np.ndarray,
            frame_rate: float,
            parent: QMainWindow = None
    ) -> None:

        super().__init__(parent)

        self.frames = frames
        self.frame_rate = frame_rate
        self.current_frame = 0

        self.central_widget = QWidget(self)
        self.layout = QGridLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle("Video Player")
        self.setMinimumSize(200, 100)

        self._image_set_ch1 = False
        self._image_set_ch2 = False

        self.toggle_button = QPushButton("Play")
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self._toggle_play)
        self.layout.addWidget(self.toggle_button, 0, 2)
        self.toggle_button.setFixedSize(60, 30)

        self.prev_button = QPushButton("⟨ Prev")
        self.prev_button.clicked.connect(self._prev_frame)
        self.prev_button.setEnabled(True)
        self.layout.addWidget(self.prev_button, 1, 2)
        self.prev_button.setFixedSize(60, 30)

        self.next_button = QPushButton("Next ⟩")
        self.next_button.clicked.connect(self._next_frame)
        self.next_button.setEnabled(True)
        self.layout.addWidget(self.next_button, 2, 2)
        self.next_button.setFixedSize(60, 30)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(self._slider_released)
        self.layout.addWidget(self.slider, 6, 0, 1, 3)

        self.add_image_view(parent)
        self._create_menubar()

    def add_image_view(self, parent: QMainWindow) -> None:
        """Add custom ImageView to the layout."""

        self.imv_ch1 = ImageView(
            self, frame_rate=self.frame_rate, view=pg.PlotItem())
        self.imv_ch2 = ImageView(
            self, frame_rate=self.frame_rate, view=pg.PlotItem())

        self.layout.addWidget(self.imv_ch1, 0, 0, 5, 1)
        self.layout.addWidget(self.imv_ch2, 0, 1, 5, 1)

        self.imv_ch1.getView().showGrid(False, False)
        self.imv_ch2.getView().showGrid(False, False)

        self.imv_ch1.getHistogramWidget().region.sigRegionChanged.connect(
            self.handle_histogram)
        self.imv_ch2.getHistogramWidget().region.sigRegionChanged.connect(
            self.handle_histogram)

        self.imv_ch1.timeLine.sigPositionChanged.connect(
            self._slider_update_from_imv)
        self.imv_ch2.timeLine.sigPositionChanged.connect(
            self._slider_update_from_imv)

        self.imv_ch1.timeLine.sigPositionChanged.connect(self._check_loop)
        self.imv_ch2.timeLine.sigPositionChanged.connect(self._check_loop)

    def handle_histogram(self) -> None:
        """Update normalization settings based on histogram changes."""

        region_ch1 = self.imv_ch1.getHistogramWidget().region.getRegion()
        self.normalization_settings_ch1 = (region_ch1[0], region_ch1[1])

        region_ch2 = self.imv_ch2.getHistogramWidget().region.getRegion()
        self.normalization_settings_ch2 = (region_ch2[0], region_ch2[1])

    def _toggle_play(self):

        if self.toggle_button.isChecked():
            self.toggle_button.setText("Pause")

            n_frames = self.frames.shape[0]
            self.slider.setEnabled(True)
            self.slider.setMinimum(0)
            self.slider.setMaximum(n_frames - 1)
            self.slider.setValue(self.current_frame)

            if not self._image_set_ch1:

                rotated_ch1 = np.rot90(
                    self.frames[:, 0, :, :], k=1, axes=(1, 2))

                self.imv_ch1.setImage(rotated_ch1)
                self._image_set_ch1 = True
            self.imv_ch1.play(rate=self.frame_rate)

            if not self._image_set_ch2:

                rotated_ch2 = np.rot90(
                    self.frames[:, 1, :, :], k=1, axes=(1, 2))

                self.imv_ch2.setImage(rotated_ch2)
                self._image_set_ch2 = True
            self.imv_ch2.play(rate=self.frame_rate)

        else:
            self.toggle_button.setText("▶ Play")
            self.imv_ch1.pause()
            self.imv_ch2.pause()

    def _prev_frame(self):

        if self.current_frame > 0:
            self.current_frame -= 1
            self._show_current_frame()

    def _next_frame(self):

        max_frame = self.frames.shape[0] - 1
        if self.current_frame < max_frame:
            self.current_frame += 1
            self._show_current_frame()

    def _show_current_frame(self):

        self.imv_ch1.setCurrentIndex(self.current_frame)

        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)

    def _slider_update_from_imv(self, pos):

        self.current_frame = int(pos.value())
        self.slider.blockSignals(True)  # avoid feedback loop
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)

    def _slider_released(self):

        value = self.slider.value()
        self.current_frame = value

        self.imv_ch1.setCurrentIndex(value)
        self.imv_ch2.setCurrentIndex(value)

    def _check_loop(self, pos):

        current_index = int(pos.value())
        n_frames = self.frames.shape[0]

        if current_index >= n_frames - 1:
            self.imv_ch1.setCurrentIndex(0)
            self.imv_ch2.setCurrentIndex(0)

    def _create_menubar(self):

        menubar = self.menuBar()

        edit_menu = menubar.addMenu("&Edit")

        self.select_colormap_action = QAction("Select &Colormap", self)
        self.select_colormap_action.triggered.connect(self._select_colormap)
        edit_menu.addAction(self.select_colormap_action)

    def _select_colormap(self):

        cmap_names = plt.colormaps()
        selected, ok = QInputDialog.getItem(
            self, "Select Colormap", "Colormap:", cmap_names, 0, False
        )

        if ok and selected:
            mpl_cmap = plt.get_cmap(selected)
            lut = (
                mpl_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

            self.imv_ch1.setLookupTable(lut)
            self.imv_ch2.setLookupTable(lut)


class ProjectionPlayer(QMainWindow):

    def __init__(
            self,
            frames: np.ndarray,
            parent: QMainWindow = None
    ) -> None:

        super().__init__(parent)

        self.frames = frames
        self.channels = [
            self.frames[:, i, :, :] for i in range(frames.shape[1])]

        self.projection_type = "Max"

        self.central_widget = QWidget(self)
        self.layout = QGridLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        self.setWindowTitle("Projection Viewer")

        self.add_image_view()
        self._create_menubar()
        self._update_projection()

    def add_image_view(self):

        self.imv_ch1 = ImageView(self, view=pg.PlotItem())
        self.imv_ch2 = ImageView(self, view=pg.PlotItem())

        self.layout.addWidget(self.imv_ch1, 0, 0, 5, 1)
        self.layout.addWidget(self.imv_ch2, 0, 1, 5, 1)

        self.imv_ch1.getView().showGrid(False, False)
        self.imv_ch2.getView().showGrid(False, False)

        self.imv_ch1.getHistogramWidget().region.sigRegionChanged.connect(
            self.handle_histogram)
        self.imv_ch2.getHistogramWidget().region.sigRegionChanged.connect(
            self.handle_histogram)

    def handle_histogram(self) -> None:
        """Update normalization settings based on histogram changes."""

        region_ch1 = self.imv_ch1.getHistogramWidget().region.getRegion()
        self.normalization_settings_ch1 = (region_ch1[0], region_ch1[1])

        region_ch2 = self.imv_ch2.getHistogramWidget().region.getRegion()
        self.normalization_settings_ch2 = (region_ch2[0], region_ch2[1])

    def _create_menubar(self):

        menubar = self.menuBar()
        edit_menu = menubar.addMenu("&Edit")

        proj_action = QAction("Select Projection", self)
        proj_action.triggered.connect(self._select_projection)
        edit_menu.addAction(proj_action)

        cmap_action = QAction("Select Colormap", self)
        cmap_action.triggered.connect(self._select_colormap)
        edit_menu.addAction(cmap_action)

    def _select_projection(self):

        options = ["Max", "Mean", "Std"]

        selected, ok = QInputDialog.getItem(
            self, "Projection Type", "Choose:", options, 0, False)

        if ok and selected:
            self.projection_type = selected
            self._update_projection()

    def _select_colormap(self):

        cmap_names = plt.colormaps()
        selected, ok = QInputDialog.getItem(
            self, "Select Colormap", "Colormap:", cmap_names, 0, False)

        if ok and selected:
            mpl_cmap = plt.get_cmap(selected)
            lut = (mpl_cmap(
                np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

            self.imv_ch1.setLookupTable(lut)
            self.imv_ch2.setLookupTable(lut)

    def _update_projection(self):

        if self.projection_type == "Max":
            img_ch1 = np.max(self.channels[0], axis=0)
            img_ch2 = np.max(self.channels[1], axis=0)

        elif self.projection_type == "Mean":
            img_ch1 = np.mean(self.channels[0], axis=0)
            img_ch2 = np.mean(self.channels[1], axis=0)

        elif self.projection_type == "Std":
            img_ch1 = np.std(self.channels[0], axis=0)
            img_ch2 = np.std(self.channels[1], axis=0)

        rotated_ch1 = np.rot90(img_ch1, k=1)
        rotated_ch2 = np.rot90(img_ch2, k=1)

        self.imv_ch1.setImage(rotated_ch1)
        self.imv_ch2.setImage(rotated_ch2)


class LocateWindow(QMainWindow):

    def __init__(
            self,
            stack: np.ndarray,
            morphology: list,
            scanfield: dict,
            parent: QMainWindow = None
    ) -> None:

        super().__init__(parent)

        self.stack = stack
        self.morphology = morphology
        self.scanfield = scanfield

        self.central_widget = QWidget(self)
        self.layout = QGridLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        self.setWindowTitle("Locate Viewer")

        self.add_image_view()
        self._create_menubar()

    def add_image_view(self):

        self.imv = ImageView(self, view=pg.PlotItem())

        self.layout.addWidget(self.imv, 0, 0)

        self.imv.getView().showGrid(True, True)

        self.imv.getHistogramWidget().region.sigRegionChanged.connect(
            self.handle_histogram)

        self.imv.getHistogramWidget().setHistogramRange(
            np.iinfo(np.int16).min, np.iinfo(np.int16).max)

        image = np.max(self.stack.image, axis=0)

        self.imv.setImage(
            image.T,
            pos=(-self.stack.width_um / 2, -self.stack.height_um / 2),
            scale=(1 / self.stack.pix_um_ratio,
                   1 / self.stack.pix_um_ratio))

        self.plot_morph()
        self.plot_scanfield()

    def handle_histogram(self) -> None:
        """Update normalization settings based on histogram changes."""

        region = self.imv.getHistogramWidget().region.getRegion()
        self.normalization_settings = (region[0], region[1])

    def _create_menubar(self):

        menubar = self.menuBar()
        edit_menu = menubar.addMenu("&Edit")

        cmap_action = QAction("Select Colormap", self)
        cmap_action.triggered.connect(self._select_colormap)
        edit_menu.addAction(cmap_action)

    def _select_colormap(self):

        cmap_names = plt.colormaps()
        selected, ok = QInputDialog.getItem(
            self, "Select Colormap", "Colormap:", cmap_names, 0, False)

        if ok and selected:
            mpl_cmap = plt.get_cmap(selected)
            lut = (mpl_cmap(
                np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

            self.imv.setLookupTable(lut)

    def plot_morph(self) -> None:
        """Plot morphology structure on the canvas."""

        x = [node.x for node in self.morphology]
        y = [node.y for node in self.morphology]

        self.plot_morph_data = pg.PlotDataItem(
            x, y,
            symbol='o',
            pen=None,
            symbolBrush='m')

        self.imv.addItem(self.plot_morph_data)

    def plot_scanfield(self) -> None:
        """Plot the scanfield ROIs on the canvas."""

        roi_rect = pg.RectROI(
            pos=self.scanfield.bottom_right,
            size=self.scanfield.size_xy[::-1],
            angle=self.scanfield.rotation_degrees,
            removable=True,
            pen=pg.mkPen(color='green', width=2))

        tooltip_text = "\n".join(
            f"{attr.lstrip('_')}: {getattr(self.scanfield, attr)}"
            for attr in dir(self.scanfield)
            if not attr.startswith("__")
            and not callable(getattr(self.scanfield, attr))
        )

        roi_rect.setToolTip(tooltip_text)

        self.imv.getView().addItem(roi_rect)
