""" Created on Wed Jun 25 13:45:20 2025
    @author: dcupolillo """

import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QMainWindow, QGridLayout, QWidget, QMessageBox, QMenu, QAction,
    QPushButton)
from spyne.core.imaging.imagingdataset import ImagingDataset
from spyne.GUI.load import LoadFiles
from spyne.GUI.utils import KernelDialog
from spyne.GUI.dataset_roi import RoiPanel
from spyne.GUI.display import (
    VideoPlayerWindow, ProjectionPlayer, LocateWindow)


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        """ Main Window with Frames for each functionality."""

        QMainWindow.__init__(self)

        self.folder_name = None

        default_kernel = (3, 3, 3)
        self.kernel = default_kernel

        self.current_roi_n = None
        self.current_sweep_n = None

        self.central_widget = QWidget(self)
        self.layout = QGridLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle('Spyne')

        self.setFixedSize(300, 600)

        self._create_actions()
        self._create_menubar()

        # add all the frames
        self.load_files = LoadFiles(self)
        self.layout.addWidget(self.load_files, 0, 0, 1, 3)
        self.openAction.triggered.connect(self.load_files.choose_folder)

        self.roi_panel = RoiPanel(self)
        self.layout.addWidget(self.roi_panel, 1, 0, 1, 3)
        self.load_files.dataset_signal.connect(self.roi_panel.set_dataset)
        self.load_files.dataset_signal.connect(self.set_dataset)
        self.load_files.roipy_signals.connect(self.set_roipy_data)
        self.roi_panel.roi_n_signal.connect(self._get_roi_n)
        self.roi_panel.sweep_n_signal.connect(self._get_sweep_n)

        self.play_button = QPushButton("Play")
        self.layout.addWidget(self.play_button, 2, 0)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._open_video_player)

        self.projection_button = QPushButton("Show projection")
        self.layout.addWidget(self.projection_button, 2, 1)
        self.projection_button.setEnabled(False)
        self.projection_button.clicked.connect(self._open_projection_player)

        self.locate_button = QPushButton("Locate")
        self.layout.addWidget(self.locate_button, 2, 2)
        self.locate_button.setEnabled(False)
        self.locate_button.clicked.connect(self._open_locate_window)

    def _create_menubar(self) -> None:

        menuBar = self.menuBar()

        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.exitAction)
        # Edit menu
        editMenu = menuBar.addMenu("&Edit")
        editMenu.addAction(self.select_kernel_action)

        # Help menu
        helpMenu = menuBar.addMenu("&Help")
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addAction(self.aboutAction)

    def _create_actions(self):

        self.newAction = QAction(self)
        self.newAction.setText("&New")

        self.openAction = QAction("&Open...", self)
        self.exitAction = QAction("&Exit", self)

        self.helpContentAction = QAction("&Help Content", self)
        self.aboutAction = QAction("&About", self)

        self.select_kernel_action = QAction("Select &Kernel", self)
        self.select_kernel_action.triggered.connect(self._select_kernel)

    def _select_kernel(self):

        dialog = KernelDialog(parent=self, kernel=self.kernel)
        dialog.kernel_selected.connect(self._handle_kernel_selected)
        dialog.kernel_selected.connect(self.load_files._get_kernel)
        dialog.exec_()

    def _handle_kernel_selected(self, kernel: tuple) -> None:

        if kernel != self.kernel:
            self.kernel = kernel
            print(f"Kernel size set to: {self.kernel}.")

    def set_dataset(
            self,
            dataset: ImagingDataset
    ) -> None:

        self.dataset = dataset

    def set_roipy_data(
            self,
            roipy_data: tuple):

        self.stack = roipy_data[0]
        self.morph = roipy_data[1]
        self.scanfields = roipy_data[2]

    def _get_roi_n(self, roi_n: int) -> None:

        self.current_roi_n = roi_n
        self._is_roi_and_sweep_n()

    def _get_sweep_n(self, sweep_n: int) -> None:

        self.current_sweep_n = sweep_n
        self._is_roi_and_sweep_n()

    def _is_roi_and_sweep_n(self):

        if all((self.current_roi_n, self.current_sweep_n)):
            self.play_button.setEnabled(True)
            self.projection_button.setEnabled(True)
            self.locate_button.setEnabled(True)

    def _open_video_player(self):

        if not hasattr(self, 'dataset') or self.dataset is None:
            QMessageBox.warning(
                self, "No Data", "Please load data before playing.")
            return

        frames = (
            self.dataset[self.current_roi_n][self.current_sweep_n].sweep)
        frame_rate = (
            self.dataset[self.current_roi_n][self.current_sweep_n].frame_rate)

        self.video_window = VideoPlayerWindow(
            frames=frames,
            frame_rate=frame_rate,
            parent=self)

        self.video_window.show()

    def _open_projection_player(self):

        frames = (
            self.dataset[self.current_roi_n][self.current_sweep_n].sweep)

        if not hasattr(self, "dataset") or frames is None:
            QMessageBox.warning(self, "No Data", "No frames loaded.")
            return

        self.projection_window = ProjectionPlayer(frames, parent=self)
        self.projection_window.show()

    def _open_locate_window(self):

        z_ind = self.dataset[self.current_roi_n][self.current_sweep_n].z_ind
        n_roi = self.dataset[self.current_roi_n][self.current_sweep_n].n_roi
        n_roi_z_relative = self.dataset[self.current_roi_n][
            self.current_sweep_n].coplanar_roi_n.index(n_roi)

        morphology = self.morph.neuron
        scanfield = self.scanfields.neuron[z_ind][n_roi_z_relative]

        self.locate_window = LocateWindow(
            self.stack, morphology, scanfield, parent=self)
        self.locate_window.show()

    def closeEvent(self, event) -> None:
        """Handle the close event triggered by the top corner X button."""

        reply = QMessageBox.question(
            self, 'Quit application',
            "Are you sure you want to quit?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
            QtWidgets.QApplication.quit()
        else:
            event.ignore()


def run_app():

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()

    app.setQuitOnLastWindowClosed(True)

    spyne_window = MainWindow()
    spyne_window.show()

    if not QtWidgets.QApplication.instance():
        sys.exit(app.exec_())
    else:
        app.exec_()
