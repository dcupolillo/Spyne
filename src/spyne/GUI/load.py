""" Created on Wed Jun 25 13:49:08 2025
    @author: dcupolillo """

import spyne
from pathlib import Path
from PyQt5.QtWidgets import (
    QMainWindow, QFrame, QLabel, QGridLayout,
    QPushButton, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
import ROIpy as rp


class LoadFiles(QFrame):

    dataset_signal = pyqtSignal(spyne.ImagingDataset)
    roipy_signals = pyqtSignal(tuple)
    started_loading = pyqtSignal()
    finished_loading = pyqtSignal()

    def __init__(self, parent: QMainWindow) -> None:
        """ Frame where path is loaded. """

        super().__init__(parent)

        default_kernel = (3, 3, 3)
        self.kernel = default_kernel

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add widgets
        title = QLabel("Load")
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title, 0, 0, 1, 3)

        self.folder_entry = QLabel()
        self.folder_entry.setWordWrap(False)
        self.folder_entry.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.layout.addWidget(QLabel('Select a folder: '), 1, 0, 1, 1)
        self.layout.addWidget(self.folder_entry, 1, 1, 1, 1)

        choose_button = QPushButton('Choose')
        self.layout.addWidget(choose_button, 1, 2, 1, 1)
        choose_button.clicked.connect(self.choose_folder)

        self.load_button = QPushButton('Load')
        self.layout.addWidget(self.load_button, 2, 2)
        self.load_button.clicked.connect(
            lambda: self.load())
        self.load_button.setEnabled(False)

    def choose_folder(self) -> None:
        """
        Opens a dialog to select a folder and searches
        for a .tif and a .swc file within it.
        Updates the label with the folder path.
        Stores the filenames of the found files.
        Runs the file loaded check.

        Parameters
        ----------
        label : QLabel
            Label widget to update with the folder path.

        Returns
        -------
        None
        """

        folder_name = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.ReadOnly
        )

        self.folder_name = Path(folder_name)

        if self.folder_name:
            self.folder_entry.setText(folder_name)

            self.are_files_loaded()

    def are_files_loaded(self) -> None:

        folder_exists = self.folder_name.exists()
        folder_not_empty = any(self.folder_name.iterdir())

        subfolders_not_empty = any(
            [any(folder.iterdir())
             for folder in self.folder_name.iterdir()
             if folder.is_dir()])

        subfolders_contains_tif = any(
            [any(folder.glob('*.tif'))
             for folder in self.folder_name.iterdir()
             if folder.is_dir()])

        if (folder_exists and
                folder_not_empty and
                subfolders_not_empty and
                subfolders_contains_tif):

            self.load_button.setEnabled(True)

    def _get_kernel(self, kernel: tuple) -> None:

        if kernel != self.kernel:
            self.kernel = kernel

    def load(self) -> None:
        """
        Creates the core Structures of ROIpy.
        Emit a signal to Main Window.
        Enables the structure buttons in the structure frame.
        """
        self.started_loading.emit()

        try:
            dataset = spyne.ImagingDataset(self.folder_name, self.kernel)

            stack_filename = [
                file for file in self.folder_name.parent.glob("*.tif")][0]
            stack = rp.Stack(stack_filename)

            swc_filename = [
                file for file in self.folder_name.parent.glob("*.swc")][0]
            morph = rp.Morphology(swc_filename, stack)

            scanfields = rp.Scanfields(morph)

            self.dataset_signal.emit(dataset)
            self.roipy_signals.emit((stack, morph, scanfields))

            QMessageBox.information(self, "Success", "DONE!")

        finally:
            self.finished_loading.emit()
