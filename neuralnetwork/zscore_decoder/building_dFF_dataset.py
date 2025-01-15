""" Created on Thu Sep 12 14:37:00 2024
    @author: dcupolillo """

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import flammkuchen


class TraceViewer(QMainWindow):
    def __init__(self, traces, timestamps):
        super().__init__()

        # Convert the traces to numpy arrays and flatten each 2D array to 1D
        self.traces = np.array([trace for sweep in traces for trace in sweep])
        self.timestamps = np.array([ts for sweep in timestamps for ts in sweep])
        self.current_index = 0  # To track the current trace

        # List to store binary decisions (1 for Include, 0 for Exclude)
        self.binary_decision = []
        # Lists to store included and excluded traces
        self.included_traces = []
        self.excluded_traces = []

        # Setup UI
        self.initUI()

    def initUI(self):
        # Create a main widget and layout
        widget = QWidget(self)
        main_layout = QHBoxLayout()

        # Create buttons layout
        buttons_layout = QVBoxLayout()

        # Create buttons
        self.include_button = QPushButton('Include', self)
        self.exclude_button = QPushButton('Exclude', self)
        self.skip_button = QPushButton('Skip', self)
        self.back_button = QPushButton('Go Back', self)
        self.save_button = QPushButton('Save & Exit', self)
        
        self.include_button.clicked.connect(self.include_trace)
        self.exclude_button.clicked.connect(self.exclude_trace)
        self.skip_button.clicked.connect(self.skip_trace)
        self.back_button.clicked.connect(self.go_back)
        self.save_button.clicked.connect(self.save_and_exit)

        # Add buttons to layout
        buttons_layout.addWidget(self.include_button)
        buttons_layout.addWidget(self.exclude_button)
        buttons_layout.addWidget(self.skip_button)
        buttons_layout.addWidget(self.back_button)
        buttons_layout.addWidget(self.save_button)

        # Create matplotlib canvases for three plots (current, included, excluded)
        self.figure_current = Figure()
        self.canvas_current = FigureCanvas(self.figure_current)

        self.figure_included = Figure()
        self.canvas_included = FigureCanvas(self.figure_included)

        self.figure_excluded = Figure()
        self.canvas_excluded = FigureCanvas(self.figure_excluded)

        # Add canvases to layout
        main_layout.addWidget(self.canvas_current)
        main_layout.addWidget(self.canvas_included)
        main_layout.addWidget(self.canvas_excluded)

        # Add button layout to the right
        main_layout.addLayout(buttons_layout)

        # Set layout and widget
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        # Plot the first trace
        self.plot_trace()

        # Set window title and size
        self.setWindowTitle('Calcium Imaging Trace Viewer')
        self.setGeometry(200, 200, 2000, 300)
        self.show()

    def plot_trace(self):
        """Plots the current trace on the canvas."""
        if self.current_index < len(self.traces):
            # Clear and plot the current trace
            self.figure_current.clear()
            ax_current = self.figure_current.add_subplot(111)
            ax_current.plot(self.timestamps[self.current_index], self.traces[self.current_index])
            ax_current.set_xlabel('Time')
            ax_current.set_ylabel('dFF')
            ax_current.set_ylim(-5, 5)
            ax_current.axvline(1.0, color='red', alpha=0.3)

            # Add title showing progress with include/exclude counts
            total_traces = len(self.traces)
            ax_current.set_title(f"{self.current_index + 1}/{total_traces}")
            self.canvas_current.draw()
        else:
            # End of traces
            self.include_button.setEnabled(False)
            self.exclude_button.setEnabled(False)
            self.skip_button.setEnabled(False)
            self.back_button.setEnabled(False)
            self.save_button.setEnabled(True)  # Enable save button at the end
            self.setWindowTitle('All Traces Processed')

    def update_included_plot(self):
        """Update the plot for included traces."""
        self.figure_included.clear()
        ax_included = self.figure_included.add_subplot(111)

        for trace in self.included_traces:
            ax_included.plot(self.timestamps[0], trace, color='gray', alpha=0.5)

        ax_included.set_title('Included Traces')
        ax_included.set_ylim(-5, 5)
        ax_included.axvline(1.0, color='red', alpha=0.3)
        ax_included.plot(self.timestamps[0], np.mean(self.included_traces, axis=0), color='green')
        ax_included.set_title(len(self.included_traces))
        self.canvas_included.draw()

    def update_excluded_plot(self):
        """Update the plot for excluded traces."""
        self.figure_excluded.clear()
        ax_excluded = self.figure_excluded.add_subplot(111)

        for trace in self.excluded_traces:
            ax_excluded.plot(self.timestamps[0], trace, color='gray', alpha=0.5)

        ax_excluded.set_title('Excluded Traces')
        ax_excluded.set_ylim(-5, 5)
        ax_excluded.axvline(1.0, color='red', alpha=0.3)
        ax_excluded.plot(self.timestamps[0], np.mean(self.excluded_traces, axis=0), color='red')
        ax_excluded.set_title(len(self.excluded_traces))
        self.canvas_excluded.draw()

    def include_trace(self):
        """Marks the current trace as included (1) in the binary decision list and updates the plots."""
        self.binary_decision.append(1)
        self.included_traces.append(self.traces[self.current_index])
        self.update_included_plot()  # Update the included traces plot
        self.next_trace()

    def exclude_trace(self):
        """Marks the current trace as excluded (0) in the binary decision list and updates the plots."""
        self.binary_decision.append(0)
        self.excluded_traces.append(self.traces[self.current_index])
        self.update_excluded_plot()  # Update the excluded traces plot
        self.next_trace()

    def skip_trace(self):
        """Skips the current trace without making a decision."""
        self.next_trace()

    def go_back(self):
        """Go back to the previous trace and remove it from the included/excluded lists."""
        if self.current_index > 0:
            # Go back one trace
            self.current_index -= 1

            # Remove the last decision and its corresponding trace
            if self.binary_decision:
                last_decision = self.binary_decision.pop()
                if last_decision == 1:
                    self.included_traces.pop()  # Remove from included traces
                    self.update_included_plot()  # Update the included plot
                elif last_decision == 0:
                    self.excluded_traces.pop()  # Remove from excluded traces
                    self.update_excluded_plot()  # Update the excluded plot

            # Re-plot the previous trace
            self.plot_trace()

    def next_trace(self):
        """Advances to the next trace and updates the plot."""
        self.current_index += 1
        self.plot_trace()

    def save_and_exit(self):
        """Saves the binary decisions, included/excluded traces, and timestamps, then exits."""
        # Save the binary decisions, traces, and timestamps using flammkuchen
        data = {
            'binary_decision': self.binary_decision,
            'included_traces': np.array(self.included_traces),
            'excluded_traces': np.array(self.excluded_traces),
            'timestamps': self.timestamps
        }

        flammkuchen.save('trace_data.h5', data)
        print("Data saved to 'trace_data.h5'")
        sys.exit()


if __name__ == '__main__':

    all_traces = []
    all_ts = []
    for segmenter in segmenter_list[:4]:
        all_traces.extend(segmenter.batch_z_scores_BLA)
        all_traces.extend(segmenter.batch_z_scores_CA3)
        all_ts.extend(segmenter.batch_ts_BLA)
        all_ts.extend(segmenter.batch_ts_CA3)

    indices = np.arange(len(all_traces))
    np.random.shuffle(indices)

    shuffled_traces = [all_traces[i] for i in indices]
    shuffled_ts = [all_ts[i] for i in indices]

    app = QApplication(sys.argv)

    viewer = TraceViewer(shuffled_traces, shuffled_ts)
    sys.exit(app.exec_())


