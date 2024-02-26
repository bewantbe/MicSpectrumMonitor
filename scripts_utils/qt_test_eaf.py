import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg
import numpy as np

class MyMainWindow(QMainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()

        # Set up your PyQtGraph plot or other components here
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        # Example x and y data
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)

        # Plot the data
        self.plot_widget.plot(x=x_data, y=y_data, pen='b')

        # Set margins for x-axis and y-axis
        margin_x = 0.5  # Example margin for x-axis
        margin_y = 0.2  # Example margin for y-axis
        self.set_axis_margins(self.plot_widget.getPlotItem().getAxis('bottom'), margin_x)
        self.set_axis_margins(self.plot_widget.getPlotItem().getAxis('left'), margin_y)

    def set_axis_margins(self, axis, margin):
        # Adjust the range of the axis to set a margin
        current_range = axis.range
        new_range = [current_range[0] - margin, current_range[1] + margin]
        axis.setRange(*new_range)

def main():
    app = QApplication(sys.argv)
    main_window = MyMainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()