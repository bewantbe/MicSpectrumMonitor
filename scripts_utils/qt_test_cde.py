import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.Qt import QtWidgets, QtCore

# Create Qt application
#app = QtGui.QApplication([])
app = QtWidgets.QApplication([])

# Create main window
#win = QtGui.QMainWindow()
win = QtWidgets.QMainWindow()
win.setWindowTitle("Sine Curve and Random Bitmap")

# Create central widget to hold the layouts
#central_widget = QtGui.QWidget()
central_widget = QtWidgets.QWidget()
win.setCentralWidget(central_widget)

# Create vertical layout to divide the window
#layout = QtGui.QVBoxLayout()
layout = QtWidgets.QVBoxLayout()
central_widget.setLayout(layout)

# --- Upper PlotWidget for sine curve ---
upper_plot = pg.PlotWidget()

# Generate sine data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot the sine curve
upper_plot.plot(x, y, pen='r')

# Add upper plot to the layout
layout.addWidget(upper_plot)

# --- Lower PlotWidget for bitmap ---
lower_plot = pg.PlotWidget()

# Generate random bitmap data
data = np.random.randint(0, 256, size=(100, 100))
img_item = pg.ImageItem(data)
lower_plot.addItem(img_item)

# Add lower plot to the layout
layout.addWidget(lower_plot)

# Show the window
win.show()

# Run the Qt application
app.exec()