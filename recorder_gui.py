import numpy as np

import pyqtgraph as pg
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import QtWidgets

app = pg.mkQApp("DockArea Example")
win = QtWidgets.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1000,500)
win.setWindowTitle('Spectrum Analyzer')

d1 = Dock("Dock1 - Waveform", size=(100, 200))     ## give this dock the minimum possible size
d2 = Dock("Dock2 - Spectrum", size=(500, 300), closable=True)
d3 = Dock("Dock3 - Spectrogram", size=(500,400))
d4 = Dock("Dock4 - Control Pannel", size=(500,200))
area.addDock(d1, 'left')        # place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
area.addDock(d2, 'bottom', d1)  # place d2 at bottom edge of d1
area.addDock(d4, 'bottom', d2)  # place d4 at bottom edge of d2
area.addDock(d3, 'right')       # place d2 at right edge of dock area


## Add widgets into each dock

## Dock 1
w1 = pg.PlotWidget(title="Waveform")
d1_plot = w1.plot(np.random.normal(size=100))
d1.addWidget(w1)

## Dock 2
#d2.hideTitleBar()
w2 = pg.PlotWidget(title="Spectrum")
d2_plot = w2.plot(np.random.normal(size=100))
d2.addWidget(w2)

## Dock 3
#w3 = pg.ImageView()

w3 = pg.GraphicsView()
vb3 = pg.ViewBox()
w3.setCentralItem(vb3)
w3_img = pg.ImageItem()
w3_img.setImage(np.random.normal(size=(100,100)))
vb3.addItem(w3_img)
d3.addWidget(w3)

## Dock 4
w4 = pg.LayoutWidget()
label = QtWidgets.QLabel("Set the parameters for recording:")
file_path_edit = QtWidgets.QLineEdit("", placeholderText="File path for saving the recording")
file_choose_btn = QtWidgets.QPushButton('Browse')
start_mon_btn = QtWidgets.QPushButton('Monitoring')
start_rec_btn = QtWidgets.QPushButton('Start recording')
stop_rec_btn  = QtWidgets.QPushButton('Stop recording')
stop_rec_btn.setEnabled(False)
w4.addWidget(label, row=0, col=0)
w4.addWidget(file_path_edit, row=1, col=0)
w4.addWidget(file_choose_btn, row=1, col=1)
w4.addWidget(start_mon_btn, row=2, col=0)
w4.addWidget(start_rec_btn, row=2, col=1)
w4.addWidget(stop_rec_btn, row=2, col=2)
d4.addWidget(w4)


state = None
save_file_name = None
def open_file_dialog():
    global save_file_name
    save_file_name = QtWidgets.QFileDialog.getSaveFileName()[0]
    file_path_edit.setText(save_file_name)

file_choose_btn.clicked.connect(open_file_dialog)

def save_rec():
    global state
    state = area.saveState()
    stop_rec_btn.setEnabled(True)
    
def stop_rec():
    global state
    area.restoreState(state)

start_rec_btn.clicked.connect(save_rec)
stop_rec_btn.clicked.connect(stop_rec)

def update():
    d1_plot.setData(np.random.normal(size=100))
    d2_plot.setData(np.random.normal(size=100))
    w3_img.setImage(np.random.normal(size=(100,100)))

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)


win.show()

if __name__ == '__main__':
    pg.exec()