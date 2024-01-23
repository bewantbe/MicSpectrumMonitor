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

dock1 = Dock("Waveform", size=(100, 200))
dock2 = Dock("Spectrum", size=(500, 300), closable=True)
dock3 = Dock("Spectrogram", size=(500,400))
dock4 = Dock("Control Pannel", size=(500,200))
area.addDock(dock1, 'left')
area.addDock(dock2, 'bottom', dock1)
area.addDock(dock4, 'bottom', dock2)
area.addDock(dock3, 'right')


## Add widgets into each dock

## Dock 1
widg1 = pg.PlotWidget(title="Waveform")
d1_plot = widg1.plot(np.random.normal(size=100))
dock1.addWidget(widg1)

## Dock 2
#dock2.hideTitleBar()
widg2 = pg.PlotWidget(title="Spectrum")
d2_plot = widg2.plot(np.random.normal(size=100))
dock2.addWidget(widg2)

## Dock 3
#widg3 = pg.ImageView()

widg3 = pg.GraphicsView()
vb3 = pg.ViewBox()
widg3.setCentralItem(vb3)
w3_img = pg.ImageItem()
w3_img.setImage(np.random.normal(size=(100,100)))
vb3.addItem(w3_img)
dock3.addWidget(widg3)

## Dock 4
widg4 = pg.LayoutWidget()
label = QtWidgets.QLabel("Set the parameters for recording:")
file_path_edit = QtWidgets.QLineEdit("", placeholderText="File path for saving the recording")
file_choose_btn = QtWidgets.QPushButton('Browse')
start_mon_btn = QtWidgets.QPushButton('Monitoring')
start_rec_btn = QtWidgets.QPushButton('Start recording')
stop_rec_btn  = QtWidgets.QPushButton('Stop recording')
stop_rec_btn.setEnabled(False)
widg4.addWidget(label, row=0, col=0)
widg4.addWidget(file_path_edit, row=1, col=0)
widg4.addWidget(file_choose_btn, row=1, col=1)
widg4.addWidget(start_mon_btn, row=2, col=0)
widg4.addWidget(start_rec_btn, row=2, col=1)
widg4.addWidget(stop_rec_btn, row=2, col=2)
dock4.addWidget(widg4)


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