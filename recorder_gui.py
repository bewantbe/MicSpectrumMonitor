import numpy as np

import pyqtgraph as pg
#from pyqtgraph.console import ConsoleWidget
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import QtWidgets

class MainWindow(QtWidgets.QMainWindow):
    """Main Window for monitoring and recording the Mic/ADC signals."""
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        area = DockArea()
        self.setCentralWidget(area)
        self.resize(1000,500)
        self.setWindowTitle('Spectrum Analyzer')

        dock1 = Dock("Waveform", size=(100, 200))
        dock2 = Dock("Spectrum", size=(500, 300), closable=True)
        dock3 = Dock("Spectrogram", size=(500,400))
        dock4 = Dock("Control Pannel", size=(500,200))
        area.addDock(dock1, 'left')
        area.addDock(dock2, 'bottom', dock1)
        area.addDock(dock4, 'bottom', dock2)
        area.addDock(dock3, 'right')
        self.area = area
        self.dock1 = dock1
        self.dock2 = dock2
        self.dock3 = dock3
        self.dock4 = dock4

        ## Add widgets into each dock

        ## Dock 1
        widg1 = pg.PlotWidget(title="Waveform")
        d1_plot = widg1.plot(np.random.normal(size=100))
        dock1.addWidget(widg1)
        self.widg1 = widg1
        self.d1_plot = d1_plot

        ## Dock 2
        #dock2.hideTitleBar()
        widg2 = pg.PlotWidget(title="Spectrum")
        d2_plot = widg2.plot(np.random.normal(size=100))
        dock2.addWidget(widg2)
        self.widg2 = widg2
        self.d2_plot = d2_plot

        ## Dock 3
        #widg3 = pg.ImageView()

        widg3 = pg.GraphicsView()
        vb3 = pg.ViewBox()
        widg3.setCentralItem(vb3)
        w3_img = pg.ImageItem()
        w3_img.setImage(np.random.normal(size=(100,100)))
        vb3.addItem(w3_img)
        dock3.addWidget(widg3)
        self.widg3 = widg3
        self.w3_img = w3_img

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
        self.widg4 = widg4
        self.file_path_edit = file_path_edit
        self.file_choose_btn = file_choose_btn
        self.start_mon_btn = start_mon_btn
        self.stop_rec_btn = stop_rec_btn

        self.save_file_name = None
        self.state = None

        file_choose_btn.clicked.connect(self.open_file_dialog)
        start_rec_btn.clicked.connect(self.save_rec)
        stop_rec_btn.clicked.connect(self.stop_rec)

        timer = pg.QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(50)
        self.timer = timer

        self.show()

    def open_file_dialog(self):
        self.save_file_name = QtWidgets.QFileDialog.getSaveFileName()[0]
        self.file_path_edit.setText(self.save_file_name)

    def save_rec(self):
        self.state = self.area.saveState()
        self.stop_rec_btn.setEnabled(True)
        
    def stop_rec(self):
        self.area.restoreState(self.state)

    def update(self):
        self.d1_plot.setData(np.random.normal(size=100))
        self.d2_plot.setData(np.random.normal(size=100))
        self.w3_img.setImage(np.random.normal(size=(100,100)))


app = pg.mkQApp("DockArea Example")
main_window = MainWindow()

if __name__ == '__main__':
    pg.exec()