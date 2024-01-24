from queue import Queue
import numpy as np

import pyqtgraph as pg
#from pyqtgraph.console import ConsoleWidget
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import QtWidgets

from record_wave import (
    recThread,
    sampleChunkThread,
    analyzerData
)

class MainWindow(QtWidgets.QMainWindow):
    """Main Window for monitoring and recording the Mic/ADC signals."""

    signal_plot_data = pg.QtCore.Signal(object)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        ## setup window
        area = DockArea()
        self.setCentralWidget(area)
        self.resize(1000,500)
        self.setWindowTitle('Spectrum Analyzer')

        ## setup layout
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
        # TODO: add color bar
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

        pcm_device = 'mic'
        if pcm_device == 'ad7606c':
            adc_conf = {
                'sampler_id': 'ad7606c',
                'sample_rate': 48000,
                'periodsize': 4800,
            }
        else:
            adc_conf = {
                'sampler_id': 'mic',
                'device'    : 'default',
                'n_channels': 1,
                'sample_rate': 48000,
                'periodsize': 1024,
                'format'    : 'S16_LE',
            }
        self.adc_conf = adc_conf

        ## setup data gernerator and analyzer
        self.buf_queue = Queue(10000)
        self.rec_thread = recThread('recorder', self.buf_queue, adc_conf)

        ana_conf = {
            'size_chunk': 1024,
            'n_ave': 1,
            'use_dBA': False,
            # TODO: calibration_path
        }
        self.ana_conf = ana_conf

        # init FFT Analyzer
        self.analyzer_data = analyzerData(
            ana_conf['size_chunk'], adc_conf['sample_rate'], ana_conf['n_ave'])
        self.analyzer_data.use_dBA = ana_conf['use_dBA']

        self.chunk_process_thread = sampleChunkThread(
            'chunking', self.func_proc_update, self.buf_queue, 0,
            ana_conf['size_chunk'], ana_conf['size_chunk']//2)
        
        # deal with the signals for plot
        self.signal_plot_data.connect(self.update_graph, pg.QtCore.Qt.ConnectionType.QueuedConnection)

        # Connect the custom closeEvent
        self.closeEvent = self.custom_close_event

        self.rec_thread.start()
        self.chunk_process_thread.start()

    def open_file_dialog(self):
        self.save_file_name = QtWidgets.QFileDialog.getSaveFileName()[0]
        self.file_path_edit.setText(self.save_file_name)

    def save_rec(self):
        self.state = self.area.saveState()
        self.stop_rec_btn.setEnabled(True)
    
    def stop_rec(self):
        self.area.restoreState(self.state)
    
    def func_proc_update(self, data_chunk):
        # usually called from data processing thread
        self.analyzer_data.put(data_chunk)
        rms_db = self.analyzer_data.get_RMS_dB()
        volt = self.analyzer_data.get_volt()
        spectrum_db = self.analyzer_data.get_spectrum_dB()
        self.signal_plot_data.emit((rms_db, volt, spectrum_db))
    
    # TODO: add decorator for callbacks
    def update_graph(self, obj):
        rms_db, volt, spectrum_db = obj
        # usually called from main thread
        self.d1_plot.setData(volt)
        self.d2_plot.setData(spectrum_db)
        self.w3_img.setImage(np.random.normal(size=(100,100)))
    
    def custom_close_event(self, event):
        self.rec_thread.b_run = False
        self.chunk_process_thread.b_run = False
        event.accept()  # Accept the close event


app = pg.mkQApp("DockArea Example")
main_window = MainWindow()

if __name__ == '__main__':
    pg.exec()