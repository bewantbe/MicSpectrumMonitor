#!/usr/bin/env python3

# Usage:
#   python3 recorder_gui.py

# Contact: xyy <bewantbe@gmail.com>
# Github: https://github.com/bewantbe/MicSpectrumMonitor

import re
import datetime
import queue
import wave
import threading

import numpy as np

import PyQt6
import PyQt6.QtWidgets

import pyqtgraph as pg
#from pyqtgraph.console import ConsoleWidget
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import (
    QtCore,
    QtWidgets
)

from record_wave import (
    recThread,
    sampleChunkThread,
    analyzerData
)

"""
Roadmap:
--------

* Introduce and test multi-channels, monitor and recorder
  - done
* write spectrogram show (1 channel then n-cahnnels)
  - done
* refactor spectrogram show to make it independent
  - done
* refactor spectrum plot to make it independent
  - done
* make waveform plot independent
  - done
* Add time-frequency axis to the spectrogram
  - done.
* Add button to save the window as a png file
  - done.
* apply analysis to multi-channels.
  - extend analyzerData to multi-channels
  - extend plots
* Add limit to FPS. ref to the fps counter design in pyqtgraph example
* show multi-channel waveform spectrum spectrogram
* Add RMS curve plot.
* User interaction design
  - start/stop recording
  - monitoring/stop
  - select channels
  - Show the recording time we saved.
  - Show possible recording time left.
* Test AD7606C
* link frequency axis of spectrum and spectrogram
* consider support RF64 format for wav file, e.g.
  - using soundfile, https://pypi.org/project/soundfile/
  - using pysndfile, https://pypi.org/project/pysndfile/, https://forge-2.ircam.fr/roebel/pysndfile
"""

class AudioSaver:
    def __init__(self):
        self.wav_handler = None
    
    def init(self, wav_path, n_channel, bit_depth, sample_rate):
        self.wav_handler = wave.open(wav_path, 'wb')
        self.wav_handler.setnchannels(n_channel)
        self.wav_handler.setsampwidth(bit_depth // 8)
        self.wav_handler.setframerate(sample_rate)
        self.n_channel = n_channel
        self.frame_bytes = bit_depth // 8 * n_channel
        self.volt_scaler = 2 ** (bit_depth - 1)
        self.np_dtype = np.dtype('int{}'.format(bit_depth))
        self.n_frame = 0
        self.wav_handler.setparams
    
    def close(self):
        if self.wav_handler is None:
            return
        #self.wav_handler.setnframes(self.n_frame)  # paired with writeframesraw
        self.wav_handler.close()
        self.wav_handler = None
    
    def write(self, data):
        if self.wav_handler is None:
            raise RuntimeError("AudioSaver is not initialized.")

        if isinstance(data, np.ndarray) and \
            ((data.dtype == np.float32) or (data.dtype == np.float64)):
            volt = data
            sample = np.array(np.round(volt * self.volt_scaler),
                              dtype = self.np_dtype, order = 'C')
            data = sample.data   # .data or .tobytes()

        # data must be a bytes-like object, len() will return number of bytes
        self.wav_handler.writeframes(data)
        #self.wav_handler.writeframesraw(data)    should be faster
        self.n_frame += len(data) // self.frame_bytes
    
    def __del__(self):
        self.close()
    
class RecorderWriteThread(threading.Thread):
    """Output end of the recorder, usually a wav file."""
    def __init__(self, buf_que, writer, writer_conf):
        threading.Thread.__init__(self)
        self.buf_que = buf_que
        self.writer = writer
        self.writer_conf = writer_conf
        self._stop_event = threading.Event()
        self._initilized = threading.Event()
        self.status_check_tick = 0.1  # second

    def run(self):
        try:
            self.writer.init(**self.writer_conf)
        except OSError:
            self.stop()
            self._initilized.set()
            return
        self._initilized.set()
        while not self._stop_event.is_set():
            try:
                s = self.buf_que.get(True, self.status_check_tick)
            except queue.Empty:
                s = []
            if (len(s) == 0):
                continue
            self.writer.write(s)
        self.writer.close()

    def stop(self):
        # usually called from other thread
        self._stop_event.set()
        # Note: make sure call the 'join()' after the stop().
    
    def is_running(self):
        # might be called from other thread
        return not self._stop_event.is_set()

class WaveformPlot:
    def __init__(self):
        self.auto_range = True
    
    def init_to_widget(self):
        #dock1.hideTitleBar()
        widg1 = pg.PlotWidget(title="Waveform")
        d1_plot = widg1.plot(np.random.normal(size=100))
        widg1.getPlotItem().getAxis('left').setWidth(50)
        self.widg1 = widg1
        self.d1_plot = d1_plot
        return widg1

    def init_param(self, analyzer, sz_hop):
        self.sz_chunk = analyzer.sz_chunk
        self.sz_hop = sz_hop              # overlap = sz_chunk - sz_hop

    def config_plot(self):
        if not self.auto_range:
            self.widg1.setRange(self.waveform_plot_range)

    @property
    def waveform_plot_range(self):
        rg = QtCore.QRectF(*map(float, [0, -1.0, self.sz_chunk, 2.0]))
        return rg  # x, y, width, height

    def update(self, volt):
        self.d1_plot.setData(volt)

class SpectrumPlot:
    def __init__(self):
        self.log_mode = False
    
    def init_to_widget(self):
        #dock2.hideTitleBar()
        widg2 = pg.PlotWidget(title="Spectrum")
        d2_plot = widg2.plot(np.random.normal(size=100))
        self.widg2 = widg2
        self.d2_plot = d2_plot
        return widg2

    def init_param(self, analyzer, sz_hop):
        self.sz_chunk = analyzer.sz_chunk
        self.sz_hop = sz_hop              # overlap = sz_chunk - sz_hop
        self.max_freq = analyzer.fqs[-1]
        self.x_freq = analyzer.fqs
        self.n_freq = len(self.x_freq)

    def config_plot(self):
        if self.log_mode:
            pass
        else:
            self.widg2.setRange(self.spectrum_plot_range)
            #plot_set.d2_plot.setData(x = self.x_freq)

    @property
    def spectrum_plot_range(self):
        rg = QtCore.QRectF(*map(float, [0, -120, self.max_freq, 120]))
        return rg  # x, y, width, height

    def update(self, fqs, spectrum_db):
        if self.log_mode:
            pass
        else:
            self.d2_plot.setData(x = fqs, y = spectrum_db)
            #self.d2_plot.setData(x = self.x_freq, y = spectrum_db)

class SpectrogramPlot:
    def __init__(self):
        self.log_mode = False
        self.spam_bmp_t_duration_set = 6.0  # sec

    def init_to_widget(self):
        # called in main thread init
        widg3 = pg.GraphicsLayoutWidget()  # use GraphicsLayoutWidget instead of GraphicsView
        plot_item = widg3.addPlot()
        w3_img = pg.ImageItem()
        w3_img.setImage(np.random.normal(size=(100,100)))
        plot_item.addItem(w3_img)
        self.widg3 = widg3
        self.w3_img = w3_img
        self.plot_item = plot_item
        # TODO: add color bar
        return widg3  # for add to dock: dock3.addWidget(widg3)
    
    def init_param(self, analyzer, sz_hop):
        t_hop = sz_hop / analyzer.sample_rate
        self.max_freq = analyzer.fqs[-1]
        self.x_freq = analyzer.fqs
        self.n_freq = len(self.x_freq)
        self.spam_len = int(self.spam_bmp_t_duration_set / t_hop)
        self.spam_bmp_t_duration = self.spam_len * t_hop    # correct the duration
        self.spam_loop_cursor = 0
        if self.log_mode:
            self.spam_bmp = np.zeros((self.spam_len, self.n_freq))   # TODO: fix this according to the zooming
        else:
            self.spam_bmp = np.zeros((self.spam_len, self.n_freq))
        self.spam_lock = threading.Lock()
    
    def config_plot(self):
        if self.log_mode:
            pass
        else:
            self.plot_item.setLabels(left='Hz', bottom='second')
            #self.plot_item.setRange(xRange=[0, self.spam_bmp_t_duration], yRange=[0, self.max_freq])
            #x_axis = self.plot_item.getAxis('bottom')
            #y_axis = self.plot_item.getAxis('left')
    
    def feed_spectrum(self, spectrum):
        # separate computation (here) and plot (in update()) in different threads
        with self.spam_lock:
            self.spam_bmp[self.spam_loop_cursor,:] = spectrum
            self.spam_loop_cursor = (self.spam_loop_cursor + 1) % self.spam_len
        return self.spam_bmp
    
    def get_spectrogram_bmp(self):
        if self.log_mode:
            return self.spam_bmp  # TODO: redraw the bmp according to zoom
        else:
            return self.spam_bmp

    def update(self):
        spam_bmp = self.get_spectrogram_bmp()
        with self.spam_lock:                   # TODO: maybe I don't need this lock
            self.w3_img.setImage(spam_bmp,
                rect=[0, 0, self.spam_bmp_t_duration, self.max_freq])

class MainWindow(QtWidgets.QMainWindow):
    """Main Window for monitoring and recording the Mic/ADC signals."""

    signal_update_graph = pg.QtCore.Signal(object)

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
        self.waveform_plot = WaveformPlot()
        dock1.addWidget(self.waveform_plot.init_to_widget())

        ## Dock 2
        self.spectrum_plot = SpectrumPlot()
        dock2.addWidget(self.spectrum_plot.init_to_widget())

        ## Dock 3
        self.spectrogram_plot = SpectrogramPlot()
        dock3.addWidget(self.spectrogram_plot.init_to_widget())

        ## Dock 4
        widg4 = pg.LayoutWidget()
        self.widg4 = widg4

        label = QtWidgets.QLabel("Set the parameters for recording:")

        file_path_edit = QtWidgets.QLineEdit("", placeholderText="File path for saving the recording")
        self.wav_save_path = None
        self.file_path_edit = file_path_edit

        file_choose_btn = QtWidgets.QPushButton('Browse')
        file_choose_btn.clicked.connect(self.open_file_dialog)
        self.file_choose_btn = file_choose_btn

        start_mon_btn = QtWidgets.QPushButton('Monitoring')
        self.start_mon_btn = start_mon_btn

        win_screenshot_btn = QtWidgets.QPushButton('Window-shot')
        win_screenshot_btn.clicked.connect(self.take_screen_shot)
        self.win_screenshot_btn = win_screenshot_btn

        start_rec_btn = QtWidgets.QPushButton('Start recording')
        start_rec_btn.clicked.connect(self.start_audio_saving)
        self.start_rec_btn = start_rec_btn

        stop_rec_btn  = QtWidgets.QPushButton('Stop recording')
        stop_rec_btn.clicked.connect(self.stop_audio_saving)
        stop_rec_btn.setEnabled(False)
        self.stop_rec_btn = stop_rec_btn

        # layout
        widg4.addWidget(label, row=0, col=0)
        widg4.addWidget(file_path_edit, row=1, col=0, colspan=2)
        widg4.addWidget(file_choose_btn, row=1, col=2)
        widg4.addWidget(start_mon_btn, row=2, col=0)
        widg4.addWidget(win_screenshot_btn, row=2, col=1)
        widg4.addWidget(start_rec_btn, row=2, col=2)
        widg4.addWidget(stop_rec_btn, row=2, col=3)
        dock4.addWidget(widg4)

        # Connect the custom closeEvent
        self.closeEvent = self.custom_close_event

        self.show()

        ## setup audio input and processings
        pcm_device = 'mic'
        if pcm_device == 'ad7606c':
            adc_conf = {
                'sampler_id': 'ad7606c',
                'sample_rate': 48000,
                'periodsize': 4800,
            }
        else:
            adc_conf = {
                'sampler_id'   : 'mic',
                'device'       : 'default',
                'sample_rate'  : 48000,
                'n_channel'   : 2,
                'value_format' : 'S16_LE',
                'periodsize'   : 1024,
            }
        self.adc_conf = adc_conf
        self.bit_depth = 16       # assume always S16_LE

        ## setup data gernerator and analyzer
        # Data flow: mic source -> buf_queue -> process{analyzer, signal to plot}
        self.data_queue_max_size = 10000
        self.buf_queue = queue.Queue(self.data_queue_max_size)
        self.rec_thread = recThread('recorder', self.buf_queue, adc_conf)
        # TODO: allow recThread to accept multiple queues (like pipelines) for multiple downstreams
        #       plan two: in sampleChunkThread, we setup another callback for receiving raw data

        self.channel_selected = [0, 1]  # select channel(s) by vector of indexes

        ana_conf = {
            'size_chunk': 1024,
            'n_ave': 1,
            'use_dBA': False,
            # TODO: calibration_path
        }
        self.ana_conf = ana_conf

        # init FFT Analyzer
        self.analyzer_data = analyzerData(
            ana_conf['size_chunk'], adc_conf['sample_rate'], ana_conf['n_ave'],
            len(self.channel_selected))
        self.analyzer_data.use_dBA = ana_conf['use_dBA']
        
        # signals for calling self.proc_analysis_plot
        self.signal_update_graph.connect(self.update_graph, pg.QtCore.Qt.ConnectionType.QueuedConnection)

        sz_chunk = ana_conf['size_chunk']
        sz_hop = ana_conf['size_chunk'] // 2
        self.chunk_process_thread = sampleChunkThread('chunking',
            self.proc_analysis_plot, self.buf_queue, self.channel_selected,
            sz_chunk, sz_hop,
            callback_raw = self.proc_orig_data)

        ## setup plots
        self.waveform_plot.init_param(self.analyzer_data, sz_hop)
        self.waveform_plot.config_plot()
        self.spectrum_plot.init_param(self.analyzer_data, sz_hop)
        self.spectrum_plot.config_plot()
        self.spectrogram_plot.init_param(self.analyzer_data, sz_hop)
        self.spectrogram_plot.config_plot()

        ## for audio saving thread
        # Data flow: process{} -> wav_data_queue -> wav_writer
        self.wav_data_queue = None
        self.wav_writer_thread = None

        self.rec_thread.start()
        self.chunk_process_thread.start()

    def open_file_dialog(self):
        self.wav_save_path = QtWidgets.QFileDialog.getSaveFileName()[0]
        self.file_path_edit.setText(self.wav_save_path)

    def start_audio_saving(self):
        self.stop_rec_btn.setEnabled(True)
        self.start_rec_btn.setEnabled(False)
        # Ensure we have a file name anyway
        if (self.wav_save_path is None) or (self.wav_save_path == ''):
            # set file name by date and time
            now = datetime.datetime.now()
            self.wav_save_path = now.strftime("untitled_%Y-%m-%d_%H%M%S.wav")
            self.file_path_edit.setText(self.wav_save_path)
        # For setup the wav writer
        wav_saver_conf = {
            'wav_path': self.wav_save_path,
            'n_channel': len(self.channel_selected),
            'bit_depth': self.bit_depth,
            'sample_rate': self.adc_conf['sample_rate']
        }
        self.wav_data_queue = queue.Queue(self.data_queue_max_size)  # Get a new queue anyway, TODO: avoid memory leak
        self.wav_writer_thread = RecorderWriteThread(
            self.wav_data_queue, AudioSaver(), wav_saver_conf)
        self.wav_writer_thread.start()
        # check bad file name
        print('waiting recorder to start...', end='')
        self.wav_writer_thread._initilized.wait()
        print('Done.')
        if not self.wav_writer_thread.is_running():
            self.start_rec_btn.setEnabled(True)
            self.stop_rec_btn.setEnabled(False)
            self.simple_message_box(f"Failed to open file {self.wav_save_path}.")
    
    def stop_audio_saving(self):
        self.start_rec_btn.setEnabled(True)
        self.stop_rec_btn.setEnabled(False)
        self.wav_writer_thread.stop()
        self.wav_writer_thread.join()
        # pop up a message box saying the file is saved
        self.simple_message_box(f"The recording is saved to {self.wav_save_path}.")
    
    def simple_message_box(self, msg_text):
        msg = QtWidgets.QMessageBox()
        #msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(msg_text)
        msg.setWindowTitle("Recording saved")
        msg.setStandardButtons(PyQt6.QtWidgets.QMessageBox.StandardButton.Ok)
        msg.exec()
    
    def proc_orig_data(self, data_chunk):
        ## usually called from data processing thread
        # if we are ready to write data, then do it (but
        # do not cost too much CPU/IO time, let the other thread to do the actual work)
        if self.wav_writer_thread is not None and\
              self.wav_writer_thread.is_running():
            self.wav_data_queue.put(data_chunk)   # TODO: should we copy the data?

    def proc_analysis_plot(self, data_chunk):
        ## usually called from data processing thread
        # analysing
        self.analyzer_data.put(data_chunk)
        fqs = self.analyzer_data.fqs
        rms_db = self.analyzer_data.get_RMS_dB()
        volt = self.analyzer_data.get_volt()[:,0]
        spectrum_db = self.analyzer_data.get_spectrum_dB()[:,0]
        self.spectrogram_plot.feed_spectrum(spectrum_db)
        # plot
        self.signal_update_graph.emit((rms_db, volt, fqs, spectrum_db))
    
    # TODO: annotate callbacks using decorator
    def update_graph(self, obj):
        # usually called from main thread
        rms_db, volt, fqs, spectrum_db = obj
        # ploting
        self.waveform_plot.update(volt)
        self.spectrum_plot.update(fqs, spectrum_db)
        self.spectrogram_plot.update()
    
    def custom_close_event(self, event):
        self.rec_thread.b_run = False
        self.chunk_process_thread.b_run = False
        event.accept()  # Accept the close event

    def take_screen_shot(self):
        app = pg.QtGui.QGuiApplication.instance()
        screen = app.primaryScreen()
        screenshot = screen.grabWindow(self.winId())
        now = datetime.datetime.now()
        png_path = now.strftime("winshot_%Y-%m-%d_%H%M%S.png")
        screenshot.save(png_path, 'png')

if __name__ == '__main__':
    app = pg.mkQApp("DockArea Example")
    main_window = MainWindow()
    pg.exec()