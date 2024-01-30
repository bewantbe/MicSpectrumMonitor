#!/usr/bin/env python3

# Usage:
#   python3 recorder_gui.py

# Contact: xyy <bewantbe@gmail.com>
# Github: https://github.com/bewantbe/MicSpectrumMonitor

import os
import shutil
import datetime
import time
import queue
import wave
import math
import threading

import logging
# enable logging
logging.basicConfig(
    level=logging.DEBUG
)

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
    analyzerData,
    FPSLimiter
)

from control_pannel import Ui_Dock4  # TODO: change "from PyQt6" to "from pyqtgraph.Qt"

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
  - extend analyzerData to multi-channels: done
  - extend plots
    + done waveform
    + done spectrum
    + done RMS
    + spectrogram -- cancelled
* Add RMS curve plot.
  - done
* Reduce plot margin.
  - done
* Add limit to FPS.
  - done.
* Allow Log freq axis mode.
  - done spectrum plot.
* Better user interaction design
  - start/stop recording
    + done
  - Show the recording time we saved.
    + done
  - Show possible recording time left.
    + done
* Add units for axis, and get scale prefix, set grid for some plot.
  - done
* longer update time for remaining space for rec.
  - done
* callback to monitoring/stop
  - done
* callback to device
  - refactor audio data pipeline
  - full restart
* callback to select channels
* callback to sampling rate
* callback to FFT length
* callback to averaging
* Test AD7606C
* Spectrogram plot log mode.
* Add show FPS, ref to the fps counter design in pyqtgraph example
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
        self.sample_rate = sample_rate
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
    
    def get_n_frame(self):
        return self.n_frame

    def get_t(self):
        return self.n_frame / self.sample_rate

    def get_n_byte(self):
        return self.n_frame * self.frame_bytes

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
        self.n_frame += data.nbytes // self.frame_bytes
    
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

def GetColorMapLut(n_point, cm_name = 'CET-C6'):
    """
    Good colormaps for our purpose:
    From Color Maps example in
    ```
    import pyqtgraph.examples
    pyqtgraph.examples.run()
    ```
    CET-C6s             (loop)
    CET-C6              (loop)
    CET-R2              (linear)
    PAL-relaxed         (loop)
    PAL-relaxed_bright  (loop)
    """
    is_loop_dic = {
        'CET-C6s': True,
        'CET-C6': True,
        'CET-R2': False,
        'PAL-relaxed': True,
        'PAL-relaxed_bright': True,
    }
    cm_loop = is_loop_dic[cm_name]
    cm = pg.colormap.get(cm_name)
    c_stop = (1 - 1.0 / n_point) if cm_loop else 1.0
    lut = cm.getLookupTable(start = 0.0, stop = c_stop, nPts = n_point)
    return lut

def time_to_HHMMSSm(t):
    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:04.1f}".format(int(hours), int(minutes), seconds)

class WaveformPlot:
    def __init__(self):
        self.auto_range = True
    
    def init_to_widget(self):
        #dock1.hideTitleBar()
        plot_widget = pg.PlotWidget(title="Waveform")
        plot_item = plot_widget.plot(
            np.random.normal(size=100),
            name = 'ch1')
        plot_widget.getPlotItem().getAxis('left').setWidth(50)
        self.plot_widget = plot_widget
        self.plot_items = [plot_item]
        return plot_widget

    def init_param(self, analyzer, sz_hop):
        self.sz_chunk = analyzer.sz_chunk
        self.sz_hop = sz_hop              # overlap = sz_chunk - sz_hop
        self.n_channel = analyzer.n_channel
        self.lut = GetColorMapLut(self.n_channel)
        self.plot_items[0].setPen(
            self.lut[0]
            #pen = pg.mkPen(color=(i, self.n_channel), width=1),
            #pen = (i, self.n_channel),
        )
        for i in range(1, self.n_channel):
            pl = self.plot_widget.plot(
                np.random.normal(size=100),
                pen = self.lut[i],
                name = f'ch{i+1}')
            self.plot_items.append(pl)

    def config_plot(self):
        #self.plot_widget.getPlotItem().setLimits(xMin=-1, xMax=self.sz_chunk)
        if not self.auto_range:
            self.plot_widget.setRange(self.waveform_plot_range)

    @property
    def waveform_plot_range(self):
        rg = QtCore.QRectF(*map(float, [0, -1.0, self.sz_chunk, 2.0]))
        return rg  # x, y, width, height

    def update(self, volt):
        for i in range(self.n_channel):
            self.plot_items[i].setData(volt[:,i])

class SpectrumPlot:
    def __init__(self):
        self.log_mode = False
    
    def init_to_widget(self):
        #dock2.hideTitleBar()
        plot_widget = pg.PlotWidget()
        plot_item = plot_widget.plot(
            np.random.normal(size=100),
            name = 'ch1')
        plot_widget.setLabel('left', units='dB')
        plot_widget.setLabel('bottom', units='Hz')
        plot_widget.showGrid(x=True, y=True)
        self.plot_widget = plot_widget
        self.plot_items = [plot_item]
        return plot_widget

    def init_param(self, analyzer, sz_hop):
        self.sz_chunk = analyzer.sz_chunk
        self.sz_hop = sz_hop              # overlap = sz_chunk - sz_hop
        self.max_freq = analyzer.fqs[-1]
        self.x_freq = analyzer.fqs
        self.n_freq = len(self.x_freq)
        self.n_channel = analyzer.n_channel
        self.lut = GetColorMapLut(self.n_channel)
        self.plot_items[0].setPen(self.lut[0])
        for i in range(1, self.n_channel):
            pl = self.plot_widget.plot(
                np.random.normal(size=100),
                pen = self.lut[i],
                name = f'ch{i+1}')
            self.plot_items.append(pl)

    def set_log_mode(self, log_mode):
        self.log_mode = log_mode
        for i in range(self.n_channel):
            self.plot_items[i].setLogMode(x = self.log_mode, y = False)

    def config_plot(self):
        self.plot_widget.setRange(self.spectrum_plot_range)

    @property
    def spectrum_plot_range(self):
        rg = QtCore.QRectF(*map(float, [0, -120, self.max_freq, 120]))
        return rg  # x, y, width, height

    def update(self, fqs, spectrum_db):
        for i in range(self.n_channel):
            self.plot_items[i].setData(x = fqs, y = spectrum_db[:,i])
            #self.plot_items.setData(x = self.x_freq, y = spectrum_db)

class RMSPlot:
    def __init__(self):
        self.fixed_range = False
        self.t_duration_set = 6.0  # sec

    def init_to_widget(self):
        #dock2.hideTitleBar()
        plot_widget = pg.PlotWidget()
        plot_item = plot_widget.plot(
            -90 * np.ones(100),
            name = 'ch1')
        self.plot_widget = plot_widget
        self.plot_items = [plot_item]
        return plot_widget
    
    def init_param(self, analyzer, sz_hop):
        self.dB_max = analyzer.RMS_db_sine_inc
        self.dB_min = 20 * np.log10(2**(-15))     # assume 16-bit
        # for RMS data
        t_hop = sz_hop / analyzer.sample_rate
        self.rms_len = int(self.t_duration_set / t_hop)
        self.t_duration = self.rms_len * t_hop    # correct the duration
        self.loop_cursor = 0
        self.n_channel = analyzer.n_channel
        self.arr_rms_db = -90 * np.ones((self.rms_len, self.n_channel))
        self.arr_t = np.arange(self.rms_len) * t_hop
        # set plot color
        self.lut = GetColorMapLut(self.n_channel)
        self.plot_items[0].setPen(self.lut[0])
        for i in range(1, self.n_channel):
            pl = self.plot_widget.plot(
                -90 * np.ones(100),
                pen = self.lut[i],
                name = f'ch{i+1}')
            self.plot_items.append(pl)
        self.data_lock = threading.Lock()
    
    def config_plot(self):
        self.plot_widget.getPlotItem().setLimits(
            #xMin=0, xMax=self.t_duration,
            yMin=self.dB_min, yMax=self.dB_max + 5.0)
        self.plot_widget.setLabel('bottom', units='second')
        self.plot_widget.setLabel('left', units='dB')
        self.plot_widget.showGrid(x=True, y=True)
        if self.fixed_range:
            self.plot_widget.setRange(self.rms_plot_range)
    
    @property
    def rms_plot_range(self):
        rg = QtCore.QRectF(*map(float,
            [0, self.dB_min, self.t_duration_set, self.dB_max - self.dB_min]))
        return rg  # x, y, width, height
    
    def feed_rms(self, rms_db):
        with self.data_lock:
            self.arr_rms_db[self.loop_cursor,:] = rms_db
        self.loop_cursor = (self.loop_cursor + 1) % self.rms_len
    
    def update(self):
        with self.data_lock:
            arr = self.arr_rms_db.copy()       # is this minimize the race condition?
        for i in range(self.n_channel):
            self.plot_items[i].setData(x = self.arr_t, y = arr[:,i])

class SpectrogramPlot:
    def __init__(self):
        self.log_mode = False
        self.spam_bmp_t_duration_set = 6.0  # sec

    def init_to_widget(self):
        # called in main thread init
        glayout_widget = pg.GraphicsLayoutWidget()  # use GraphicsLayoutWidget instead of GraphicsView
        plot_item = glayout_widget.addPlot()
        img_item = pg.ImageItem()
        img_item.setImage(np.random.normal(size=(100,100)))
        plot_item.addItem(img_item)
        self.glayout_widget = glayout_widget
        self.img_item = img_item
        self.plot_item = plot_item
        # TODO: add color bar
        return glayout_widget  # for add to dock: dock3.addWidget(glayout_widget)
    
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
            self.plot_item.setLabel('left', units='Hz')
            self.plot_item.setLabel('bottom', units='second')
            #self.plot_item.setRange(xRange=[0, self.spam_bmp_t_duration], yRange=[0, self.max_freq])
            #x_axis = self.plot_item.getAxis('bottom')
            #y_axis = self.plot_item.getAxis('left')
    
    def feed_spectrum(self, spectrum):
        # separate computation (here) and plot (in update()) in different threads
        with self.spam_lock:
            self.spam_bmp[self.spam_loop_cursor,:] = spectrum[:,0]
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
            self.img_item.setImage(spam_bmp,
                rect=[0, 0, self.spam_bmp_t_duration, self.max_freq])

# copy from M3F20xm.py
def pretty_num_unit(v, n_prec = 4):
    # print 100000 as 100k, etc.
    if v == 0:
        return '0'
        #return f'%.{n_prec}f'%(v,)
    scale_st = {
        -1:'m', -2:'u', -3:'n', -4:'p', -5:'f', -6:'a',
        0:'', 1:'k', 2:'M', 3:'G', 4:'T', 5:'P', 6:'E'
    }
    sign = -1 if v < 0 else 1
    v = v * sign
    scale = int(math.floor(math.log(v) / math.log(1000.0)))
    if scale > 6:
        scale = 6
    if scale < -6:
        scale = -6
    v = v * 1000.0 ** (-scale)
    st = f'%.{n_prec}g%s'%(v, scale_st[scale])
    return st

class AnalyzerParameters:
    def __init__(self):
        self._ana_conf_keys = [
            'size_chunk', 'size_hop', 'n_ave', 'use_dBA']

    def load_device_default(self, device_name):
        if device_name == 'mic':
            self.load_mic_default()
        elif device_name == 'AD7606C':
            self.load_AD7606C_default()
        else:
            raise ValueError(f'Unknown device name "{device_name}"')

    def ui_connect(self, main_wnd):
        self.main_wnd = main_wnd
        # first update
        self.update_channel_selected()
        # auto update the channel selected
        self.main_wnd.ui_dock4.lineEdit_ch.textChanged.connect(self.update_channel_selected)
        self.update_to_ui()
    
    def update_to_ui(self):
        # if we have sampler_id defined, means we are initialized
        if not hasattr(self, 'sampler_id'):
            return
        # TODO: update the UI
        ui = self.main_wnd.ui_dock4
        ui.comboBox_dev.setCurrentText(self.device_name)
        ui.comboBox_sr.clear()
        ui.comboBox_sr.addItems(self.dic_sample_rate.keys())
        ui.comboBox_sr.setCurrentText(pretty_num_unit(self.sample_rate) + 'Hz')

    def update_channel_selected(self):
        channel_selected_text = self.main_wnd.ui_dock4.lineEdit_ch.text()
        try:
            # note: channel index starts from 0 in the code, but starts from 1 in the UI
            chs = [int(c) - 1 for c in channel_selected_text.split(',')]
        except ValueError:
            # set text box to light-red background
            self.main_wnd.ui_dock4.lineEdit_ch.setStyleSheet("background-color: LightSalmon")
            return
        else:
            # set normal background
            self.main_wnd.ui_dock4.lineEdit_ch.setStyleSheet("background-color: white")
        self.channel_selected = chs

    def get_adc_conf(self):
        adc_conf = {}
        for k in self._adc_conf_keys:
            adc_conf[k] = getattr(self, k)
        return adc_conf

    def get_ana_conf(self):
        ana_conf = {}
        for k in self._ana_conf_keys:
            ana_conf[k] = getattr(self, k)
        return ana_conf

    def load_mic_default(self):
        # for mic / ADC
        self.sampler_id   = 'mic'
        self.device       = 'default'
        self.device_name  = 'System mic'
        self.sample_rate  = 48000
        self.n_channel    = 2
        self.value_format = 'S16_LE'
        self.bit_depth    = 16       # assume always S16_LE
        self.periodsize   = 1024
        # allowable values
        self.dic_sample_rate = {    # might be generated
            '48kHz': 48000,
            '44.1kHz': 44100,
            '32kHz': 32000,
            '16kHz': 16000,
            '8kHz': 8000,
        }
        # pipeline
        self.channel_selected = [1, 2]
        self.data_queue_max_size = 1000
        # for FFT analyzer
        self.size_chunk   = 1024
        self.size_hop     = self.size_chunk // 2
        self.n_ave        = 1
        self.use_dBA      = False
        # TODO: calibration_path
        self._adc_conf_keys = [
            'sampler_id', 'device', 'sample_rate', 'n_channel',
            'value_format', 'periodsize']

    def load_AD7606C_default(self):
        # for mic / ADC
        self.sampler_id   = 'ad7606c'
        self.device       = 'default'
        self.device_name  = 'AD7606C'
        self.sample_rate  = 48000
        self.n_channel    = 8
        self.value_format = 'S16_LE' # depends on the range setup
        self.bit_depth    = 16       # assume always S16_LE
        self.periodsize   = 4800
        self.dic_sample_rate = {    # might be generated
            '48kHz' : 48000,
            '250kHz': 250000,
            '500kHz': 500000,
        }
        # pipeline
        self.channel_selected = [1, 2]
        self.data_queue_max_size = 1000
        # for FFT analyzer
        self.size_chunk   = 1024
        self.size_hop     = self.size_chunk // 2
        self.n_ave        = 1
        self.use_dBA      = False
        self._adc_conf_keys = [
            'sampler_id', 'sample_rate', 'periodsize']

class AudioSaverManager:
    def __init__(self, main_wnd):
        self.main_wnd = main_wnd
        self.ui_dock4 = main_wnd.ui_dock4
        self.ana_param = main_wnd.ana_param
        ## for audio saving thread
        # Data flow: process(chunk_process_thread) -> wav_data_queue -> wav_writer
        self.wav_data_queue = None
        self.wav_writer_thread = None
        self.ui_dock4.label_rec_time_timer = None
        # utilizer
        self._disk_space_update_interval = 10.0
        self._last_update_disk_space_left = time.time() - self._disk_space_update_interval
        self._disk_space_left = float('inf')

    def init_ui(self):
        self.ui_dock4.pushButton_rec.clicked.connect(self.start_stop_saving)
        self.ui_dock4.toolButton_path.clicked.connect(self.open_file_dialog)
        self.wav_save_path = None

    def open_file_dialog(self):
        self.wav_save_path = QtWidgets.QFileDialog.getSaveFileName()[0]
        self.ui_dock4.lineEdit_wavpath.setText(self.wav_save_path)

    def is_rec_on(self):
        # Test if the audio saving (to WAV) is on
        if self.wav_writer_thread is None:
            return False
        return self.wav_writer_thread.is_running()

    def feed_data(self, data_chunk):
        self.wav_data_queue.put(data_chunk)   # TODO: should we copy the data?

    def close(self):
        if self.wav_writer_thread is not None:
            self.wav_writer_thread.stop()

    # TODO: maybe we need a saver manager class
    def start_stop_saving(self):
        if self.is_rec_on():
            # stop the recording
            self.stop_audio_saving()
            # set the button text to 'Start recording', make background grey
            self.ui_dock4.pushButton_rec.setText('Start recording')
            self.ui_dock4.pushButton_rec.setStyleSheet("background-color: grey")
            # invalidate the file name in the lineedit, put old text to the placeholder
            self.ui_dock4.lineEdit_wavpath.setText('')
            self.ui_dock4.lineEdit_wavpath.setPlaceholderText(self.wav_save_path)
            self.wav_save_path = None
            # Disable time recorded message
            if self.ui_dock4.label_rec_time_timer is not None:
                self.ui_dock4.label_rec_time_timer.stop()
                self.ui_dock4.label_rec_time_timer = None
            # set rec time label color to grey
            self.ui_dock4.label_rec_remain.setStyleSheet("color: grey")
        else:
            # start the recording
            ok = self.start_audio_saving()
            if not ok:
                # failed to start... hmmm
                logging.error("Failed to start audio saving. Path: %s", self.wav_save_path)
                return
            # set the button text to 'Stop recording', make background red
            self.ui_dock4.pushButton_rec.setText('Stop recording')
            self.ui_dock4.pushButton_rec.setStyleSheet("background-color: LightSalmon")
            # Enable time recorded message
            # Update once a second, by using a timer
            self.ui_dock4.label_rec_remain.setText('Rec: ')
            self.ui_dock4.label_rec_remain.show()
            self.ui_dock4.label_rec_time_timer = QtCore.QTimer()
            self.ui_dock4.label_rec_time_timer.timeout.connect(self.update_rec_time)
            self.ui_dock4.label_rec_time_timer.start(100)
            self.ui_dock4.label_rec_remain.setStyleSheet("color: black")
    
    def update_rec_time(self):
        if self.ui_dock4.label_rec_time_timer is None:
            logging.warning('Why do you call me')
            return
        if (self.wav_writer_thread is None) or \
           (self.wav_writer_thread.writer is None):
            logging.warning('Nothing can be provided.')
            return
        t_rec = self.wav_writer_thread.writer.get_t()
        t_rec_str = time_to_HHMMSSm(t_rec)
        # calculate how far we are form 4GB limit, in terms of time
        sz_4g_left = 3.99 * 2**30 - self.wav_writer_thread.writer.get_n_byte()
        sz_per_sec = self.wav_writer_thread.writer.frame_bytes * self.wav_writer_thread.writer.sample_rate
        t_4g_left = sz_4g_left / sz_per_sec
        # get file system space left, TODO: query it less often, like every 10 sec
        t_disk_left = self._lazy_update_disk_space_left() / sz_per_sec
        t_min_left = min(t_4g_left, t_disk_left)
        t_left_str = time_to_HHMMSSm(t_min_left)
        # time rec and time left
        self.ui_dock4.label_rec_remain.setText('Rec: ' + t_rec_str + '  (left: ' + t_left_str + ')')

    def _lazy_update_disk_space_left(self):
        t_now = time.time()
        if t_now < self._last_update_disk_space_left + self._disk_space_update_interval:
            # no need to update
            return self._disk_space_left
        self._last_update_disk_space_left = t_now
        sav_dir = os.path.dirname(os.path.abspath(self.wav_save_path))
        disk_usage = shutil.disk_usage(sav_dir)
        self._disk_space_left = disk_usage.free
        return self._disk_space_left

    def start_audio_saving(self):
        # Ensure we have a file name anyway
        self.wav_save_path = self.ui_dock4.lineEdit_wavpath.text()
        if (self.wav_save_path is None) or (self.wav_save_path == ''):
            # set file name by date and time
            now = datetime.datetime.now()
            self.wav_save_path = now.strftime("untitled_%Y-%m-%d_%H%M%S.wav")
            self.ui_dock4.lineEdit_wavpath.setText(self.wav_save_path)
        # For setup the wav writer
        wav_saver_conf = {
            'wav_path': self.wav_save_path,
            'n_channel': len(self.ana_param.channel_selected),
            'bit_depth': self.ana_param.bit_depth,
            'sample_rate': self.ana_param.sample_rate
        }
        # Get a new queue anyway, TODO: avoid memory leak
        self.wav_data_queue = queue.Queue(self.ana_param.data_queue_max_size)
        self.wav_writer_thread = RecorderWriteThread(
            self.wav_data_queue, AudioSaver(), wav_saver_conf)
        self.wav_writer_thread.start()
        # check bad file name
        print('waiting recorder to start...', end='')
        self.wav_writer_thread._initilized.wait()
        print('Done.')
        if not self.wav_writer_thread.is_running():
            self.main_wnd.simple_message_box(f"Failed to open file {self.wav_save_path}.")
            return False
        return True
    
    def stop_audio_saving(self):
        self.wav_writer_thread.stop()
        self.wav_writer_thread.join()
        # pop up a message box saying the file is saved
        #self.main_wnd.simple_message_box(f"The recording is saved to {self.wav_save_path}.")
    
class AudioPipeline(pg.QtCore.QObject):
    """Holding audio pipeline for the recorder."""

    # Ref. https://www.pythonguis.com/tutorials/pyqt6-signals-slots-events/
    # See pyqtgraph's example console_exception_inspection.py.
    signal_update_graph = pg.QtCore.Signal(object)

    def __init__(self):
        super().__init__()   # for QObject and the signal
        pg.QtCore.QObject.__init__(self)

    def init(self, ana_param, cb_update_graph, audio_saver_manager):
        self.audio_saver_manager = audio_saver_manager
        adc_conf = ana_param.get_adc_conf()

        ## setup data gernerator and analyzer
        # Data flow: mic source -> buf_queue -> process{analyzer, signal to plot}
        self.buf_queue = queue.Queue(ana_param.data_queue_max_size)
        self.rec_thread = recThread('recorder', self.buf_queue, adc_conf)
        # TODO: allow recThread to accept multiple queues (like pipelines) for multiple downstreams
        #       plan two: in sampleChunkThread, we setup another callback for receiving raw data

        ana_conf = ana_param.get_ana_conf()

        # init FFT Analyzer
        self.analyzer_data = analyzerData(
            ana_conf['size_chunk'], adc_conf['sample_rate'], ana_conf['n_ave'],
            len(ana_param.channel_selected))
        self.analyzer_data.use_dBA = ana_conf['use_dBA']
        
        # signals for calling self.proc_analysis_plot
        self.signal_update_graph.connect(cb_update_graph,
                                         pg.QtCore.Qt.ConnectionType.QueuedConnection)

        sz_chunk = ana_conf['size_chunk']
        sz_hop   = ana_conf['size_hop']
        self.chunk_process_thread = sampleChunkThread('chunking',
            self.proc_analysis_plot, self.buf_queue,
            ana_param.channel_selected,
            sz_chunk, sz_hop,
            callback_raw = self.proc_orig_data)
    
    def is_device_on(self):
        # Test if the mic/ADC is on
        if self.rec_thread is None:
            return False
        return self.rec_thread.b_run

    def start(self, cb_update_rms, cb_update_spectrum):
        self.cb_update_rms = cb_update_rms
        self.cb_update_spectrum = cb_update_spectrum
        self.rec_thread.start()
        self.chunk_process_thread.start()

    def close(self):
        self.rec_thread.b_run = False
        self.chunk_process_thread.b_run = False

    def proc_orig_data(self, data_chunk):
        ## usually called from data processing thread
        # if we are ready to write data, then do it (but
        # do not cost too much CPU/IO time, let the other thread to do the actual work)
        if self.audio_saver_manager.is_rec_on():
            self.audio_saver_manager.feed_data(data_chunk)

    def proc_analysis_plot(self, data_chunk):
        ## usually called from data processing thread
        # analysing
        self.analyzer_data.put(data_chunk)
        fqs = self.analyzer_data.fqs
        rms_db = self.analyzer_data.get_RMS_dB()
        volt = self.analyzer_data.get_volt()
        spectrum_db = self.analyzer_data.get_spectrum_dB()
        self.cb_update_rms(rms_db)
        self.cb_update_spectrum(spectrum_db)
        # plot
        self.signal_update_graph.emit((rms_db, volt, fqs, spectrum_db))

class MainWindow(QtWidgets.QMainWindow):
    """Main Window for monitoring and recording the Mic/ADC signals."""

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        ## setup window
        area = DockArea()
        self.setCentralWidget(area)
        self.resize(1000,500)
        self.setWindowTitle('Spectrum Analyzer')

        ## setup layout
        dock1 = Dock("Waveform", size=(100, 200))
        dock2 = Dock("Spectrum", size=(500, 300))  # removed closable
        dock3 = Dock("Spectrogram", size=(500,400))
        dock5 = Dock("RMS", size=(500,200))
        dock4 = Dock("Control Pannel", size=(500,250))
        area.addDock(dock1, 'left')
        area.addDock(dock2, 'bottom', dock1)
        area.addDock(dock4, 'bottom', dock2)
        area.addDock(dock3, 'right')
        area.addDock(dock5, 'bottom', dock3)
        self.area = area
        self.dock1 = dock1
        self.dock2 = dock2
        self.dock3 = dock3
        self.dock4 = dock4
        self.dock5 = dock5

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

        ## Dock 5
        self.rms_plot = RMSPlot()
        dock5.addWidget(self.rms_plot.init_to_widget())

        ## Dock 4
        # See also: https://build-system.fman.io/qt-designer-download
        #           https://doc.qt.io/qt-6/designer-quick-start.html
        # pyuic6 control_pannel.ui -o control_pannel.py
        # python -m PyQt6.uic.pyuic -o output.py -x input.ui
        ui_dock4 = Ui_Dock4()
        ui_dock4.setupUi(self.dock4)
        self.ui_dock4 = ui_dock4

        ui_dock4.pushButton_mon.clicked.connect(self.start_stop_monitoring)
        ui_dock4.pushButton_screenshot.clicked.connect(self.take_screen_shot)

        # setup audio input and basic parameters
        pcm_device = 'mic'
        self.ana_param = AnalyzerParameters()
        self.ana_param.load_device_default(pcm_device)
        self.ana_param.ui_connect(self)                  # put data on the GUI

        # audio saver relies on ana_param
        self.audio_saver_manager = AudioSaverManager(self)
        self.audio_saver_manager.init_ui()

        # Connect the custom closeEvent
        self.closeEvent = self.custom_close_event

        self.show()

        self.fps_lim = FPSLimiter(30)        # for limiting update_graph

        # core audio pipeline is managed here
        self.audio_pipeline = AudioPipeline()
        self.audio_pipeline.init(self.ana_param, self.update_graph, self.audio_saver_manager)

        ## setup plots
        sz_hop = self.ana_param.size_hop
        analyzer_data = self.audio_pipeline.analyzer_data
        self.waveform_plot.init_param(analyzer_data, sz_hop)
        self.waveform_plot.config_plot()
        self.spectrum_plot.init_param(analyzer_data, sz_hop)
        self.spectrum_plot.config_plot()
        self.rms_plot.init_param(analyzer_data, sz_hop)
        self.rms_plot.config_plot()
        self.spectrogram_plot.init_param(analyzer_data, sz_hop)
        self.spectrogram_plot.config_plot()

        self.b_monitor_on = True

        self.audio_pipeline.start(
            self.rms_plot.feed_rms,
            self.spectrogram_plot.feed_spectrum
        )

    def is_monitoring_on(self):
        # Test if the monitor is on
        if not self.audio_pipeline.is_device_on():
            return False
        return self.b_monitor_on

    def start_stop_monitoring(self):
        if self.is_monitoring_on():
            # stop the monitoring
            self.b_monitor_on = False
            # set the button text to 'Start monitoring', make background grey
            self.ui_dock4.pushButton_mon.setText('Start monitoring')
            self.ui_dock4.pushButton_mon.setStyleSheet("background-color: grey")
        else:
            # start the monitoring
            self.b_monitor_on = True
            # set the button text to 'Stop monitoring', make background red
            self.ui_dock4.pushButton_mon.setText('Stop monitoring')
            self.ui_dock4.pushButton_mon.setStyleSheet("background-color: white")

    def simple_message_box(self, msg_text):
        msg = QtWidgets.QMessageBox()
        #msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(msg_text)
        msg.setWindowTitle("Recording saved")
        msg.setStandardButtons(PyQt6.QtWidgets.QMessageBox.StandardButton.Ok)
        msg.exec()
    
    # TODO: annotate callbacks using decorator
    def update_graph(self, obj):
        if not self.b_monitor_on:
            return
        if not self.fps_lim.checkFPSAllow():
            return
        # usually called from main thread
        rms_db, volt, fqs, spectrum_db = obj
        # ploting
        self.waveform_plot.update(volt)
        self.spectrum_plot.update(fqs, spectrum_db)
        self.rms_plot.update()
        self.spectrogram_plot.update()
    
    def custom_close_event(self, event):
        self.audio_pipeline.close()
        self.audio_saver_manager.close()
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
