#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

## PyAudioSpectra

# Usage:
#   # cd to Root directory of this project
#   python -m audiospectra
# Or:
#   # cd audiospectra/
#   python recorder_gui.py

# This file contains GUI part of the audio spectrum anlyzer.

# Contact: xyy <bewantbe@gmail.com>
# Github: https://github.com/bewantbe/MicSpectrumMonitor

# recommend to use pyqt6 with pyqtgraph in this project

import os
import shutil
import json
import datetime
import time
from time import perf_counter
from copy import deepcopy
import array
import queue
import wave
import math
import threading

#import pprint
import logging

import numpy as np

import pyqtgraph as pg
#from pyqtgraph.console import ConsoleWidget
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import (
    QtCore,
    QtWidgets
)

# enable logging
logging.basicConfig(
    format = '%(levelname)s:%(message)s',
    level = logging.DEBUG
)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
#logger = logging.getLogger('gui')

from .record_wave import (
    recThread,
    sampleChunkThread,
    analyzerData,
    FPSLimiter
)

from .control_pannel import Ui_Dock4  # TODO: change "from PyQt6" to "from pyqtgraph.Qt"

DEFAULT_CONF_FILE = '.analyzer.conf'

def mk_list(a):
    if isinstance(a, list):
        return a
    return [a]

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
        elif isinstance(data, array.array):
            if data.typecode == 'H':
                s = np.array(data, dtype = np.uint16)
                s = s - 32768   # then can be interpreted as signed int16
                data = s.data
            else:
                # assume 'h', make it a bytes-like object
                data = data.tobytes()
        else:
            raise ValueError("data type not supported")

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

class CustomPlotWidget(pg.PlotWidget):
    """Might be used to replace PlotWidget to measure plot performance."""
    def __init__(self, *args, fps_counter_notify, **kwargs):
        super().__init__(*args, **kwargs)
        self._fps_counter_notify = fps_counter_notify
        self._tick_count = 0
        self._tick_count_old = 0
        self._lock_cnt = threading.Lock()

    def frame_tick_inc(self):
        with self._lock_cnt:
            self._tick_count += 1

    def paintEvent(self, event):
        super().paintEvent(event)
        with self._lock_cnt:
            n_advance = self._tick_count - self._tick_count_old
            self._tick_count_old = self._tick_count
        for ii in range(n_advance):
            # do not count repeated paint events in the same frame
            # but also count skipped paint event(s)
            self._fps_counter_notify()

class WaveformPlot:
    def __init__(self):
        self.auto_range = True
    
    #def init_to_widget(self):
    def init_to_widget(self, fps_counter_notify):
        #dock1.hideTitleBar()
        #plot_widget = pg.PlotWidget()
        plot_widget = CustomPlotWidget(fps_counter_notify = fps_counter_notify)
        plot_data_item = plot_widget.plot(
            np.random.normal(size=100),
            name = 'ch1')
        # Set the margins of the layout (left, top, right, bottom)
        plot_item = plot_widget.getPlotItem()
        plot_item.layout.setContentsMargins(0, 10, 10, 0)
        plot_item.getAxis('left').setWidth(50)
        plot_item.setLimits(minYRange=1/65536)
        self.plot_widget = plot_widget
        self.plot_data_items = [plot_data_item]  # may try PlotCurveItem()
        return plot_widget

    def init_param(self, analyzer, sz_hop):
        self.sz_chunk = analyzer.sz_chunk
        self.sz_hop = sz_hop              # overlap = sz_chunk - sz_hop
        self.n_channel = analyzer.n_channel
        self.lut = GetColorMapLut(self.n_channel)
        # allow re-init by remove old plots
        if len(self.plot_data_items) > 1:
            # remove old plots
            for i in range(1, len(self.plot_data_items)):
                self.plot_widget.removeItem(self.plot_data_items[i])
            self.plot_data_items = self.plot_data_items[0:1]
        # set plots for each channel
        self.plot_data_items[0].setPen(self.lut[0])
        for i in range(1, self.n_channel):
            pl = self.plot_widget.plot(
                np.random.normal(size=100),
                pen = self.lut[i],
                name = f'ch{i+1}')
            self.plot_data_items.append(pl)

    def config_plot(self):
        #self.plot_widget.getPlotItem().setLimits(xMin=-1, xMax=self.sz_chunk)
        if not self.auto_range:
            self.plot_widget.setRange(self.waveform_plot_range)

    @property
    def waveform_plot_range(self):
        rg = QtCore.QRectF(*map(float, [0, -1.0, self.sz_chunk, 2.0]))
        return rg  # x, y, width, height

    def update(self, volt):
        self.plot_widget.frame_tick_inc()
        for i in range(self.n_channel):
            self.plot_data_items[i].setData(volt[:,i])

class SpectrumPlot:
    def __init__(self):
        self.log_mode = False
        self.data_feed_count = 0
        self.data_feed_count_old = 0
    
    def init_to_widget(self):
        #dock2.hideTitleBar()
        plot_widget = pg.PlotWidget()
        plot_data_item = plot_widget.plot(
            np.random.normal(size=100),
            name = 'ch1')
        plot_widget.setLabel('left', units='dB')
        plot_widget.setLabel('bottom', units='Hz')
        plot_widget.showGrid(x=True, y=True)
        # Set the margins of the layout (left, top, right, bottom)
        plot_widget.getPlotItem().layout.setContentsMargins(0, 10, 10, 0)
        self.plot_widget = plot_widget
        self.plot_data_items = [plot_data_item]
        return plot_widget

    def init_param(self, analyzer, sz_hop):
        self.sz_chunk = analyzer.sz_chunk
        self.sz_hop = sz_hop              # overlap = sz_chunk - sz_hop
        self.max_freq = analyzer.fqs[-1]
        self.x_freq = analyzer.fqs
        self.n_freq = len(self.x_freq)
        self.n_channel = analyzer.n_channel
        self.lut = GetColorMapLut(self.n_channel)
        self.lower_bound_spectrum_db = analyzer.lower_bound_spectrum_dB()
        # allow re-init by remove old plots
        if len(self.plot_data_items) > 1:
            # remove old plots
            for i in range(1, len(self.plot_data_items)):
                self.plot_widget.removeItem(self.plot_data_items[i])
            self.plot_data_items = self.plot_data_items[0:1]
        # set plots for each channel
        self.plot_data_items[0].setPen(self.lut[0])
        for i in range(1, self.n_channel):
            pl = self.plot_widget.plot(
                np.random.normal(size=100),
                pen = self.lut[i],
                name = f'ch{i+1}')
            self.plot_data_items.append(pl)
        self.saved_spectrum_db = self.lower_bound_spectrum_db * \
                                 np.ones((self.n_freq, self.n_channel))
        self.data_lock = threading.Lock()

    def set_log_mode(self, log_mode):
        self.log_mode = log_mode
        for i in range(self.n_channel):
            self.plot_data_items[i].setLogMode(x = self.log_mode, y = False)

    def config_plot(self):
        self.plot_widget.setRange(self.spectrum_plot_range)

    @property
    def spectrum_plot_range(self):
        rg = QtCore.QRectF(*map(float, [
            0, self.lower_bound_spectrum_db,
            self.max_freq, 5 - self.lower_bound_spectrum_db]))
        return rg  # x, y, width, height

    def feed_spectrum(self, spectrum_db):
        with self.data_lock:
            self.saved_spectrum_db[:] = spectrum_db
            self.data_feed_count += 1

    def update(self, fqs = None, spectrum_db = None):
        if (fqs is not None) and (spectrum_db is not None):
            for i in range(self.n_channel):
                self.plot_data_items[i].setData(x = fqs, y = spectrum_db[:,i])
        else:
            with self.data_lock:
                n_frames_feed = self.data_feed_count - self.data_feed_count_old
                if n_frames_feed == 0:
                    return
                self.data_feed_count_old = self.data_feed_count
                sp_db = self.saved_spectrum_db.copy()
            for i in range(self.n_channel):
                self.plot_data_items[i].setData(x = self.x_freq, y = sp_db[:,i])

class RMSPlot:
    def __init__(self):
        self.fixed_range = False
        self.data_feed_count = 0
        self.data_feed_count_old = 0

    def init_to_widget(self):
        #dock2.hideTitleBar()
        plot_widget = pg.PlotWidget()
        plot_data_item = plot_widget.plot(
            -90 * np.ones(100),
            name = 'ch1')
        # Set the margins of the layout (left, top, right, bottom)
        plot_widget.getPlotItem().layout.setContentsMargins(0, 10, 10, 0)
        self.plot_widget = plot_widget
        self.plot_data_items = [plot_data_item]
        return plot_widget
    
    def init_param(self, analyzer, sz_hop, spectrogram_duration):
        self.dB_max = analyzer.RMS_db_sine_inc
        self.dB_min = analyzer.lower_bound_RMS_dB()
        # for RMS data
        self.n_ave = analyzer.ave_num
        self.t_duration_set = spectrogram_duration
        t_hop = sz_hop / analyzer.sample_rate
        self.rms_len = int(self.t_duration_set / t_hop)
        self.t_duration = self.rms_len * t_hop    # correct the duration
        self.loop_cursor = 0
        self.n_channel = analyzer.n_channel
        self.arr_rms_db = self.dB_min * np.ones((self.rms_len, self.n_channel))
        self.arr_t = np.arange(self.rms_len) * t_hop
        # set plot color
        self.lut = GetColorMapLut(self.n_channel)
        # allow re-init by remove old plots
        if len(self.plot_data_items) > 1:
            # remove old plots
            for i in range(1, len(self.plot_data_items)):
                self.plot_widget.removeItem(self.plot_data_items[i])
            self.plot_data_items = self.plot_data_items[0:1]
        # set plots for each channel
        self.plot_data_items[0].setPen(self.lut[0])
        for i in range(1, self.n_channel):
            pl = self.plot_widget.plot(
                self.dB_min * np.ones(100),
                pen = self.lut[i],
                name = f'ch{i+1}')
            self.plot_data_items.append(pl)
        self.data_lock = threading.Lock()
    
    def config_plot(self):
        self.plot_widget.getPlotItem().setLimits(
            #xMin=0, xMax=self.t_duration,
            yMin=self.dB_min - 6.0, yMax=self.dB_max + 6.0)
        self.plot_widget.setLabel('bottom', units='second')
        self.plot_widget.setLabel('left', units='dB')
        self.plot_widget.showGrid(x=True, y=True)
        if self.fixed_range:
            self.plot_widget.setRange(self.rms_plot_range)
    
    @property
    def rms_plot_range(self):
        rg = QtCore.QRectF(*map(float,
            [0, self.dB_min,
             self.n_ave * self.t_duration_set, self.dB_max - self.dB_min]))
        return rg  # x, y, width, height
    
    def set_n_ave(self, n_ave):
        self.n_ave = n_ave
    
    def feed_rms(self, rms_db):
        with self.data_lock:
            self.arr_rms_db[self.loop_cursor,:] = rms_db
            self.data_feed_count += 1
        self.loop_cursor = (self.loop_cursor + 1) % self.rms_len
    
    def update(self):
        with self.data_lock:
            n_frames_feed = self.data_feed_count - self.data_feed_count_old
            if n_frames_feed == 0:
                return
            self.data_feed_count_old = self.data_feed_count
            arr = self.arr_rms_db.copy()       # is this minimize the race condition?
        for i in range(self.n_channel):
            self.plot_data_items[i].setData(x = self.n_ave * self.arr_t, y = arr[:,i])

class SpectrogramPlot:
    def __init__(self):
        self.log_mode = False
        self.data_feed_count = 0
        self.data_feed_count_old = 0

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
    
    def init_param(self, analyzer, sz_hop, spectrogram_duration):
        """Allow re-init."""
        t_hop = sz_hop / analyzer.sample_rate
        self.max_freq = analyzer.fqs[-1]
        self.x_freq = analyzer.fqs
        self.n_freq = len(self.x_freq)
        self.n_ave = analyzer.ave_num
        self.lower_bound_spectrum_db = analyzer.lower_bound_spectrum_dB()
        self.spam_bmp_t_duration_set = spectrogram_duration  # sec
        self.spam_len = int(self.spam_bmp_t_duration_set / t_hop)
        self.spam_bmp_t_duration = self.spam_len * t_hop    # correct the duration
        self.spam_loop_cursor = 0
        if self.log_mode:
            # TODO: fix this according to the zooming
            self.spam_bmp = -120 * np.ones((self.spam_len, self.n_freq))
        else:
            self.spam_bmp = -120 * np.ones((self.spam_len, self.n_freq))
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
    
    def set_n_ave(self, n_ave):
        self.n_ave = n_ave
    
    def feed_spectrum(self, spectrum, idx_ch = 0):
        # separate computation (here) and plot (in update()) in different threads
        s = spectrum[:, idx_ch]
        s[s < self.lower_bound_spectrum_db-6] = self.lower_bound_spectrum_db-6
        with self.spam_lock:
            self.spam_bmp[self.spam_loop_cursor,:] = s
            self.spam_loop_cursor = (self.spam_loop_cursor + 1) % self.spam_len
            self.data_feed_count += 1
        return self.spam_bmp
    
    def get_spectrogram_bmp(self):
        if self.log_mode:
            return self.spam_bmp  # TODO: redraw the bmp according to zoom
        else:
            return self.spam_bmp

    def update(self):
        spam_bmp = self.get_spectrogram_bmp()
        with self.spam_lock:                   # TODO: maybe I don't need this lock
            n_frames_feed = self.data_feed_count - self.data_feed_count_old
            if n_frames_feed == 0:
                return
            self.data_feed_count_old = self.data_feed_count
            self.img_item.setImage(spam_bmp,
                rect=[0, 0, self.n_ave * self.spam_bmp_t_duration, self.max_freq])

def simple_message_box(msg_text):
    msg = QtWidgets.QMessageBox()
    #msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(msg_text)
    msg.setWindowTitle("Recording saved")
    msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
    msg.exec()

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
    def __init__(self, conf_path = DEFAULT_CONF_FILE):
        self._ana_conf_keys = [
            'size_chunk', 'size_hop', 'n_ave', 'use_dBA']
        self.devices_conf = deepcopy(self.devices_conf_default)
        self.conf_path = conf_path
        self.current_device = None
        self.last_device = None
        self.load_saved_conf()

    def load_saved_conf(self):
        """load conf from file to namespace"""
        if not os.path.isfile(self.conf_path):
            logging.info(f'No saved configuration file "{self.conf_path}" found.')
            return
        logging.info(f'loading conf from file "{self.conf_path}"')
        with open(self.conf_path, 'r') as f:
            d = json.load(f)
        self.devices_conf.update(d)
        self.last_device = self.devices_conf["current_device"]
        # if bad file, remove the conf file

    def dump_self_conf(self):
        """current namespace to conf, then conf to file"""
        logging.info(f'dump conf to file "{self.conf_path}"')
        self.self_to_dict()
        with open(self.conf_path, 'w') as f:
            json.dump(self.devices_conf, f, indent=2)

    def load_device(self, device_name):
        """save current to file, empty current, load to current"""
        if self.current_device is not None:
            self.dump_self_conf()
            # off-load old conf
            self.do_empty_self()
        # load new conf
        if device_name == 'System mic':  # deal with alias
            device_name = 'mic'
        self.dict_to_self(self.devices_conf[device_name])
        self.current_device = device_name

    def do_empty_self(self):
        """empty current namespace"""
        d = self.devices_conf[self.current_device]
        for k in d:
            if hasattr(self, k):
                delattr(self, k)
            else:
                logging.warning(f'Key "{k}" not found in analyzer parameters'
                                f'for device "{self.current_device}".')
        self.last_device = self.current_device
        self.current_device = None

    def dict_to_self(self, conf):
        """conf dict -> namespace"""
        for k, v in conf.items():
            setattr(self, k, v)

    def self_to_dict(self):
        # from current namespace to the dict
        d = self.devices_conf[self.current_device]  # by ref
        for k in d:
            if hasattr(self, k):
                d[k] = getattr(self, k)
            else:
                logging.warning(f'Key "{k}" not found in analyzer parameters.')
        self.devices_conf["current_device"] = self.current_device

    def update_to_ui(self, ui):
        """update UI from current device conf"""
        # if we have sampler_id defined, means we are initialized
        if not hasattr(self, 'sampler_id'):
            return
        # current device
        ui.comboBox_dev.setCurrentText(self.device_name)
        # sample rates allowed for the device
        ui.comboBox_sr.clear()
        ui.comboBox_sr.addItems(self.dic_sample_rate.keys())
        ui.comboBox_sr.setCurrentText(pretty_num_unit(self.sample_rate) + 'Hz')
        # channels selected
        st_chs = ','.join([str(c+1) for c in self.channel_selected])
        ui.lineEdit_ch.setText(st_chs)
        # max channel number
        ui.label_ch.setText(f'Channels (1~{self.n_channel}):')
        # fft length
        ui.comboBox_fftlen.setCurrentText(str(self.size_chunk))
        # averaging number
        ui.comboBox_nave.setCurrentText(str(self.n_ave))

    def update_channel_by_ui(self, line_edit_ch):
        #line_edit_ch = self.main_wnd.ui_dock4.lineEdit_ch
        channel_selected_text = line_edit_ch.text()
        try:
            # note: channel index starts from 0 in the code, but starts from 1 in the UI
            chs = [int(c) - 1 for c in channel_selected_text.split(',')]
            assert np.all(np.array(chs) < self.n_channel)
            assert np.all(np.array(chs) >= 0)
        except (ValueError, AssertionError):
            # set text box to light-red background
            line_edit_ch.setStyleSheet("background-color: LightSalmon")
            return False
        else:
            # set normal background
            line_edit_ch.setStyleSheet("background-color: white")
        self.channel_selected = chs
        return True

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

    def set_fft_len(self, fft_len):
        ratio = self.size_hop / self.size_chunk
        self.size_chunk = fft_len
        min_t = 1.0 / 30  # 30 FPS at most
        min_sp = int(2 ** np.ceil(np.log2(min_t * self.sample_rate)))
        self.periodsize = max(fft_len // 2, min_sp)
        self.size_hop = int(np.round(fft_len * ratio))

    def render_time_cost_estimation(self):
        """time cost coefficient for each frame"""
        # for wave, spectrum and rms
        t_wave = 2.5  + 1.85e-4 * self.n_channel / 2 * self.size_chunk
        t_spum = 1.55 + 0.79e-4 * self.n_channel / 2 * self.size_chunk
        #n_t = self.spectrogram_duration / (self.size_hop / self.sample_rate)
        t_rms  = 2.95 + 1.15e-4 * self.n_channel / 2 * self.size_chunk
        t_spam = 2.15 + 0.246e-4 * self.size_chunk
        return t_wave, t_rms, t_spum, t_spam

    devices_conf_default = {
        "mic": {
            # for mic / ADC
            "sampler_id"   : 'mic',
            "device"       : 'default',
            "device_name"  : 'System mic',
            "sample_rate"  : 48000,
            "n_channel"    : 2,
            "value_format" : 'S16_LE',
            "bit_depth"    : 16,      # assume always S16_LE
            "periodsize"   : 1024,    # usually half the chunk size
            # allowable values
            "dic_sample_rate" : {    # might be generated
                '48kHz': 48000,
                '44.1kHz': 44100,
                '32kHz': 32000,
                '16kHz': 16000,
                '8kHz': 8000,
            },
            # pipeline
            "channel_selected" : [0, 1],
            "data_queue_max_size" : 1000,
            # for FFT analyzer
            "size_chunk"   : 1024,
            "size_hop"     : 1024 // 2,
            "n_ave"        : 2,
            "spectrogram_duration" : 6.0,
            "use_dBA"      : False,
            # TODO: calibration_path
            "_adc_conf_keys" : [
                'sampler_id', 'device', 'sample_rate', 'n_channel',
                'value_format', 'periodsize'],
        },
        "AD7606C": {
            "sampler_id"   : 'ad7606c',
            "device"       : 'default',
            "device_name"  : 'AD7606C',
            "sample_rate"  : 500000,
            "n_channel"    : 8,
            "volt_range"   : [0, 5],
            "value_format" : 'S16_LE', # depends on the range setup
            "bit_depth"    : 16,       # assume always S16_LE
            "periodsize"   : 4096,
            "dic_sample_rate" : {    # might be generated
                '48kHz' : 48000,
                '250kHz': 250000,
                '500kHz': 500000,
            },
            # pipeline
            "channel_selected" : [0, 1, 2, 3, 4, 5, 6, 7],
            "data_queue_max_size" : 1000,
            # for FFT analyzer
            "size_chunk"   : 8192,
            "size_hop"     : 8192 // 2,
            "n_ave"        : 8,
            "spectrogram_duration" : 1.0,
            "use_dBA"      : False,
            "_adc_conf_keys" : [
                'sampler_id', 'sample_rate', 'periodsize', 'volt_range'],
        },
        "OSCA02": {
            "sampler_id"   : 'osca02',
            "device"       : 'default',
            "device_name"  : 'OSCA02',
            "sample_rate"  : 781000,
            "n_channel"    : 2,
            "volt_range"   : 5,
            "value_format" : 'U8',
            "bit_depth"    : 8,
            "periodsize"   : 64000,
            "indicate_discontinuous" : True,
            "dic_sample_rate" : {
                '100MHz': 100e6,
                '12.5MHz': 12.5e6,
                '781kHz': 781e3,
                '49kHz': 49e3,
            },
            "channel_selected" : [0, 1],
            "data_queue_max_size" : 1000,
            "size_chunk"   : 64000 // 32,
            "size_hop"     : 64000 // 32,
            "n_ave"        : 1,
            "spectrogram_duration" : 1.0,
            "use_dBA"      : False,
            "_adc_conf_keys" : [
                'sampler_id', 'sample_rate', 'periodsize', 'volt_range',
                'indicate_discontinuous'],
        }
    }

class AudioSaverManager:
    """ Manage the UI related to audio saving and manage wav saver."""
    def __init__(self):
        ## for audio saving thread
        # Data flow: process(chunk_process_thread) -> wav_data_queue -> wav_writer
        self.wav_data_queue = None
        self.wav_writer_thread = None
        # utilizer
        self._disk_space_update_interval = 10.0
        self._last_update_disk_space_left = time.time() - self._disk_space_update_interval
        self._disk_space_left = float('inf')

    def connect_button_events(self, ui_dock4):
        self.ui_dock4 = ui_dock4
        self.ui_dock4.label_rec_time_timer = None
        self.ui_dock4.pushButton_rec.clicked.connect(self.start_stop_saving)
        self.ui_dock4.toolButton_path.clicked.connect(self.open_file_dialog)

    def update_wav_param(self, ana_param):
        self.ana_param = ana_param

    def open_file_dialog(self):
        #self.wav_save_path = QtWidgets.QFileDialog.getSaveFileName()[0]
        self.wav_save_path = QtWidgets.QFileDialog.getExistingDirectory()

    @property
    def wav_save_path(self):
        _wav_save_path = self.ui_dock4.lineEdit_wavpath.text()
        return _wav_save_path

    @wav_save_path.setter
    def wav_save_path(self, path):
        self.ui_dock4.lineEdit_wavpath.setText(path)

    def is_rec_on(self):
        # Test if the audio saving (to WAV) is on
        if self.wav_writer_thread is None:
            return False
        return self.wav_writer_thread.is_running()

    def feed_data(self, data_chunk):
        # might be called from other thread
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
            # Just to avoid overwrite the same file unintentionally.
            wave_save_dir = os.path.dirname(os.path.abspath(self.wav_save_path))
            self.ui_dock4.lineEdit_wavpath.setPlaceholderText(self.wav_save_path)
            self.wav_save_path = wave_save_dir
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
        sav_abs_path = os.path.abspath(self.wav_save_path)
        if os.path.isdir(sav_abs_path):
            # if it is a directory, auto fill the file name by date and time
            now = datetime.datetime.now()
            self.wav_save_path = os.path.join(sav_abs_path,
                now.strftime("untitled_%Y-%m-%d_%H%M%S.wav"))
        elif not sav_abs_path.endswith('.wav'):
            # if file name not ends with known extension,
            # auto applend date and time
            now = datetime.datetime.now()
            self.wav_save_path = sav_abs_path + \
                now.strftime("_%Y-%m-%d_%H%M%S.wav")

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
            simple_message_box(f"Failed to open file {self.wav_save_path}.")
            return False
        return True
    
    def stop_audio_saving(self):
        self.wav_writer_thread.stop()
        self.wav_writer_thread.join()
        # pop up a message box saying the file is saved
        #simple_message_box(f"The recording is saved to {self.wav_save_path}.")
    
class AudioPipeline():
    """Holding audio pipeline for the recorder."""

    def __init__(self):
        #super().__init__()   # for QObject and the signal
        #pg.QtCore.QObject.__init__(self)
        self.rec_thread = None

    def init(self, ana_param):
        """Note: allow re-enter after full close."""
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
            len(ana_param.channel_selected), volt_zero='auto')
        self.analyzer_data.use_dBA = ana_conf['use_dBA']
        
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

    def set_n_ave(self, n_ave):
        self.analyzer_data.set_n_ave(n_ave)

    def start(self, audio_saver_manager, cb_update_rms, cb_update_spectrum, cb_plot):
        self.audio_saver_manager = audio_saver_manager
        self.cb_update_rms = cb_update_rms
        self.cb_update_spectrum = mk_list(cb_update_spectrum)
        self.cb_plot = cb_plot
        self.rec_thread.start()
        self.chunk_process_thread.start()

    def close(self, wait = False):
        self.rec_thread.b_run = False
        self.chunk_process_thread.b_run = False
        if not wait:
            return
        self.rec_thread.join()
        self.chunk_process_thread.join()
        # drink the rest of the queue
        while not self.buf_queue.empty():
            self.buf_queue.get_nowait()

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
        volt = self.analyzer_data.get_volt()
        if self.analyzer_data.has_new_data():
            rms_db = self.analyzer_data.get_FFT_RMS_dBA()
            fqs = self.analyzer_data.fqs
            spectrum_db = self.analyzer_data.get_spectrum_dB()
            for cb in self.cb_update_spectrum:
                cb(spectrum_db)
            self.cb_update_rms(rms_db)
        else:
            rms_db = None
            fqs = None
            spectrum_db = None
        # plot
        self.cb_plot((rms_db, volt, fqs, spectrum_db))
        
def PopOldEventsAndExecute(func):
    """Used to let the padding events to be executed in the main thread, then
       excute the `func` in the main thread.
       Usually used in restarting the audio pipeline, for clearing the padding
       graph plots, so that new (possibly incompatable) data can be forward to
       the plots.
    """
    QtCore.QTimer.singleShot(0, func)
    # alternative: QtCore.QCoreApplication.processEvents()
    # app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)

class MainWindow(QtWidgets.QMainWindow):
    """Main Window for monitoring and recording the Mic/ADC signals."""

    # Ref. https://doc.qt.io/qtforpython-6/overviews/signalsandslots.html
    #      https://doc.qt.io/qtforpython-6/PySide6/QtCore/Signal.html
    #      https://www.pythonguis.com/tutorials/pyqt6-signals-slots-events/
    # See also pyqtgraph's example console_exception_inspection.py.
    graph_data_updated = pg.QtCore.Signal(object)

    def __init__(self, *args, qapp, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.qapp = qapp

        ## setup window
        area = DockArea()
        self.setCentralWidget(area)
        self.resize(1000,500)
        self.setWindowTitle('Spectrum Analyzer')

        ## setup layout
        dock1 = Dock("Waveform", size=(100, 200), closable=True)
        dock2 = Dock("Spectrum", size=(500, 300))
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

        # for limiting update_graph
        self.fps_limiter = FPSLimiter(60)
        self.fps_limiter_wave = FPSLimiter(30)
        self.fps_limiter_fft  = FPSLimiter(30)

        ## Add widgets into each dock

        ## Dock 1
        self.waveform_plot = WaveformPlot()
        dock1.addWidget(self.waveform_plot.init_to_widget(self.fps_limiter.notifyRenderFinished))
        #dock1.addWidget(self.waveform_plot.init_to_widget())

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

        # basic parameters holder
        self.ana_param = AnalyzerParameters()
        self.audio_saver_manager = AudioSaverManager()
        # core audio pipeline is managed here
        self.audio_pipeline = AudioPipeline()

        # connect monitor button
        ui_dock4.pushButton_mon.clicked.connect(self.start_stop_monitoring)
        self.b_monitor_on = not ui_dock4.pushButton_mon.isChecked()
        # connect screenshot button
        ui_dock4.pushButton_screenshot.clicked.connect(self.take_screen_shot)
        # connect device selection comboBox
        self.on_combobox_dev_refresh()
        self.ui_dock4.comboBox_dev.activated.connect(self.on_combobox_dev_activated)
        # connect sample rate comboBox
        self.ui_dock4.comboBox_sr.activated.connect(self.on_combobox_sr_activated)
        # connect channel text box (and sanity check)
        self.ui_dock4.lineEdit_ch.textChanged.connect(self.on_lineedit_ch_text_changed)
        # connect fft length comboBox
        self.ui_dock4.comboBox_fftlen.activated.connect(self.on_combobox_fftlen_activated)
        # connect n average comboBox
        self.ui_dock4.comboBox_nave.activated.connect(self.on_combobox_nave_activated)
        # connect recording related buttons and text boxes
        self.audio_saver_manager.connect_button_events(self.ui_dock4)
        # connect plot event
        #   Queued connection is used to put the rendering in main thread,
        #   and hopefully speed up (by non-blocking) the data processing thread.
        self.graph_data_updated.connect(self.update_graph, 
                                        pg.QtCore.Qt.ConnectionType.QueuedConnection)
        # update time every 0.2 second
        self.datetime_update_timer = QtCore.QTimer()
        self.datetime_update_timer.timeout.connect(self.update_current_datetime)
        self.datetime_update_timer.start(200)
        # basic stat timer
        self.report_stat_timer = QtCore.QTimer()
        self.report_stat_timer.timeout.connect(self.print_basic_stat)
        self.report_stat_timer.start(2000)
        # connect the custom closeEvent
        self.closeEvent = self.custom_close_event

        self.show()

        self.fpsLastUpdate = tic()
        self.elapsed_wave = [1e-3]
        self.elapsed_spum = [1e-3]
        self.elapsed_rms = [1e-3]
        self.elapsed_spam = [1e-3]

        default_device_idx = 0
        self.on_combobox_dev_activated(default_device_idx)

    def stop_data_pipeline(self):
        if self.audio_pipeline.is_device_on():
            self.audio_pipeline.close(wait=True)
        logging.info('rec stopped')

    def start_data_pipeline(self, dev_name = None):
        # start new device
        if dev_name is not None:
            self.ana_param.load_device(dev_name)
            self.ana_param.update_to_ui(self.ui_dock4)
        else:
            # assume ana_param is ready (modified in the UI)
            pass
        self.audio_saver_manager.update_wav_param(self.ana_param)
        self.audio_pipeline.init(self.ana_param)

        ## setup plots
        sz_hop = self.ana_param.size_hop
        analyzer_data = self.audio_pipeline.analyzer_data
        self.waveform_plot.init_param(analyzer_data, sz_hop)
        self.waveform_plot.config_plot()
        self.spectrum_plot.init_param(analyzer_data, sz_hop)
        self.spectrum_plot.config_plot()
        self.rms_plot.init_param(analyzer_data, sz_hop,
                                 self.ana_param.spectrogram_duration)
        self.rms_plot.config_plot()
        self.spectrogram_plot.init_param(analyzer_data, sz_hop,
                                         self.ana_param.spectrogram_duration)
        self.spectrogram_plot.config_plot()

        # set FPS limiters
        #t_wave, t_rms, t_spum, t_spam = self.ana_param.render_time_cost_estimation()
        #fps_lim = min(
        #    30,
        #    500.0 / (t_wave + t_rms + t_spum + t_spam)
        #    )
        #logging.info(f'FPS limit: {fps_lim}')
        #self.fps_limiter_wave.dt = 1.0 / fps_lim
        #self.fps_limiter_fft.dt  = 1.0 / fps_lim

        # set to false to start the monitoring
        self.ui_dock4.pushButton_mon.setChecked(False)
        self.start_stop_monitoring(False)

        self.fps_limiter.clear()

        self.audio_pipeline.start(
            self.audio_saver_manager,
            self.rms_plot.feed_rms,
            [self.spectrum_plot.feed_spectrum,
             self.spectrogram_plot.feed_spectrum],
            self.update_request
        )
        logging.info('rec restarted')

    def on_combobox_dev_refresh(self):
        self.ui_dock4.comboBox_dev.clear()
        self.ui_dock4.comboBox_dev.addItems(self.ana_param.devices_conf_default.keys())
        self.ui_dock4.comboBox_dev.addItem('refresh list')
        self.ui_dock4.comboBox_dev.addItem('none')
        self.ui_dock4.comboBox_dev.setCurrentIndex(0)

    def on_combobox_dev_activated(self, index):
        dev_name = self.ui_dock4.comboBox_dev.itemText(index)
        logging.info(f'Device: Item[{index}] = "{dev_name}" was selected')
        if dev_name == 'refresh list':
            return  # TODO
        if dev_name == 'none':
            self.stop_data_pipeline()
            return
        self.stop_data_pipeline()
        PopOldEventsAndExecute(
            lambda: self.start_data_pipeline(dev_name)
        )

    def on_combobox_sr_activated(self, index):
        sr_text = self.ui_dock4.comboBox_sr.itemText(index)
        logging.info(f'Sample rate: Item[{index}] = "{sr_text}" was selected')
        sr = self.ana_param.dic_sample_rate[sr_text]
        self.ana_param.sample_rate = sr
        # full restart. TODO: allow partial restart
        self.stop_data_pipeline()
        PopOldEventsAndExecute(
            self.start_data_pipeline
        )

    def on_lineedit_ch_text_changed(self, text):
        logging.info(f'lineedit_ch_text_changed: {text}')
        ok = self.ana_param.update_channel_by_ui(self.ui_dock4.lineEdit_ch)
        if not ok:
            return
        if not self.audio_pipeline.is_device_on():
            return
        # full restart. TODO: allow partial restart
        self.stop_data_pipeline()
        PopOldEventsAndExecute(
            self.start_data_pipeline
        )

    def on_combobox_fftlen_activated(self, index):
        l = int(self.ui_dock4.comboBox_fftlen.itemText(index))
        logging.info(f'combobox fftlen: {l}')
        self.ana_param.set_fft_len(l)
        # full restart. TODO: allow partial restart
        self.stop_data_pipeline()
        PopOldEventsAndExecute(
            self.start_data_pipeline
        )

    def on_combobox_nave_activated(self, index):
        n = int(self.ui_dock4.comboBox_nave.itemText(index))
        logging.info(f'combobox nave: {n}')
        self.ana_param.n_ave = n
        self.audio_pipeline.set_n_ave(n)
        self.rms_plot.set_n_ave(n)
        self.spectrogram_plot.set_n_ave(n)

    def is_monitoring_on(self):
        # Test if the monitor is on
        if not self.audio_pipeline.is_device_on():
            return False
        return self.b_monitor_on

    def start_stop_monitoring(self, checked):
        """checked == False: running, and the button is used to stop the monitoring"""
        logging.debug(f'start_stop_monitoring: {checked}')
        btn = self.ui_dock4.pushButton_mon
        if checked:
            # stop the monitoring
            self.b_monitor_on = False
            btn.setText('Start monitoring')
            #btn.setStyleSheet("background-color: grey")
        else:
            # start the monitoring
            self.b_monitor_on = True
            btn.setText('Stop monitoring')
            #btn.setStyleSheet("background-color: white")

    def update_request(self, obj):
        """Called from other thread"""
        if not self.b_monitor_on:
            return
        if self.fps_limiter.checkFPSAllow():
            # signal main thread to call update_graph
            self.graph_data_updated.emit(obj)

    # TODO: annotate callbacks using decorator
    def update_graph(self, obj):
        # must be called in main thread
        rms_db, volt, fqs, spectrum_db = obj
        # ploting
        self.waveform_plot.update(volt)

        self.rms_plot.update()
        self.spectrum_plot.update()
        self.spectrogram_plot.update()

        #self.fps_limiter.notifyRenderFinished()

    def update_graph_t_measure(self, obj):
        if not self.b_monitor_on:
            return
        # usually called from main thread
        rms_db, volt, fqs, spectrum_db = obj
        # ploting
        if self.fps_limiter_wave.checkFPSAllow():
            tic()
            self.waveform_plot.update(volt)
            self.qapp.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
            self.elapsed_wave.append(toc())

        if spectrum_db is not None:
            if self.fps_limiter_fft.checkFPSAllow():
                tic()
                self.rms_plot.update()
                self.qapp.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
                self.elapsed_rms.append(toc())
                tic()
                self.spectrum_plot.update(fqs, spectrum_db)
                self.qapp.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
                self.elapsed_spum.append(toc())
                tic()
                self.spectrogram_plot.update()
                self.qapp.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
                self.elapsed_spam.append(toc())

        t_end = tic()
        # update fps at most once every 0.2 secs
        if t_end - self.fpsLastUpdate > 5.0:
            self.fpsLastUpdate = t_end
            fps_wave = 1 / np.mean(self.elapsed_wave)
            fps_rms  = 1 / np.mean(self.elapsed_rms)
            fps_spum = 1 / np.mean(self.elapsed_spum)
            fps_spam = 1 / np.mean(self.elapsed_spam)
            print(f'{1000/fps_wave:0.5f} ms (wave)')
            print(f'{1000/fps_rms:0.5f} ms (rms)')
            print(f'{1000/fps_spum:0.5f} ms (spum)')
            print(f'{1000/fps_spam:0.5f} ms (spam)')
            print('---')
            self.elapsed_wave = []
            self.elapsed_rms = []
            self.elapsed_spum = []
            self.elapsed_spam = []
        
    def update_current_datetime(self):
        now = datetime.datetime.now()
        self.ui_dock4.label_datetime.setText(now.strftime("%Y-%m-%d %H:%M:%S"))

    def print_basic_stat(self):
        return
        print('--- Stat ---')
        if self.audio_pipeline is not None:
            print(f'Audio queue size: {self.audio_pipeline.buf_queue.qsize()}')

    def custom_close_event(self, event):
        # TODO: disable the recorder first
        self.audio_pipeline.close(wait=False)
        self.audio_saver_manager.close()
        self.ana_param.dump_self_conf()
        event.accept()  # Accept the close event

    def take_screen_shot(self):
        app = pg.QtGui.QGuiApplication.instance()
        screen = app.primaryScreen()
        screenshot = screen.grabWindow(self.winId())
        now = datetime.datetime.now()
        png_path = now.strftime("winshot_%Y-%m-%d_%H%M%S.png")
        screenshot.save(png_path, 'png')

_t0 = None

def tic():
    global _t0
    #_t0 = time.process_time_ns() / 1e9
    _t0 = perf_counter()
    return _t0

def toc(s = None):
    global _t0
    #t = time.process_time_ns() / 1e9 - _t0
    t = perf_counter() - _t0
    if s:
        print(f'time = {t:.6f} s, ({s})')
    return t

def main():
    app = pg.mkQApp("Spectrum Analyzer - docked plots")
    main_window = MainWindow(qapp = app)
    pg.exec()

if __name__ == '__main__':
    main()