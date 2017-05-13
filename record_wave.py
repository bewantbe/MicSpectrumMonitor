#!/usr/bin/env python

from __future__ import print_function

import sys
import time
import struct
import Queue
import threading

import numpy as np
from numpy import log10
from numpy import sqrt
from numpy import sum

# https://larsimmisch.github.io/pyalsaaudio/
import alsaaudio

# to run:  `python record_wave.py default 8192 16`
# to kill: `pkill -f record_wave.py`
# to record: `arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 48000 -c 1 --duration=20 r1_imm_1.wav`

# Use `arecord -L` to list all recording sources
# To set volume
"""
pacmd set-source-port 1 analog-input-headset-mic
pacmd set-source-mute 1 false
pacmd set-source-volume 1 6554   # "base volume" see `pacmd list-sources`
pacmd set-source-output-volume `pacmd list-source-outputs | grep index | cut -c 12-` 13000 && pacmd list-source-outputs | grep volume
pacmd set-source-volume 1 6554 && pacmd list-sources | grep volume
"""

# xde: UMIK-1 vol=65536 (0 dB) <-dBA-> PMIK-1 vol=16800 (-35.47 dB)
# xde: UMIK-1 vol=65536 (0 dB) <-dBA-> iMM-6 id=8, vol=16800 + 6.2 dB
# xde: UMIK-1 vol=65536 (0 dB) <-dBA-> iMM-6 id=8, vol=21000 (-29.66 dB)

# xde: UMIK-1 vol=26090 (-24.00 dB) <-dBA-> PMIK-1 vol=6700 (-59.42 dB)
# xde: UMIK-1 vol=26090 (-24.00 dB) <-dBA-> iMM-6 id=8, vol=8500 (-53.22 dB)

# xde: UMIK-2 vol=52000 (-6.03 dB) <-dBA-> huawei: PMIK-1 (rec)

class cosSignal:
    """ cos signal generator """
    def __init__(self, freq, sample_rate):
        self.t0 = 0
        self.freq = freq
        self.sample_rate = sample_rate

    def get(self, size):
        fq = 1.0 * self.freq / self.sample_rate
        sample_d = np.cos(2*np.pi * fq * (self.t0 + np.arange(size)))
        self.t0 += size
        return sample_d.reshape((1, size))

# Note: RMS(sine) + 10*log10(2) = RMS(square)

RMS_normalize_to_sine = False
if RMS_normalize_to_sine:
    RMS_db_sine_inc = 10*log10(2)
else:
    RMS_db_sine_inc = 0

class whiteSignal:
    """ white noise generator """
    def get(self, size):
        return np.random.rand(1, size)*2-1
    
    def spectrumLevel(self, wnd):
        return 10*log10(1.0/3*sum(wnd**2)*2/sum(wnd)**2) + RMS_db_sine_inc
    
    def RMS(self):
        return 10*log10(1.0/3) + RMS_db_sine_inc

cos_signal = cosSignal(1000, 48000)
white_signal = whiteSignal()

class recThread(threading.Thread):
    """ Recorder thread """
    def __init__(self, name, buf_que, device, n_channels=1, sample_rate=44100, periodsize=256, format=alsaaudio.PCM_FORMAT_S16_LE):
        threading.Thread.__init__(self)
        self.name = name
        self.buf_que = buf_que    # "output" port of recording data
        if (len(device) > 0):
            self.device = device
        else:
            self.device = 'default'
        self.n_channels  = n_channels
        self.sample_rate = sample_rate * 1.0  # keep float, so easier for math
        self.periodsize  = periodsize
        self.format = format
        self.b_run = False
    
    def decode_raw_samples(self, data):
        b = bytearray(data)
        if self.format == alsaaudio.PCM_FORMAT_S16_LE:
            # S16_LE
            sample_s = struct.unpack_from('%dh'%(len(data)/2/self.n_channels), b)
            sample_d = np.array(sample_s) / 32768.0
        else:
            # S24_3LE
            sample_d = np.zeros(len(b)/3)
            for i in range(len(sample_d)):
                v = b[3*i] + 0x100*b[3*i+1] + 0x10000*b[3*i+2]
                sample_d[i] = v - ((v & 0x800000) << 1)
            sample_d /= 0x1000000 * 1.0
        # separate channels
        sample_d = sample_d.reshape((len(sample_d)/self.n_channels, self.n_channels)).T
        return sample_d

    def run(self):
        # PCM Objects
        # http://larsimmisch.github.io/pyalsaaudio/libalsaaudio.html#pcm-objects
        inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device=self.device)
        inp.setchannels(self.n_channels)
        inp.setrate(int(self.sample_rate))
        inp.setformat(self.format)
        inp.setperiodsize(self.periodsize)   # frames per period
        
        self.b_run = True
        while self.b_run:
            l, data = inp.read()     # Read data from device
            if l == 0:
                continue
            if l < 0:
                print("recorder overrun at t = %.3f sec, some samples are lost." % (time.time()))
                continue
            sample_d = self.decode_raw_samples(data)
#            sample_d = cos_signal.get(self.periodsize)
            sample_d = white_signal.get(self.periodsize)

            if not self.buf_que.full():
                self.buf_que.put(sample_d, True)
            else:
                print('recThread: Buffer overrun.')
        print('Thread ', self.name, ' exited.')

# Analyze data
class analyzerData():
    """ Data analyzer """
    def __init__(self, sz_chunk, rec_th, ave_num = 1):
        self.sample_rate = rec_th.sample_rate
        self.sz_chunk = sz_chunk     # data size for one FFT
        self.sz_fft   = sz_chunk     # FFT frequency points
        self.rms = 0
        self.v = np.zeros(sz_chunk)
        # hold spectrums, no negative frequency
        self.sp_cumulate = np.zeros((self.sz_fft + 2) / 2)
        self.sp_vo = np.zeros((self.sz_fft + 2) / 2)
        self.sp_db = np.ones((self.sz_fft + 2) / 2) * float('-inf')
        self.sp_cnt = 0
        self.ave_num = ave_num       # number of averages to get one spectrum
        self.lock_data = threading.Lock()
        # window function
        self.wnd = 0.5 + 0.5 * np.cos((np.arange(1, sz_chunk+1) / (sz_chunk+1.0) - 0.5) * 2 * np.pi)
#        self.wnd = np.ones(sz_chunk)
        self.wnd *= len(self.wnd) / sum(self.wnd)
        self.wnd_factor = 2.0 / sum(self.wnd) ** 2   # 1*sin(t) = 0 dBFS
        # factor for dBA
        sqr = lambda x: x*x
        fqs = 1.0 * np.arange(len(self.sp_vo)) / self.sz_fft * self.sample_rate
        self.fqs = fqs
        r = sqr(12194.0)*sqr(sqr(fqs)) / ((fqs*fqs+sqr(20.6)) * sqrt((fqs*fqs+sqr(107.7)) * (fqs*fqs+sqr(737.9))) * (fqs*fqs+sqr(12194.0)))
        self.dBAFactor = r * r * 10 ** (1/5.0)
        self.use_dBA = False
        self.loadCalib('')

    def loadCalib(self, fname):
        if len(fname) == 0:  # clear the calibration
            self.calib_db = []
            self.calib_pow = []
            self.calib_centre_freq = []
            self.calib_centre_db = []
            return
        
        calib_file_type = ''
        calib_orig = []
        with open(fname, "r") as fin:
            for l in fin:
                l = l.strip(' \t\n')
                if len(l) == 0: continue
                if l[0] == '*':  # imm header
                    calib_file_type = 'imm'
                    self.calib_centre_freq, self.calib_centre_db =\
                        map(float, l[1:].lower().split('hz'))
                    print('calib : %.2f dB at %.1f Hz' % (self.calib_centre_db, self.calib_centre_freq))
                elif l[0] == '#':
                    # comment line
                    pass
                elif '0'<=l[0] and l[0]<'9' or l[0]=='-' or l[0]=='+':
                    fq_db = map(float, l.split())
                    calib_orig.append(fq_db) 
        
        if len(calib_orig) == 0 or (np.array(map(len, calib_orig))-2).any():
            # abnormal calibration file
            print('File "%s" format not recognized.' % (fname))
            return
        calib_orig = np.array(calib_orig).T
	self.calib_db = np.interp(self.fqs, calib_orig[0], calib_orig[1])
	self.calib_pow = 10 ** (self.calib_db / 10)
	print('Using calibration file "%s": %d entries' % (fname, len(calib_orig[0])))

    def put(self, data):
        # volt trace
        self.lock_data.acquire()
        self.v[:] = 1.0 * data[:]    # save a copy, minize lock time
        self.lock_data.release()
        
        # spectrum
        tmp_amp = np.fft.rfft(self.v * self.wnd, self.sz_fft)
        tmp_pow = (tmp_amp * tmp_amp.conj()).real * self.wnd_factor
        if len(self.calib_db) > 0:
            tmp_pow /= self.calib_pow
        if self.sz_fft % 2 == 0:
            tmp_pow = np.concatenate([[tmp_pow[0]/2], tmp_pow[1:-1], [tmp_pow[-1]/2]])
        else:
            tmp_pow = np.concatenate([[tmp_pow[0]/2], tmp_pow[1:]])
        self.sp_cumulate += tmp_pow
        self.sp_cnt += 1
        if self.sp_cnt >= self.ave_num:
            self.sp_cumulate /= self.sp_cnt
            self.sp_cnt = 0
            self.lock_data.acquire()
            self.sp_vo[:] = self.sp_cumulate[:]
            if self.use_dBA:
                self.sp_vo *= self.dBAFactor
            np.seterr(divide='ignore')       # for God's sake
            self.sp_db = 10 * log10(self.sp_vo)  #  in dB
            self.lock_data.release()
            self.sp_cumulate[:] = 0
        
        self.rms = sqrt(sum(self.v ** 2) / len(self.v))

    def getV(self):
        self.lock_data.acquire()
        tmpv = self.v.copy()
        self.lock_data.release()
        return tmpv

    def getSpectrumDB(self):
        self.lock_data.acquire()
        tmps = self.sp_db.copy()
        self.lock_data.release()
        return tmps

    def getRMS_dB(self):
        np.seterr(divide='ignore')       # for God's sake
        return 20*log10(self.rms) + RMS_db_sine_inc

    def getFFTRMS_dBA(self):
        self.lock_data.acquire()
        fft_rms = sqrt(2 * sum(self.sp_vo) / self.wnd_factor / self.sz_fft / sum(self.wnd ** 2))
        self.lock_data.release()
        np.seterr(divide='ignore')       # for God's sake
        return 20*log10(fft_rms) + RMS_db_sine_inc

class FPSLimiter:
    """ fps limiter """
    def __init__(self, fps):
        self.dt = 1.0 / fps
        self.time_to_update = time.time()
    
    def checkFPSAllow(self):
        time_now = time.time()
        if (self.time_to_update > time_now):
            return False
        self.time_to_update += self.dt
        if time_now > self.time_to_update :
            self.time_to_update = time_now + self.dt
        return True

fps_lim1 = FPSLimiter(3)

# py plot
# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

import matplotlib
import matplotlib.pyplot as plt

# http://matplotlib.org/users/customizing.html
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 11}
matplotlib.rc('font', **font)

# https://matplotlib.org/users/event_handling.html
# ResizeEvent

# ploter for audio data
class plotAudio:
    def __init__(self, analyzer_data, condition_variable):
        self.b_run = False
        self.analyzer_data = analyzer_data
        self.cv = condition_variable
        self.fig, self.ax = plt.subplots(2, 1)

        # init volt draw
        self.plt_line, = self.ax[0].plot([], [], 'b')
        self.ax[0].set_xlim(0, size_chunk / analyzer_data.sample_rate)
        self.ax[0].set_ylim(-1.3, 1.3)
        self.text_1 = self.ax[0].text(0.0, 0.91, '', transform=self.ax[0].transAxes)
        self.text_1.set_text('')
        self.plt_line.set_animated(True)
        self.text_1.set_animated(True)

        # init spectum draw
        self.spectrum_line, = self.ax[1].plot([], [], 'b')
        self.ax[1].set_xlim(1, analyzer_data.sample_rate / 2)
#        self.ax[1].set_ylim(-140, 1)
        self.ax[1].set_ylim(-43, -33)
        self.ax[1].set_xscale('log')
        self.text_2 = self.ax[1].text(0.0, 0.91, '', transform=self.ax[1].transAxes)
        self.spectrum_line.set_animated(True)
        self.text_2.set_animated(True)
        # calib line
        if len(analyzer_data.calib_db) > 0:
            fqs = analyzer_data.fqs
            sp = -analyzer_data.calib_db + white_signal.spectrumLevel(analyzer_data.wnd)
            self.ax[1].plot(fqs, sp, 'r')
        
        self.saveBackground()
        
        self.event_cids = [ \
            self.fig.canvas.mpl_connect('resize_event', self.onResize), \
            self.fig.canvas.mpl_connect('close_event', self.onClose)]

    def __del__(self):
        for cid in self.event_cids:
            self.fig.canvas.mpl_disconnect(cid)
        
    def onResize(self, event):
        self.saveBackground()
    
    def onClose(self, event):
        self.b_run = False
        with self.cv:
            self.cv.notify()
    
    def saveBackground(self):
        # For blit
        # http://stackoverflow.com/questions/8955869/why-is-plotting-with-matplotlib-so-slow
        # http://scipy-cookbook.readthedocs.io/items/Matplotlib_Animations.html
        self.fig.show()
        self.fig.canvas.draw()
        self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.ax]
        
    def plotVolt(self):
        analyzer_data = self.analyzer_data
        # volt
        y = analyzer_data.getV()
        if y.any():
            self.ax[0].set_ylim(np.array([-1.3, 1.3])*y.max())
        # Animating xaxis range is still a problem:
        # https://github.com/matplotlib/matplotlib/issues/2324
        x = np.arange(0, len(y), dtype='float') / analyzer_data.sample_rate
        self.plt_line.set_data(x, y)        
        # RMS
        rms = analyzer_data.getRMS_dB()
        self.text_1.set_text("%.3f, RMS = %5.2f dB" % (time.time(), rms))

    def plotSpectrum(self):
        analyzer_data = self.analyzer_data
        # spectrum
        y = analyzer_data.getSpectrumDB()
        x = np.arange(0, len(y), dtype='float') / analyzer_data.sz_chunk * analyzer_data.sample_rate
        self.spectrum_line.set_data(x, y)
        fft_rms = analyzer_data.getFFTRMS_dBA()
        self.text_2.set_text("dBA RMS = %5.2f dB" % (fft_rms))

    def graph_update(self):
        if not fps_lim1.checkFPSAllow() :
            return

        analyzer_data = self.analyzer_data
        print("\rRMS: % 5.2f dB, % 5.2f dBA    " % (analyzer_data.getRMS_dB(), analyzer_data.getFFTRMS_dBA()), end='')
        sys.stdout.flush()
        
        self.fig.canvas.restore_region(self.backgrounds[0])
        self.plotVolt()
        self.ax[0].draw_artist(self.plt_line)
        self.ax[0].draw_artist(self.text_1)
#        self.ax[0].draw_artist(self.ax[0].get_yaxis())  # slow
        self.fig.canvas.blit(self.ax[0].bbox)
        
        self.fig.canvas.restore_region(self.backgrounds[1])
        self.plotSpectrum()
        self.ax[1].draw_artist(self.spectrum_line)
        self.ax[1].draw_artist(self.text_2)
        self.fig.canvas.blit(self.ax[1].bbox)
        
        self.fig.canvas.flush_events()
        
    def show(self):
#        use this like `plt.show()`
        self.b_run = True
        while self.b_run:
            with self.cv:
                self.cv.wait()
            self.graph_update()    # need lock or not?

class processThread(threading.Thread):
    """ data dispatch thread """
    def __init__(self, name, buf_que, condition_variable, sz_chunk, sz_hop=0):
        threading.Thread.__init__(self)
        self.name = name
        self.buf_que = buf_que
        self.b_run = False
        self.sz_chunk = sz_chunk
        self.sz_hop = sz_hop if sz_hop > 0 else sz_chunk
        self.cv = condition_variable

    def process(self, chunk):
        analyzer_data.put(chunk)
        # notify UI thread (for plot) that new data comes
        with self.cv:
            self.cv.notify()

    def run(self):
        self.b_run = True
        sz_chunk = self.sz_chunk
        s_chunk = np.zeros(sz_chunk)
        chunk_pos = 0             # position in chunk
        # collect sampling data, call process() when ever get sz_chunk data
        while self.b_run:
            try:
                s = self.buf_que.get(True, 0.1)
            except Queue.Empty:
                s = []
            if (s == []):
                continue
            s = s[0, :]                    # select left channel
            s_pos = 0
            # `s` cross boundary
            while sz_chunk - chunk_pos <= len(s) - s_pos:
                s_chunk[chunk_pos:] = s[s_pos : s_pos+sz_chunk-chunk_pos]
                s_pos += sz_chunk-chunk_pos
                self.process(s_chunk)
                chunk_pos = sz_chunk - self.sz_hop
                s_chunk[0:chunk_pos] = s_chunk[self.sz_hop:]

            s_chunk[chunk_pos : chunk_pos+len(s)-s_pos] = s[s_pos:]   # s is fit into chunk
            chunk_pos += len(s)-s_pos
        print('Thread ', self.name, ' exited.')

###########################################################################
# main

# parse input parameters
pcm_device = 'default'
if len(sys.argv) > 1:
    pcm_device = sys.argv[1]

size_chunk = 16384
if len(sys.argv) > 2:
    size_chunk = int(sys.argv[2])

n_ave = 1
if len(sys.argv) > 3:
    n_ave = int(sys.argv[3])

# buffer that transmit data from recorder to processor
buf_queue = Queue.Queue(10000)

# prepare recorder
print("using device: ", pcm_device)
if pcm_device == 'default':
    rec_thread = recThread('rec', buf_queue, 'default', \
        1, 48000, 1024, alsaaudio.PCM_FORMAT_S16_LE)
elif pcm_device == 'hw:CARD=U18dB,DEV=0':
    rec_thread = recThread('rec', buf_queue, 'hw:CARD=U18dB,DEV=0', \
        2, 48000, 1024, alsaaudio.PCM_FORMAT_S24_LE)
else:
    rec_thread = recThread('rec', buf_queue, device, \
        1, 48000, 1024, alsaaudio.PCM_FORMAT_S16_LE)

# init analyzer data
analyzer_data = analyzerData(size_chunk, rec_thread, n_ave)
analyzer_data.loadCalib('99-21328.txt')
analyzer_data.use_dBA = True

# lock for UI thread
condition_variable = threading.Condition()

# init ploter
plot_audio = plotAudio(analyzer_data, condition_variable)

# init data dispatcher
process_thread = processThread('dispatch', buf_queue, condition_variable, size_chunk, size_chunk/2)
process_thread.start()

rec_thread.start()

plot_audio.show()

rec_thread.b_run = False
process_thread.b_run = False

print('\nExiting...')

# vim: set expandtab shiftwidth=4 softtabstop=4:
