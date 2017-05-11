#!/usample_rate/bin/env python

from __future__ import print_function

import sys
import time
import struct
import Queue
import threading

import alsaaudio

import numpy as np

# Threading
# https://docs.python.org/2/library/threading.html
# https://www.tutorialspoint.com/python/python_multithreading.htm

# use `arecord --list-devices` to list recording devices
# use `arecord -L` to list all recording sources
# use `pacmd list-sources` to list available recording sources in system using pulseaudio

t0 = 0

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

            # test signal
            global t0
            freq = 20.0/48000;
            sample_d = np.cos(2*np.pi*freq * (t0 + np.arange(self.periodsize)))
            t0 += self.periodsize
            sample_d = sample_d.reshape((1, len(sample_d)))

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
        self.sp_db = np.zeros((self.sz_fft + 2) / 2)
        self.sp_cnt = 0
        self.ave_num = ave_num       # number of averages to get one spectrum
        self.lock_data = threading.Lock()
        # window function
        self.wnd = 0.5 + 0.5 * np.cos((np.arange(1, sz_chunk+1) / (sz_chunk+1.0) - 0.5) * 2 * np.pi)
        self.wnd *= len(self.wnd) / np.sum(self.wnd)
        self.wnd_factor = 4.0 / np.sum(self.wnd) ** 2   # 1*sin(t) = 0 dBFS
        # factor for dBA
        self.dBAFactor = np.zeros(len(self.sp_vo))  # apply to power spectrum
        # TODO: use np.arange() to replace for loop
        sqr = lambda x: x*x
        for i in range(len(self.dBAFactor)):
            f = float(i)/self.sz_fft * self.sample_rate;
            r = sqr(12200.0)*sqr(sqr(f)) / ((f*f+sqr(20.6)) * np.sqrt((f*f+sqr(107.7)) * (f*f+sqr(737.9))) * (f*f+sqr(12200.0)))
            self.dBAFactor[i] = r * r * 10 ** (1/5.0)

    def put(self, data):
        # volt trace
        self.lock_data.acquire()
        self.v[:] = 1.0 * data[:]    # save a copy, minize lock time
        self.lock_data.release()
        
        # spectrum
        tmp_amp = np.fft.rfft(self.v * self.wnd, self.sz_fft)
        tmp_pow = (tmp_amp * tmp_amp.conj()).real * self.wnd_factor
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
            np.seterr(divide='ignore')       # for God's sake
            self.sp_db = 10 * np.log10(self.sp_vo)  #  in dB
            self.lock_data.release()
            self.sp_cumulate[:] = 0
        
        self.rms = np.sqrt(np.sum(self.v ** 2) / len(self.v))

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
        return 20*np.log10(self.rms) + 10*np.log10(2)

    def getFFTRMS_dBA(self, dBAFactor = []):
        if dBAFactor == []:
            dBAFactor = self.dBAFactor
        fftlen = self.sz_fft
        self.lock_data.acquire()
        fft_rms = np.sqrt(2 * np.sum(self.sp_vo * dBAFactor) / self.wnd_factor / fftlen / np.sum(self.wnd ** 2))
        self.lock_data.release()
        np.seterr(divide='ignore')       # for God's sake
        return 20*np.log10(fft_rms) + 10*np.log10(2)

# py plot
# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

import matplotlib.pyplot as plt

# ploter for audio data
class plotAudio:
    def __init__(self, analyzer_data):
        self.fig, self.ax = plt.subplots(2, 1)

        # init volt draw
        self.plt_line, = self.ax[0].plot([], [], 'b')
        self.ax[0].set_xlim(0, size_chunk / analyzer_data.sample_rate)
        self.ax[0].set_ylim(-1.3, 1.3)
        self.text_1 = self.ax[0].text(0.0, 0.94, '', transform=self.ax[0].transAxes)
        self.text_1.set_text('01')

        # init spectum draw
        self.spectrum_line, = self.ax[1].plot([], [], 'b')
        self.ax[1].set_xlim(1, analyzer_data.sample_rate / 2)
#        self.ax[1].set_ylim(-120, 1)
        self.ax[1].set_ylim(-5, 5)
        self.ax[1].set_xscale('log')

    def graph_update(self, analyzer_data):
        # volt
        y = analyzer_data.getV()
        x = np.arange(0, len(y), dtype='float') / analyzer_data.sample_rate
        self.plt_line.set_data(x, y)
        
        # RMS
        rms = analyzer_data.getRMS_dB()
        fft_rms = analyzer_data.getFFTRMS_dBA()
        self.text_1.set_text("%.3f, rms = %5.2f dB, dBA rms = %5.2f dB" % (time.time(), rms, fft_rms))
        self.plt_line.figure.canvas.draw_idle()

        # spectrum
        y = analyzer_data.getSpectrumDB()
        x = np.arange(0, len(y), dtype='float') / analyzer_data.sz_chunk * analyzer_data.sample_rate
        self.spectrum_line.set_data(x, y)
        self.spectrum_line.figure.canvas.draw_idle()
        
    def show(self):
        plt.show()

# Analyzer thread
class processThread(threading.Thread):
    def __init__(self, name, buf_que, ploter, sz_chunk, sz_hop=0):
        threading.Thread.__init__(self)
        self.name = name
        self.buf_que = buf_que
        self.b_run = False
        self.sz_chunk = sz_chunk
        self.sz_hop = sz_hop if sz_hop > 0 else sz_chunk
        self.ploter = ploter

    def process(self, chunk):
#        print('Thread ', self.name, ' process() - 1')
        analyzer_data.put(chunk)
#        print('Thread ', self.name, ' process() - 2')
        self.ploter.graph_update(analyzer_data)
#        print('Thread ', self.name, ' process() - 3')

    def run(self):
        self.b_run = True
        sz_chunk = self.sz_chunk
        s_chunk = np.zeros(sz_chunk)
        chunk_pos = 0             # position in chunk
        # collect sampling data, call process() when ever get sz_chunk data
        while self.b_run:
#            print('Thread ', self.name, ' 1')
            try:
                s = self.buf_que.get(True, 0.1)
            except Queue.Empty:
                s = []
#            print('Thread ', self.name, ' 2')
            if (s == []):
                continue
            s = s[0, :]                    # select left channel
            s_pos = 0
            # `s` cross boundary
#            print('Thread ', self.name, ' 3')
            while sz_chunk - chunk_pos <= len(s) - s_pos:
                s_chunk[chunk_pos:] = s[s_pos : s_pos+sz_chunk-chunk_pos]
                s_pos += sz_chunk-chunk_pos
                self.process(s_chunk)
                chunk_pos = sz_chunk - self.sz_hop
                s_chunk[0:chunk_pos] = s_chunk[self.sz_hop:]

#            print('Thread ', self.name, ' 4')
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

n_ave = 1;
if len(sys.argv) > 3:
    n_ave = int(sys.argv[3])


# buffer that transmit data from recorder to processor
buf_queue = Queue.Queue(10000)

# prepare recorder
print("using device: ", pcm_device)
if pcm_device == 'default':
    rec_thread = recThread('rec', buf_queue, 'default', 1, 48000, 1024, alsaaudio.PCM_FORMAT_S16_LE)
elif pcm_device == 'hw:CARD=U18dB,DEV=0':
    rec_thread = recThread('rec', buf_queue, 'hw:CARD=U18dB,DEV=0', 2, 48000, 1024, alsaaudio.PCM_FORMAT_S24_LE)
else:
    rec_thread = recThread('rec', buf_queue, device, 1, 48000, 1024, alsaaudio.PCM_FORMAT_S16_LE)

# init analyzer data
analyzer_data = analyzerData(size_chunk, rec_thread, n_ave)

# init ploter
plot_audio = plotAudio(analyzer_data)

# init data dispatcher
process_thread = processThread('dispatch', buf_queue, plot_audio, size_chunk, size_chunk/2)
process_thread.start()

rec_thread.start()

plot_audio.show()

rec_thread.b_run = False
process_thread.b_run = False

print('Haha')

# vim: set expandtab shiftwidth=4 softtabstop=4:
