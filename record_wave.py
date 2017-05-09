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

# arecord --list-devices
# use `arecord -L` to list recording devices

# Recorder thread
class recThread(threading.Thread):
    def __init__(self, name, buf_que, device, n_channels=1, sample_rate=44100, periodsize=160, format=alsaaudio.PCM_FORMAT_S16_LE):
        threading.Thread.__init__(self)
        self.name = name
        self.buf_que = buf_que
        if (len(device) > 0):
            self.device = device
        else:
            self.device = 'default'
        self.n_channels  = n_channels
        self.sample_rate = sample_rate * 1.0  # keep float, so easier for math
        self.periodsize  = periodsize
        self.format = format
        self.bytes_per_value = format == 2 if alsaaudio.PCM_FORMAT_S16_LE else 3
        self.b_run = False

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
            sample_d = sample_d.reshape((self.n_channels, len(sample_d)/self.n_channels))
            if not self.buf_que.full():
                self.buf_que.put(sample_d, True)
            else:
                print('recThread: Buffer overrun.')

# hold analyzed data
class analyzerData():
    def __init__(self, sz_chunk, rec_th):
        self.sz_chunk = sz_chunk
        self.rms = 0
        self.v = np.zeros(sz_chunk)
        self.lock_data = threading.Lock()
        self.sample_rate = rec_th.sample_rate

    def put(self, dat):
        self.lock_data.acquire()
        self.v[:] = 1.0 * dat[:]
        self.lock_data.release()
        self.rms = 10 * np.log10(np.sum(self.v**2) / len(self.v) * 2) if len(self.v) > 0 else float('-inf')

    def getV(self):
        self.lock_data.acquire()
        tmpv = self.v.copy()
        self.lock_data.release()
        return tmpv

    def getRMS(self):
        return self.rms

# py plot
# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

import matplotlib.pyplot as plt

# ploter for audio data
class plotAudio:
    def __init__(self, analyzer_data):
        self.fig, self.ax = plt.subplots()
        self.plt_line, = plt.plot([], [], 'b', animated=True)
        #self.plt_line, = plt.plot([], [], 'b')
        self.text_1 = self.ax.text(0.0, 0.94, '', transform=self.ax.transAxes)

        self.ax.set_xlim(0, size_chunk / analyzer_data.sample_rate)
        self.ax.set_ylim(-1.1, 1.1)
        self.text_1.set_text('01')
        x = [0, 0.1]
        y = [-1, 1]
        self.plt_line.set_data(x, y)
        plt.draw()

    def graph_update(self, analyzer_data):
        y = analyzer_data.getV()
        x = np.arange(0, len(y), dtype='float') / analyzer_data.sample_rate
        self.plt_line.set_data(x, y)
        rms = analyzer_data.getRMS()
        self.text_1.set_text("%.3f, rms = %5.1f dB" % (time.time(), rms))
        
#        self.plt_line.figure.canvas.draw_idle()
#        self.fig.canvas.draw_idle()
#        plt.draw()
        
#        print('rms = %.3f' % (rms))
#        print(' x = % .5f % .5f ' % (x[0], x[-1]))
#        print(x.shape)
#        print(' y = % .5f % .5f ' % (y[0], y[-1]))
#        print(y.shape)

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
        analyzer_data.put(chunk)
        self.ploter.graph_update(analyzer_data)

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

###########################################################################
# main

# buffer that transmit data from recorder to processor
buf_queue = Queue.Queue(10000)

# prepare recorder
pcm_device = 'default'
if len(sys.argv) > 1:
    pcm_device = sys.argv[1]
print("using device: ", pcm_device)
if pcm_device == 'default':
    rec_thread = recThread('rec', buf_queue, 'default', 1, 48000, 1024, alsaaudio.PCM_FORMAT_S16_LE)
elif pcm_device == 'hw:CARD=U18dB,DEV=0':
    rec_thread = recThread('rec', buf_queue, 'hw:CARD=U18dB,DEV=0', 2, 48000, 1024, alsaaudio.PCM_FORMAT_S24_LE)
else:
    rec_thread = recThread('rec', buf_queue, device, 1, 48000, 1024, alsaaudio.PCM_FORMAT_S16_LE)

# init analyzer data
size_chunk = 16384
analyzer_data = analyzerData(size_chunk, rec_thread)

# init ploter
plot_audio = plotAudio(analyzer_data)

# init data dispatcher
process_thread = processThread('dispatch', buf_queue, plot_audio, size_chunk)
process_thread.start()

rec_thread.start()

plot_audio.show()

rec_thread.b_run = False
process_thread.b_run = False

print('Haha')

# vim: set expandtab shiftwidth=4 softtabstop=4:
