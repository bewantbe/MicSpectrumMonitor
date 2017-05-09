#!/usample_rate/bin/env python

from __future__ import print_function

import sys
import time
import getopt
import alsaaudio
import struct

import Queue
import threading

import numpy as np

#######################################################################
# Recorder thread

# Threading
# https://docs.python.org/2/library/threading.html
# https://www.tutorialspoint.com/python/python_multithreading.htm

# arecord --list-devices
# use `arecord -L` to list recording devices

buf_queue = Queue.Queue(10000)

class recThread(threading.Thread):
    def __init__(self, name, device, n_channels=1, sample_rate=44100, periodsize=160, format=alsaaudio.PCM_FORMAT_S16_LE):
        threading.Thread.__init__(self)
        self.name = name
        if (len(device) > 0):
            self.device = device
        else:
            self.device = 'default'
        self.n_channels  = n_channels
        self.sample_rate = sample_rate
        self.periodsize  = periodsize
        self.format = format
        self.bytes_per_value = format == 2 if alsaaudio.PCM_FORMAT_S16_LE else 3
        self.b_run = False

    def run(self):
        # PCM Objects
        # http://larsimmisch.github.io/pyalsaaudio/libalsaaudio.html#pcm-objects
        inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device=self.device)
        inp.setchannels(self.n_channels)
        inp.setrate(self.sample_rate)
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
                sample_s = struct.unpack_from('%dh'%(len(data)/2/self.n_channels), b)
                sample_d = np.array(sample_s) / 32768.0
            else:
                # S24_3LE
                sample_d = np.zeros(len(b)/3)
                for i in range(len(sample_d)):
                    v = b[3*i] + 0x100*b[3*i+1] + 0x10000*b[3*i+2]
                    sample_d[i] = v - ((v & 0x800000) << 1)
                sample_d = sample_d / (0x1000000 * 1.0)
            # separate channels
            sample_d = sample_d.reshape((self.n_channels, len(sample_d)/self.n_channels))
            if not buf_queue.full():
                buf_queue.put(sample_d, True)

# hold analyzed data
class analyzerData():
    def __init__(self, sz_chunk):
        self.sz_chunk = sz_chunk
        self.rms = 0
        self.v = np.zeros(sz_chunk)
        self.lock_data = threading.Lock()

    def put(self, dat):
        self.lock_data.acquire()
        self.v[:] = 1.0 * dat[:]
        self.lock_data.release()
        self.rms = 10 * np.log10(np.sum(self.v**2) / len(self.v) * 2)

    def getV(self):
        self.lock_data.acquire()
        tmpv = self.v.copy()
        self.lock_data.release()
        return tmpv

    def getRMS(self):
        return self.rms

size_chunk = 16384
analyzer_data = analyzerData(size_chunk)

# Analyzer thread
class processThread(threading.Thread):
    def __init__(self, name, buf_que, sz_chunk, sz_hop=0):
        threading.Thread.__init__(self)
        self.name = name
        self.buf_que = buf_que
        self.b_run = False
        self.sz_chunk = sz_chunk
        self.sz_hop = sz_hop if sz_hop > 0 else sz_chunk

    def process(self, chunk):
        analyzer_data.put(chunk)

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

process_thread = processThread('dispatch', buf_queue, size_chunk)
process_thread.start()

pcm_device = 'default'
if len(sys.argv) > 1:
    pcm_device = sys.argv[1]
print("using device: ", pcm_device)
if pcm_device == 'default':
    rec_thread = recThread('rec', 'default', 1, 48000, 1024, alsaaudio.PCM_FORMAT_S16_LE)
elif pcm_device == 'hw:CARD=U18dB,DEV=0':
    rec_thread = recThread('rec', 'hw:CARD=U18dB,DEV=0', 2, 48000, 1024, alsaaudio.PCM_FORMAT_S24_LE)
else:
    rec_thread = recThread('rec', device, 1, 48000, 1024, alsaaudio.PCM_FORMAT_S16_LE)

rec_thread.start()

# py plot
# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

# animation
# https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
# http://matplotlib.org/api/animation_api.html

# array operation
# https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html
# https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
plt_line, = plt.plot([], [], 'b', animated=True)
text_1 = ax.text(0.0, 0.94, '', transform=ax.transAxes)

def graph_init():
    ax.set_xlim(0, size_chunk / (1.0*rec_thread.sample_rate))
    ax.set_ylim(-1.1, 1.1)
    text_1.set_text('01')
    return plt_line,text_1

def graph_update(frame):
    if not rec_thread.isAlive():
        return 0,
    y = analyzer_data.getV()
#    y = np.random.rand(1000)
    l = len(y)
#    print(y.shape)
    x = np.arange(0, l, dtype='float') / rec_thread.sample_rate
    plt_line.set_data(x, y)
    rms = analyzer_data.getRMS()
    text_1.set_text("%.3f, rms = %5.1f dB" % (time.time(), rms))
    return plt_line,text_1

ani = FuncAnimation(fig, graph_update, frames=300, interval=30,
                    init_func=graph_init, blit=True)
plt.show()

rec_thread.b_run = False
process_thread.b_run = False

print('Haha')

# vim: set expandtab shiftwidth=4 softtabstop=4:
