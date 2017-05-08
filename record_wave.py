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

# global variable that pass sample data from recorder to plot
sample_d = np.array([0])

# Threading
# https://docs.python.org/2/library/threading.html
# https://www.tutorialspoint.com/python/python_multithreading.htm

# arecord --list-devices
# use `arecord -L` to list recording devices

lock_sample_d = threading.Lock()

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
        self.b_run = False;

    def run(self):
        # PCM Objects
        # http://larsimmisch.github.io/pyalsaaudio/libalsaaudio.html#pcm-objects
        inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device=self.device)
        inp.setchannels(self.n_channels)
        inp.setrate(self.sample_rate)
        inp.setformat(self.format)
        inp.setperiodsize(self.periodsize)   # frames per period
        
        global sample_d

        self.b_run = True;
        while self.b_run:
            l, data = inp.read()     # Read data from device
            if l == 0:
                continue
            if l < 0:
                print("recorder overrun at t = %.3f sec, some samples are lost." % (time.time()))
                continue
            b = bytearray(data)
            lock_sample_d.acquire()
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
            lock_sample_d.release()


###########################################################################
# main
#rec_thread = recThread('rec', 'default', 1, 48000, 512, alsaaudio.PCM_FORMAT_S16_LE)
#rec_thread = recThread('rec', 'default', 1, 48000, int(48000*0.05), alsaaudio.PCM_FORMAT_S16_LE)
rec_thread = recThread('rec', 'hw:CARD=U18dB,DEV=0', 2, 48000, 1024, alsaaudio.PCM_FORMAT_S24_LE)

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
    ax.set_xlim(0, rec_thread.periodsize)
    ax.set_ylim(-1.1, 1.1)
    text_1.set_text('01')
    return plt_line,text_1

def graph_update(frame):
    if not rec_thread.isAlive():
        return 0,
    global sample_d
    lock_sample_d.acquire()
    if rec_thread.n_channels == 1:
        y = sample_d.copy()
    else:
        y = sample_d[::2].copy()
    lock_sample_d.release()
#    y = np.random.rand(1000)
    l = len(y)
#    print(y.shape)
    x = np.arange(0, l, dtype='float')
    plt_line.set_data(x, y)
    rms = 10 * np.log10(np.sum(y**2) / len(y) * 2)
    text_1.set_text("%.3f, rms = %5.1f dB" % (time.time(), rms))
    return plt_line,text_1

ani = FuncAnimation(fig, graph_update, frames=300, interval=30,
                    init_func=graph_init, blit=True)
plt.show()

rec_thread.b_run = False

print('Haha')

# vim: set expandtab shiftwidth=4 softtabstop=4:
