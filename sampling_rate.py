from __future__ import print_function
import time
import struct
import alsaaudio
import math
import numpy as np

from recmonitor import shortPeriodDectector, overrunChecker

def log10(x):
    return math.log10(x) if x>0 else float('-inf') if x==0 else float('nan')

class recThread:
    """ Recorder thread """
    def __init__(self, device, n_channels=1, sample_rate=44100, periodsize=256, format=alsaaudio.PCM_FORMAT_S16_LE):
        if (len(device) > 0):
            self.device = device
        else:
            self.device = 'default'
        self.n_channels  = n_channels
        self.sample_rate = sample_rate * 1.0  # keep float, so easier for math
        self.periodsize  = periodsize
        self.format = format
        self.b_run = False
        self.sample_bits = 16 if self.format == alsaaudio.PCM_FORMAT_S16_LE else 24
        self.sample_maxp1 = 2 ** (self.sample_bits-1)
    
    def decode_raw_samples(self, data):
        b = bytearray(data)
        if self.format == alsaaudio.PCM_FORMAT_S16_LE:
            # S16_LE
            #for j in range(len(data)):
                #print("d[%4d] = % 5d,    " % (j, ord(data[j])), end='')
            #print("")
            sample_s = struct.unpack_from('%dh'%(len(data)/2), b)
            #for j in range(len(sample_s)):
                #print("v[%4d] = % 5d,    " % (j, sample_s[j]), end='')
            #print("")
            sample_d = np.array(sample_s) / 32768.0
        else:
            # S24_3LE
            sample_d = np.zeros(len(b)/3)
            for i in range(len(sample_d)):
                v = b[3*i] + 0x100*b[3*i+1] + 0x10000*b[3*i+2]
                sample_d[i] = v - ((v & 0x800000) << 1)
            sample_d /= 0x1000000 * 1.0
        # separate channels
        sample_d = sample_d.reshape((len(sample_d)/self.n_channels, self.n_channels))
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

        overrun_checker = overrunChecker(self.sample_rate, self.periodsize)
        overrun_checker.start()
        while self.b_run:
            l, data = inp.read()     # Read data from device
            if l < 0:
                print("recorder overrun at t = %.3f sec, some samples are lost." % (time.time()))
                continue
            if l * self.n_channels * self.sample_bits/8 != len(data):
                raise IOError('number of channel or sample size do not match')
            overrun_checker.updateState(l)
            if l < self.periodsize:
                print("\nread sample: %d, requested: %d" \
                    % (l, self.periodsize))
            if l == 0:
                continue
            sample_d = self.decode_raw_samples(data)
            #sample_d = sample_d[:,0]
            rms = (np.sum(sample_d ** 2, 0) / sample_d.shape[0])**(1.0/2)
            m1 = np.max(sample_d, 0) * self.sample_maxp1
            m2 = np.min(sample_d, 0) * self.sample_maxp1
            #print("l = %d, datalen = %d" % (l, len(data)), ", shape = ", sample_d.shape)
            print("RMS: %7.2f dB, maxmin = (%5.0f,%5.0f)" % \
                    (20*log10(rms[0]), m1[0], m2[0]))
            if len(rms)>1:
                print("RMS: %7.2f dB, maxmin = (%5.0f,%5.0f)" % \
                        (20*log10(rms[1]), m1[1], m2[1]))

d = 'default'
d = 'hw:CARD=PCH,DEV=0'
d = 'hw:CARD=Device,DEV=0'
rec = recThread(d, 1, 48000, 1024)
rec.run()

