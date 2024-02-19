import time
import numpy as np
import struct      # unpack the packed bytes
import alsaaudio
from . import tssabc

class AlsaAudio(tssabc.SampleReader):

    sampler_id = 'alsa'

    def init(self, sample_rate, stream_callback=None, n_channel=1, periodsize=256, format='S16_LE', device='default', **kwargs):
        self.n_channel  = n_channel
        self.sample_rate = sample_rate * 1.0  # keep float, so easier for math
        self.periodsize  = periodsize
        self.format = eval('alsaaudio.PCM_FORMAT_' + format)
        self.sample_bits = 16 if self.format == alsaaudio.PCM_FORMAT_S16_LE else 24
        self.sample_maxp1 = 2 ** (self.sample_bits-1)
        # initialize the device
        self.inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, 
           rate = int(self.sample_rate), channels = self.n_channel,
           format = self.format, periodsize = self.periodsize,
           periods = 4, device = device)

    def decode_raw_samples(self, data):
        b = bytearray(data)
        if self.format == alsaaudio.PCM_FORMAT_S16_LE:
            # S16_LE
            sample_s = struct.unpack_from('%dh'%(len(data)/2), b)
            sample_d = np.array(sample_s) / 32768.0
        else:
            # S24_3LE
            sample_d = np.zeros(len(b)//3)
            for i in range(len(sample_d)):
                v = b[3*i] + 0x100*b[3*i+1] + 0x10000*b[3*i+2]
                sample_d[i] = v - ((v & 0x800000) << 1)
            sample_d /= 0x1000000 * 1.0
        # separate channels
        sample_d = sample_d.reshape((len(sample_d)//self.n_channel, self.n_channel))
        return sample_d

    def read(self):
        l, data = self.inp.read()     # Read data from device
        if l < 0:
            print("recorder overrun at t = %.3f sec, some samples are lost." % (time.time()))
            return None
        if l * self.n_channel * self.sample_bits/8 != len(data):
            raise IOError('number of channel or sample size do not match')
        if l < self.periodsize:
            print("\nread sample: %d, requested: %d" \
                  % (l, self.periodsize))
        sample_d = self.decode_raw_samples(data)
        print(sample_d[:10, :])
        return sample_d
