import time
import numpy as np
from numpy import (
    log10, sqrt, sum
)
from . import tssabc

class SineSource(tssabc.SampleReader):
    sampler_id = 'sine'
    device_name = 'Sine wave'
    capability = {
        'sample_format': ['S16_LE'],
        'sample_rate': [48000, 44100, 8000, ...],
        'n_channel': [1],
        'period_size': [2048, 256, 512, 1024, ...],
        'freq': [440.0, ...]
    }

    def init(self, sample_rate, period_size, freq):
        self.sample_rate = sample_rate
        self.period_size = period_size
        self.w = 2 * np.pi * freq
        self.t_last = time.time()
        self.phase_last = 0
        return self

    def read(self, n_frames = None):
        if n_frames is None:
            n_frames = self.period_size
        fq = self.w / self.sample_rate
        sample_d = np.sin(self.phase_last + fq * np.arange(n_frames))
        self.t_last += n_frames / self.sample_rate
        self.phase_last = (self.phase_last + fq * n_frames) % (2 * np.pi)
        # Simulate the blocking behavior:
        #   When no more data, block until all requested frames are ready.
        t_wait = self.t_last - time.time()
        if t_wait > 0:
            time.sleep(t_wait)
        return sample_d.reshape((n_frames, 1))
    
    def close(self):
        pass
    
class WhiteSource(tssabc.SampleReader):
    sampler_id = 'white'
    device_name = 'White noise'
    capability = {
        'sample_format': ['S16_LE'],
        'sample_rate': [48000, 44100, 8000, ...],
        'n_channel': [1],
        'period_size': [2048, 256, 512, 1024, ...],
    }
    def init(self, sample_rate, period_size, fn_cb = None):
        self.sample_rate = sample_rate
        self.period_size = period_size
        self.t_last = time.time()
        return self
    
    def read(self, n_frames = None):
        if n_frames is None:
            n_frames = self.period_size
        s = np.random.rand(1, n_frames) * 2 - 1
        self.t_last += n_frames / self.sample_rate
        # Simulate the blocking behavior:
        #   When no more data, block until all requested frames are ready.
        t_wait = self.t_last - time.time()
        if t_wait > 0:
            time.sleep(t_wait)
        return s.reshape((n_frames, 1))
    
    @staticmethod
    def spectrumLevel(wnd, normalize_level):
        return 10*log10(1.0/3*sum(wnd**2)*2/sum(wnd)**2) + normalize_level
    
    @staticmethod
    def RMS(normalize_level):
        return 10*log10(1.0/3) + normalize_level
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test the source
    sine_source = SineSource()
    sine_source.init(48000, 1024, 100)
    vals = [None] * 3
    for i in range(3):
        vals[i] = sine_source.read(1024)
    sine_source.close()

    vals = np.concatenate(vals, axis=1)
    print(vals.shape)

    plt.figure(1)
    plt.plot(vals.T)
    plt.title('sine source')

    white_source = WhiteSource()
    white_source.init(48000, 1024)
    vals = [None] * 3
    for i in range(3):
        vals[i] = white_source.read(1024)
    white_source.close()

    vals = np.concatenate(vals, axis=1)
    print(vals.shape)
    plt.figure(2)
    plt.plot(vals.T)
    plt.title('white source')

    plt.show()