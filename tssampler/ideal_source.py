import time
import numpy as np
from numpy import (
    log10, sqrt, sum
)
import tssabc

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

class SineSource(tssabc.SampleReader):
    sampler_id = 'sine'

    def init(self, sample_rate, chunk_size, freq):
        self.sample_rate = sample_rate
        self.w = 2 * np.pi * freq
        self.t_last = time.time()
        self.phase_last = 0

    def read(self, n_frames):
        fq = self.w / self.sample_rate
        sample_d = np.sin(self.phase_last + fq * np.arange(n_frames))
        self.t_last += n_frames / self.sample_rate
        self.phase_last = (self.phase_last + fq * n_frames) % (2 * np.pi)
        # Simulate the blocking behavior:
        #   When no more data, block until all requested frames are ready.
        t_wait = self.t_last - time.time()
        if t_wait > 0:
            time.sleep(t_wait)
        return sample_d.reshape((1, n_frames))
    
    def close(self):
        pass
    
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

    plt.plot(vals.T)
    plt.show()