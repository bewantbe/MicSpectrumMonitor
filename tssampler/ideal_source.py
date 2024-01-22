import time
import numpy as np
from numpy import (
    log10, sqrt, sum
)
import tssabc

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
    
class WhiteSource(tssabc.SampleReader):
    sampler_id = 'white'

    def init(self, sample_rate, chunk_size, fn_cb = None, normalize_level = 0):
        self.sample_rate = sample_rate
        self.t_last = time.time()
        self.normalize_level = normalize_level  # can be RMS_db_sine_inc
        return self
    
    def read(self, n_frames):
        s = np.random.rand(1, n_frames) * 2 - 1
        self.t_last += n_frames / self.sample_rate
        # Simulate the blocking behavior:
        #   When no more data, block until all requested frames are ready.
        t_wait = self.t_last - time.time()
        if t_wait > 0:
            time.sleep(t_wait)
        return s.reshape((1, n_frames))
    
    def spectrumLevel(self, wnd):
        return 10*log10(1.0/3*sum(wnd**2)*2/sum(wnd)**2) + self.normalize_level
    
    def RMS(self):
        return 10*log10(1.0/3) + self.normalize_level
    
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