
import time
import numpy as np

#from . import tssabc
import tssabc     # use as script

import sys
sys.path.append('C:\\Users\\xyy82\\soft\\USB2AD7606BC\\temp-release\\customapp')
from M3F20xm import M3F20xmADC, dbg_print

class AD7607CReader(tssabc.SampleReader):
    sampler_id = 'ad7607c'

    def __init__(self):
        self.initilized = False

    def init(self, sample_rate, periodsize=None, stream_callback=None, **kwargs):
        self.chunk_size = periodsize           # TODO: are they the same
        self.adc = M3F20xmADC(reset = True)
        self.adc.set_sampling_rate(sample_rate, 1.0)
        self.n_channels = self.adc.n_channels
        self.sample_rate = 1.0 / self.adc.get_sampling_interval()
        self.adc.start()
        self.initilized = True
        return self

    def read(self, n_frames):
        n_frames_left = self.adc.get_fifo_frames_left()
        t_wait = (2 * n_frames - n_frames_left) / self.sample_rate
        if t_wait > 0:
            # we better wait for more data
            time.sleep(t_wait)
        v = self.adc.read(n_frames)   # non-block
        a = np.array(v).reshape((len(v)//self.n_channels, self.n_channels)).T
        return (a - 0) / 32768.0

    def close(self):
        if self.initilized:
            self.adc.stop()
            self.initilized = False

    def __del__(self):
        self.close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # test the class
    mic_reader = AD7607CReader()
    mic_reader.init(44100)
    vals = [None] * 5
    for i in range(len(vals)):
        vals[i] = mic_reader.read(1024)
    mic_reader.close()

    vals = np.concatenate(vals, axis=1)
    print(vals.shape)

    plt.plot(vals.T)
    plt.show()