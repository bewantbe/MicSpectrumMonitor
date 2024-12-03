# SPDX-License-Identifier: GPL-3.0-or-later

import time
import numpy as np

from . import tssabc
#import tssabc     # use as script

import os
import sys
# get path of current python script
_cwd_ = os.path.dirname(os.path.abspath(__file__))
_pyd_ = os.path.join(_cwd_, '..', '..', '..', 'PyAD7606C')
sys.path.append(_pyd_)
from M3F20xm import M3F20xmADC, dbg_print

class AD7606CReader(tssabc.SampleReader):
    sampler_id = 'ad7606c'
    device_name = 'AD7606C'

    capability = {
        'sample_format': ['U16_LE'],   # maybe 'S16_LE', depends on volt_range
        'sample_rate': list(1e6/np.array([2, 4, 8, 20])),
        'n_channel': [8],
        'period_size': [4096, 256, 512, 1024, 2048, 8192, 16384, 32768, ...],
        'volt_range': [(0, 5), (0,10), (0,12.5)]
    }

    def __init__(self):
        self.initilized = False

    def init(self, sample_rate, period_size,
             stream_callback=None, volt_range=[-2.5, 2.5], ends='single', **kwargs):
        self.period_size = period_size           # TODO: are they the same
        self.adc = M3F20xmADC(reset = True)
        self.initilized = True
        self.adc.set_input_range(volt_range, ends)
        self.value_mid = 0 if self.adc.typecode == 'h' else 32768
        self.adc.set_sampling_rate(sample_rate)
        self.n_channel = self.adc.n_channel
        self.sample_rate = 1.0 / self.adc.get_sampling_interval()
        self.adc.show_essential_info()
        self.adc.start(self.period_size)
        return self
    
    def read(self):
        """
        Always return a 2D array, with shape (period_size, n_channel),
        and the values are always within [-1, 1).
        """
        n_frames_left = self.adc.get_fifo_frames_left()
        t_wait = (2 * self.period_size - n_frames_left) / self.sample_rate
        # TODO: sometimes, the resolution of time.sleep is not enough,
        #       we better set lower bound for t_wait
        if t_wait > 0:
            # we better wait for more data
            time.sleep(t_wait)
        v = self.adc.read()   # non-block
        a = np.array(v, dtype=np.float32).reshape((-1, self.n_channel))
        return (a - self.value_mid) / 32768.0

    def close(self):
        self.adc.close()
        self.initilized = False

    def __del__(self):
        if self.initilized:
            self.close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # test the class
    mic_reader = AD7606CReader()
    mic_reader.init(48000, 8000, volt_range=[-2.5, 2.5], ends='single')
    mic_reader.adc.show_config()
    mic_reader.adc.show_reg()
    vals = [None] * (6*2)
    for i in range(len(vals)):
        vals[i] = mic_reader.read()
    mic_reader.close()

    vals = np.concatenate(vals, axis=1)
    print(vals.shape)

    plt.plot(vals.T)
    plt.show()
