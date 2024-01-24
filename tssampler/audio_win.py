# TO test:
#   python audio_win.py

import pyaudio
import struct      # unpack the packed bytes
import numpy as np
from . import tssabc
#import tssabc     # use as script

class MicReader(tssabc.SampleReader):
    
    sampler_id = 'pyaudio'

    def __init__(self):
        self.initilized = False

    def init(self, sample_rate, periodsize, stream_callback=None, **kwargs):
        # Ref. https://people.csail.mit.edu/hubert/pyaudio/docs/
        self.pya = pyaudio.PyAudio()
        self.sample_rate = sample_rate
        self.chunk_size = periodsize   # TODO: are they the same
        self.dtype = 'int16'
        self.n_channels = 1
        self.stream = self.pya.open(
            rate = self.sample_rate,
            channels = self.n_channels,
            format = pyaudio.paInt16,
            input = True,
            input_device_index = None,
            frames_per_buffer = self.chunk_size,
            stream_callback = stream_callback
        )
        self.stream.start_stream()
        self.initilized = True
        return self

    def read(self, n_frames):
        if n_frames == None:
            n_frames = self.chunk_size
        data = self.stream.read(n_frames)
        vals = struct.unpack('h' * (len(data) // 2), data)
        # TODO: or try numpy.frombuffer
        sample_d = np.array(vals) / 32768.0
        # separate channels
        sample_d = sample_d.reshape((len(sample_d)//self.n_channels, self.n_channels))
        return sample_d

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pya.terminate()
        self.initilized = False

    def __del__(self):
        self.close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # test the class by 1 call to read()
    mic_reader = MicReader()
    mic_reader.init(44100, 1024)
    vals = [None] * 5
    for i in range(len(vals)):
        vals[i] = mic_reader.read(1024)
    mic_reader.close()

    vals = np.concatenate(vals, axis=1)
    print(vals.shape)

    plt.plot(vals.T)
    plt.show()