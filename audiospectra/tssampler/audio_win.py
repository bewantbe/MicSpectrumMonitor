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

    def init(self, device, sample_rate, n_channel, value_format, periodsize, stream_callback=None):
        # value_format: S16_LE, int16
        # device: device index
        # Ref. https://people.csail.mit.edu/hubert/pyaudio/docs/
        self.pya = pyaudio.PyAudio()
        self.sample_rate = sample_rate
        self.chunk_size = periodsize   # TODO: are they the same
        self.dtype = value_format      # TODO: not really used yet
        self.n_channel = n_channel
        if device == 'default':
            device = None
        self.stream = self.pya.open(
            rate = self.sample_rate,
            channels = self.n_channel,
            format = pyaudio.paInt16,
            input = True,
            input_device_index = device,
            frames_per_buffer = self.chunk_size,
            stream_callback = stream_callback
        )
        self.stream.start_stream()
        self.initilized = True
        return self

    def read(self, n_frames = None):
        # return formatted data: (n_frames, n_channel)
        if n_frames == None:
            n_frames = self.chunk_size
        data = self.stream.read(n_frames)
        vals = struct.unpack('h' * (len(data) // 2), data)
        # TODO: or try numpy.frombuffer
        sample_d = np.array(vals) / 32768.0
        # separate channels
        sample_d = sample_d.reshape((len(sample_d)//self.n_channel, self.n_channel))
        return sample_d

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pya.terminate()
        self.initilized = False

    def __del__(self):
        if self.initilized:
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