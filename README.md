PyAudioSpectra - A Mic Spectrum Monitor
=======================================

This is a real-time spectrum and spectrogram monitor (some call it an analyzer) written in pure Python. Conceptually, it's a "Desktop" version of the app [Audio Spectrum Analyzer for Android](https://github.com/bewantbe/audio-analyzer-for-android) or [Audio Spectrum Analyzer](https://play.google.com/store/apps/details?id=github.bewantbe.audio_analyzer_for_android).

The GUI is implemented using [pyqtgraph](https://pyqtgraph.readthedocs.io).

Features:

* It displays waveform, spectrum, spectrogram and RMS-time plots in the same window.
* Easy to adjust plot widget size, and easy to zoom in or out thanks to `pyqtgraph`.
* Tested up to 8 channels at 500kHz sample rate, which is suitable for typical acoustic and ultrasonic applications.
* Capable of saving audio data to a file, taking window screenshots.

See [roadmap](./roadmap) for the current development status and future plans.

A simple "time series sampler" API was used to abstract signal sources. Implementing this API for specific devices requires only a thin layer on top of the device library. It is possible to support devices such as a data acquisition (DAQ) board using this approach. An AD7606c-based ADC device is supported in this manner. The "AD7606c" interface library is available [here](https://github.com/bewantbe/PyAD7606C).


Install
-------

Install the packages in `requirements.txt`. e.g.

```
cd DIRECTORY_TO_THE_SOURCE_CODE
pip install -r requirements.txt
```

Note that in Linux, you may want to install ALSA version of the audio lib.

```
pip install -r requirements_linux.txt
```

And sometimes Qt6 may need extra lib to work:

```
sudo apt-get install libxcb-cursor0
```


Usage
-----

After installation, just run

```
python recorder_gui.py
```

This python app is not yet organized in a typical app file layout.


GUI interface
-------------

It should be to easy use...


Legacy GUI record_wave.py
-------------------------

The 'record_wave.py' script was used to aid in the development of the [Audio Spectrum Analyzer for Android](https://github.com/bewantbe/audio-analyzer-for-android), such as verifying computations, prototyping functions, and calibrating using a professional microphone. It utilizes [Matplotlib](https://matplotlib.org/) as its GUI toolkit. The actual Short-time Fourier transform (STFT) computation and microphone reading thread are also implemented here.

```bash
# to run the GUI using FFT length 8192, average over 128 windows, and using mic calibration file '99-21328.txt'
python record_wave.py -d default -l 8192 -n 128 --calib='99-21328.txt'

# to kill, in case `q` key does not work or has bug.
pkill -f record_wave.py

# to record using ALSA toolkit
arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 48000 -c 1 --duration=20 r1_imm_1.wav

# to list all recording devices
arecord -L
```

Data flow
---------

In `record_wave.py`.

Microphone data is read in this way:

```python
class recThread(threading.Thread):
    def run():
        inp = alsaaudio.PCM(ALSA device parameters...)
        while is_recording:
            data_size, data = inp.read()     # blocked IO
            sample_d = decode(data)          # bytes-like object to float
            qu.put(sample_d)  # thread-safe queue, accessable in both thread
```

Then feed the data to analyzer and notify the update of plot:

```python
class processThread(threading.Thread):
    def run():
        while running:
            s = qu.get(True, 0.1)      # wait for data, with timeout to escape
            while (s is non empty):
                (fill `s_chunk` by `s` with desired chunk size)
                process(s_chunk)

    def process(chunk):
        analyzer_data.put(chunk)       # STFT
        update_condition.notify()
```

In main thread, matplotlib will get notified and update the plot:

```python
qu = queue.Queue()
update_condition = threading.Condition()

rec = recThread()
proc = processThread()

rec.start()
proc.start()

while running:
    with update_condition:
        update_condition.wait()
    graph_update()                     # matplotlib canvas draw
```
