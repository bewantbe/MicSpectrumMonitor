Usage
=====

```bash
# to run
python record_wave.py -d default -l 8192 -n 128 --calib='99-21328.txt'

# to kill
pkill -f record_wave.py

# to record
arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 48000 -c 1 --duration=20 r1_imm_1.wav

# to list all recording sources
arecord -L
```

Data flow
=========
record_wave.py

Reading from mic
```python
class recThread(threading.Thread):
    def run():
        inp = alsaaudio.PCM(ALSA device parameters...)
        while 1:
            l, data = inp.read()     # blocked IO
            sample_d = decode(data)
            queue.put(sample_d)      # thread-safe queue
```

Feed data to analyzer and show it
```python
class processThread(threading.Thread):
    def run():
        while 1:
            s = queue.get(True, 0.1)   # wait for data, with timeout=0.1sec
            while s has data left:
                s_chunk <- s           # fill chunk with desired chunk size.
                process(s_chunk)

    def process(chunk):
        analyzer_data.put(chunk)       # FFT stuff
        notify_graph_update()          # matplotlib canvas draw stuff
```
