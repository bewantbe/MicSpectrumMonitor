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
