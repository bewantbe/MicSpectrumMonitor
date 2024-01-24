#!/usr/bin/env python3

import sys
import time
import struct
import re
import queue
import threading

import numpy as np
from numpy import log10
from numpy import sqrt
from numpy import sum

# https://larsimmisch.github.io/pyalsaaudio/
from tssampler import get_sampler
from tssampler.ideal_source import WhiteSource

# to run:  `python record_wave.py -d default -l 8192 -n 128 --calib='99-21328.txt'`
# to kill: `pkill -f record_wave.py`
# to record: `arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 48000 -c 1 --duration=20 r1_imm_1.wav`

# Use `arecord -L` to list all recording sources
# To set volume
"""
pacmd set-source-port 1 analog-input-headset-mic
pacmd set-source-mute 1 false
pacmd set-source-volume 1 6554   # "base volume" see `pacmd list-sources`
pacmd set-source-output-volume `pacmd list-source-outputs | grep index | cut -c 12-` 13000 && pacmd list-source-outputs | grep volume
pacmd set-source-volume 1 6554 && pacmd list-sources | grep volume
"""

# xde: UMIK-1 vol=65536 (0 dB) <-dBA-> PMIK-1 vol=16630 (-35.74 dB)
# xde: UMIK-1 vol=65536 (0 dB) <-dBA-> iMM-6 id=8, vol=16800 + 6.2 dB
# xde: UMIK-1 vol=65536 (0 dB) <-dBA-> iMM-6 id=8, vol=21000 (-29.66 dB)

# xde: UMIK-1 vol=26090 (-24.00 dB) <-dBA-> PMIK-1 vol=6700 (-59.42 dB)
# xde: UMIK-1 vol=26090 (-24.00 dB) <-dBA-> iMM-6 id=8, vol=8500 (-53.22 dB)

# xde: UMIK-1 vol=52000 (-6.03 dB) <-dBA-> huawei: PMIK-1 (rec)

from recmonitor import shortPeriodDectector, overrunChecker

class recThread(threading.Thread):
    """ Recorder thread """
    def __init__(self, name, buf_que, conf):
        threading.Thread.__init__(self)
        self.name = name
        self.buf_que = buf_que    # "output" port of recording data
        self.conf = conf
        self.periodsize = conf['periodsize']
        self.sampler = get_sampler(conf.pop('sampler_id'))
        self.b_run = False
    
    def run(self):
        # PCM Objects
        # http://larsimmisch.github.io/pyalsaaudio/libalsaaudio.html#pcm-objects
        self.sampler.init(**self.conf)
        self.b_run = True

        overrun_checker = overrunChecker(
            self.conf['sample_rate'], self.conf['periodsize'])
        overrun_checker.start()
        while self.b_run:
            sample_d = self.sampler.read(self.conf['periodsize'])
            #overrun_checker.updateState(l)
            if not self.buf_que.full():
                self.buf_que.put(sample_d, True)
            else:
                print('recThread: Buffer overrun.')
        print('Thread ', self.name, ' exited.')

# Analyze data
class analyzerData():
    """ Data analyzer """
    def __init__(self, sz_chunk, sample_rate, ave_num = 1, RMS_normalize_to_sine = False):
        self.sample_rate = sample_rate
        self.sz_chunk = sz_chunk     # data size for one FFT
        self.sz_fft   = sz_chunk     # FFT frequency points
        self.update_RMS_normalize_to_sine(RMS_normalize_to_sine)
        self.rms = 0
        self.v = np.zeros(sz_chunk)
        # hold spectrums, no negative frequency
        self.sp_cumulate = np.zeros((self.sz_fft + 2) // 2)
        self.sp_vo = np.zeros((self.sz_fft + 2) // 2)
        self.sp_db = np.ones((self.sz_fft + 2) // 2) * float('-inf')
        self.sp_cnt = 0
        self.ave_num = ave_num       # number of averages to get one spectrum
        self.lock_data = threading.Lock()
        # window function
        self.wnd = 0.5 + 0.5 * np.cos((np.arange(1, sz_chunk+1) / (sz_chunk+1.0) - 0.5) * 2 * np.pi)
#        self.wnd = np.ones(sz_chunk)
        self.wnd *= len(self.wnd) / sum(self.wnd)
        self.wnd_factor = self.RMS_sine_factor * 2.0 / sum(self.wnd) ** 2
        # factor for dBA
        sqr = lambda x: x*x
        fqs = 1.0 * np.arange(len(self.sp_vo)) / self.sz_fft * self.sample_rate
        self.fqs = fqs
        r = sqr(12194.0)*sqr(sqr(fqs)) / ((fqs*fqs+sqr(20.6)) * sqrt((fqs*fqs+sqr(107.7)) * (fqs*fqs+sqr(737.9))) * (fqs*fqs+sqr(12194.0)))
        self.dBAFactor = r * r * 10 ** (1/5.0)
        self.use_dBA = False
        self.loadCalib('')

    def update_RMS_normalize_to_sine(self, z_sine):
        # Note: RMS(sine) + 10*log10(2) = RMS(square)
        self.RMS_normalize_to_sine = z_sine
        if z_sine:
            self.RMS_sine_factor = 2.0
            self.RMS_db_sine_inc = 10*log10(2.0)
        else:
            self.RMS_sine_factor = 1.0
            self.RMS_db_sine_inc = 0.0

    def loadCalib(self, fname):
        if len(fname) == 0:  # clear the calibration
            self.calib_db = []
            self.calib_pow = []
            self.calib_centre_freq = []
            self.calib_centre_db = []
            return
        
        calib_orig = []
        with open(fname, "r") as fin:
            for l in fin:
                l = l.strip(' \t\n')
                if len(l) == 0: continue
                if l[0] == '*':
                    print('loadCalib: Dayton Audio iMM-6 header')
                    self.calib_centre_freq, self.calib_centre_db =\
                        map(float, l[1:].lower().split('hz'))
                    print('loadCalib: %.2f dB at %.1f Hz' % (self.calib_centre_db, self.calib_centre_freq))
                elif l[0] == '#':
                    # comment line
                    pass
                elif l[0:12] == '"Sens Factor':
                    print('loadCalib: miniDSP UMIK header')
                    # "Sens Factor =-.6383dB, SERNO: 7023270"
                    m = re.search('([+-]?[.0-9]+)', l)
                    if m is not None:
                        self.calib_centre_db = float(m.group(1))
                        self.calib_centre_freq = []
                        m2 = re.search('SERNO:[ \t]*([0-9]+)', l[m.end():])
                        if m2 is not None:
                            print('loadCalib:', m2.group())
                elif l[0:13] == '"miniDSP PMIK':
                    print('loadCalib: miniDSP PMIK header')
                    # "miniDSP PMIK-1 calibration file, serial: 8000348, format: txt"
                    m2 = re.search('serial:[ \t]*([0-9]+)', l)
                    if m2 is not None:
                        print('loadCalib:', m2.group())
                elif '0'<=l[0] and l[0]<='9' or l[0]=='-' or l[0]=='+':
                    fq_db = map(float, l.split())
                    calib_orig.append(fq_db)
                else:
                    print('loadCalib: ignored: %s' % (l))
        
        if len(calib_orig) == 0 or (np.array(map(len, calib_orig))-2).any():
            # abnormal calibration file
            print('File "%s" format not recognized.' % (fname))
            return
        calib_orig = np.array(calib_orig).T
        self.calib_db = np.interp(self.fqs, calib_orig[0], calib_orig[1])
        self.calib_pow = 10 ** (self.calib_db / 10)
        print('Using calibration file "%s": %d entries' % (fname, len(calib_orig[0])))

    def put(self, data):
        if len(self.v) != len(data): return
        # volt trace
        self.lock_data.acquire()
        self.v[:] = 1.0 * data[:]    # save a copy, minize lock time
        self.lock_data.release()
        
        # spectrum
        tmp_amp = np.fft.rfft(self.v * self.wnd, self.sz_fft)
        tmp_pow = (tmp_amp * tmp_amp.conj()).real * self.wnd_factor
        if len(self.calib_db) > 0:
            tmp_pow /= self.calib_pow
        if self.sz_fft % 2 == 0:
            tmp_pow = np.concatenate([[tmp_pow[0]/2], tmp_pow[1:-1], [tmp_pow[-1]/2]])
        else:
            tmp_pow = np.concatenate([[tmp_pow[0]/2], tmp_pow[1:]])
        self.sp_cumulate += tmp_pow
        self.sp_cnt += 1
        if self.sp_cnt >= self.ave_num:
            self.sp_cumulate /= self.sp_cnt
            self.sp_cnt = 0
            self.lock_data.acquire()
            self.sp_vo[:] = self.sp_cumulate[:]
            if self.use_dBA:
                self.sp_vo *= self.dBAFactor
            np.seterr(divide='ignore')       # for God's sake
            self.sp_db = 10 * log10(self.sp_vo)  #  in dB
            self.lock_data.release()
            self.sp_cumulate[:] = 0
        
        self.rms = sqrt(self.RMS_sine_factor * sum(self.v ** 2) / len(self.v))

    def get_volt(self):
        self.lock_data.acquire()  # TODO: rewrite using context management protocol (with lock:)
        tmpv = self.v.copy()
        self.lock_data.release()
        return tmpv

    def get_spectrum_dB(self):
        self.lock_data.acquire()
        tmps = self.sp_db.copy()
        self.lock_data.release()
        return tmps

    def get_RMS_dB(self):
        np.seterr(divide='ignore')       # for God's sake
        return 20*log10(self.rms)  # already count RMS_db_sine_inc

    def get_FFT_RMS_dBA(self):
        self.lock_data.acquire()
        fft_rms = sqrt(2 * sum(self.sp_vo) / self.wnd_factor / self.sz_fft / sum(self.wnd ** 2))
        self.lock_data.release()
        np.seterr(divide='ignore')       # for God's sake
        return 20*log10(fft_rms) + self.RMS_db_sine_inc

class FPSLimiter:
    """ fps limiter """
    def __init__(self, fps):
        self.dt = 1.0 / fps
        self.time_to_update = time.time()
    
    def checkFPSAllow(self):
        time_now = time.time()
        if (self.time_to_update > time_now):
            return False
        self.time_to_update += self.dt
        if time_now > self.time_to_update :
            self.time_to_update = time_now + self.dt
        return True

fps_lim1 = FPSLimiter(3)

# py plot
# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

import matplotlib
import matplotlib.pyplot as plt

# http://matplotlib.org/users/customizing.html
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 11}
matplotlib.rc('font', **font)

# https://matplotlib.org/users/event_handling.html
# ResizeEvent

# ploter for audio data
class plotAudio:
    def __init__(self, analyzer_data, condition_variable):
        self.b_run = False
        self.analyzer_data = analyzer_data
        self.cv = condition_variable
        self.fig, self.ax = plt.subplots(2, 1)

        self.drawBackground()
        self.saveBackground()
        
        self.event_cids = [ \
            self.fig.canvas.mpl_connect('resize_event', self.onResize), \
            self.fig.canvas.mpl_connect('close_event', self.onClose), \
            self.fig.canvas.mpl_connect('key_press_event', self.onPressed)]

    def __del__(self):
        if hasattr(self, 'event_cids'):
            for cid in self.event_cids:
                self.fig.canvas.mpl_disconnect(cid)
        
    def onResize(self, event):
        self.saveBackground()
    
    def onClose(self, event):
        self.b_run = False
        with self.cv:
            self.cv.notify()
    
    def onPressed(self, event):
        if not event: return
        global b_start
        global size_chunk
        global n_ave
        print('you pressed', event.key, event.xdata, event.ydata)
        k = event.key
        if k == 'h':  # help
            print('a: toggle dBA or dB\nf2: toggle RMS sine or square normalize\nf3: Load/unload calibration\nq: exit')
        elif k == 'a':
            global use_dBA
            use_dBA = not use_dBA
            self.analyzer_data.use_dBA = use_dBA
            print('Toggled dBA or dB')
        elif k == 'f2':
            self.analyzer_data.update_RMS_normalize_to_sine(
                not self.analyzer_data.RMS_normalize_to_sine)
            print('RMS sine or square')
        elif k == 'f3':
            ad = self.analyzer_data
            if len(ad.calib_db) == 0:
                print('Load calib: %s' % (calib_path))
                ad.loadCalib(calib_path)
            else:
                print('Unload calib.')
                ad.loadCalib('')
            self.drawBackground()
            self.saveBackground()
        elif k == '-':
            b_start = True
            self.onClose(None)
            if size_chunk > 256: size_chunk = int(size_chunk/2)
        elif k == '=' or k == '+':
            b_start = True
            self.onClose(None)
            size_chunk *= 2
        elif k == '[':
            b_start = True
            self.onClose(None)
            if n_ave > 1: n_ave = int(n_ave/2)
        elif k == ']':
            b_start = True
            self.onClose(None)
            n_ave *= 2
        elif k == 'q':
            self.onClose(None)
    
    def drawBackground(self):
        analyzer_data = self.analyzer_data
        self.ax[0].clear()
        self.ax[1].clear()
        # init volt draw
        self.plt_line, = self.ax[0].plot([], [], 'b')
        self.ax[0].set_xlim(0, size_chunk / analyzer_data.sample_rate)
        self.ax[0].set_ylim(-1.3, 1.3)
        self.ax[0].set_xlabel('t')
        self.text_1 = self.ax[0].text(0.0, 0.91, '', transform=self.ax[0].transAxes)
        self.text_1.set_text('')
        self.ax[0].yaxis.set_animated(True)
        self.plt_line.set_animated(True)
        self.text_1.set_animated(True)

        # init spectum draw
        self.spectrum_line, = self.ax[1].plot([], [], 'b')
        self.ax[1].set_xlim(1, analyzer_data.sample_rate / 2)
        self.ax[1].set_ylim(-140, 1)
#        self.ax[1].set_ylim(-43, -33)  # for debug
        self.ax[1].set_xscale('log')
        self.ax[1].set_xlabel('freq(Hz)')
        self.text_2 = self.ax[1].text(0.0, 0.91, '', transform=self.ax[1].transAxes)
        self.spectrum_line.set_animated(True)
        self.text_2.set_animated(True)
        # calib line
        calib = analyzer_data.calib_db if len(analyzer_data.calib_db) \
            else np.zeros(len(analyzer_data.fqs))
        fqs = analyzer_data.fqs
        sp = -calib + WhiteSource.spectrumLevel(analyzer_data.wnd, self.analyzer_data.RMS_db_sine_inc)
        self.ax[1].plot(fqs, sp, 'r')
    
    def saveBackground(self):
        # For blit
        # http://stackoverflow.com/questions/8955869/why-is-plotting-with-matplotlib-so-slow
        # http://scipy-cookbook.readthedocs.io/items/Matplotlib_Animations.html
        self.fig.show()
        self.fig.canvas.draw()
        self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.ax]
        
    def plotVolt(self):
        analyzer_data = self.analyzer_data
        # volt
        y = analyzer_data.get_volt()
        if y.any():
            self.ax[0].set_ylim(np.array([-1.3, 1.3])*y.max())
            #self.fig.canvas.draw()
        # Animating xaxis range is still a problem:
        # https://github.com/matplotlib/matplotlib/issues/2324
        x = np.arange(0, len(y), dtype='float') / analyzer_data.sample_rate
        self.plt_line.set_data(x, y)        
        # RMS
        rms = analyzer_data.get_RMS_dB()
        self.text_1.set_text("RMS = %5.2f dB" % (rms))

    def plotSpectrum(self):
        analyzer_data = self.analyzer_data
        # spectrum
        y = analyzer_data.get_spectrum_dB()
        x = np.arange(0, len(y), dtype='float') / analyzer_data.sz_chunk * analyzer_data.sample_rate
        self.spectrum_line.set_data(x, y)
        fft_rms = analyzer_data.get_FFT_RMS_dBA()
        self.text_2.set_text("RMS = %5.2f %s %s" % (fft_rms, self.str_dBA, self.str_normalize))

    def graph_update(self):
        if not fps_lim1.checkFPSAllow() :
            return

        analyzer_data = self.analyzer_data
        self.str_normalize = '(sine=0dB)' if self.analyzer_data.RMS_normalize_to_sine else '(square=0dB)'
        self.str_dBA = 'dBA' if use_dBA else 'dB'
        print("\rRMS: % 5.2f dB, % 5.2f %s %s   " % (analyzer_data.get_RMS_dB(), analyzer_data.get_FFT_RMS_dBA(), self.str_dBA, self.str_normalize), end='')
        sys.stdout.flush()
        
        self.fig.canvas.restore_region(self.backgrounds[0])
        self.plotVolt()
        self.ax[0].draw_artist(self.plt_line)
        self.ax[0].draw_artist(self.text_1)
        #self.ax[0].draw_artist(self.ax[0].get_yaxis())  # slow
        self.ax[0].draw_artist(self.ax[0].yaxis)  # slow
        self.fig.canvas.blit(self.ax[0].bbox)
        
        self.fig.canvas.restore_region(self.backgrounds[1])
        self.plotSpectrum()
        self.ax[1].draw_artist(self.spectrum_line)
        self.ax[1].draw_artist(self.text_2)
        self.fig.canvas.blit(self.ax[1].bbox)
        
        self.fig.canvas.flush_events()
        
    def show(self):
#        use this like `plt.show()`
        self.b_run = True
        while self.b_run:
            with self.cv:
                self.cv.wait()
            self.graph_update()    # need lock or not?
        plt.close(self.fig)

def process_func(analyzer_data, condition_variable, chunk):
    """ process data """
    analyzer_data.put(chunk)
    # notify UI thread (for plot) that new data comes
    with condition_variable:
        condition_variable.notify()

class sampleChunkThread(threading.Thread):
    """ data dispatch thread """
    def __init__(self, name, func_process, buf_que, channel_select, sz_chunk, sz_hop=0):
        """
        Param:
            func_process: is the function to process data
            buf_que: sampling data comes from this queue
            channel_select: Channel is selected as sample_data[channel_select, :]
            sz_chunk: is size of one feed to the process function
            sz_hop: is the hop size between two feed,
                    when sz_hop < sz_chunk, overlap happens;
                    when sz_hop > sz_chunk, some data is ommited.
                    sz_hop == 0 means sz_hop = sz_chunk.
        """
        threading.Thread.__init__(self)
        self.name = name
        self.func_process = func_process
        self.buf_que = buf_que
        self.channel_select = channel_select
        self.sz_chunk = sz_chunk
        self.sz_hop = sz_hop if sz_hop > 0 else sz_chunk
        self.b_run = False

    def run(self):
        """Continuousely poll data from the queue"""
        self.b_run = True
        sz_chunk = self.sz_chunk
        chunk_feed = None
        chunk_pos = 0             # position in chunk
        # collect sampling data, call process() when ever get sz_chunk data
        while self.b_run:
            try:
                s = self.buf_que.get(True, 0.1)
            except queue.Empty:
                s = []
            if (len(s) == 0):
                continue
            s = s[self.channel_select, :]
            if chunk_feed is None:
                if len(s.shape) == 2:
                    chunk_feed = np.zeros(s.shape[0], sz_chunk)
                else: # vector
                    chunk_feed = np.zeros(sz_chunk)
            s_pos = 0
            # `s` cross boundary
            while sz_chunk - chunk_pos <= len(s) - s_pos:
                chunk_feed[chunk_pos:] = s[s_pos : s_pos+sz_chunk-chunk_pos]
                s_pos += sz_chunk-chunk_pos
                self.func_process(chunk_feed)
                chunk_pos = sz_chunk - self.sz_hop
                chunk_feed[0:chunk_pos] = chunk_feed[self.sz_hop:]

            chunk_feed[chunk_pos : chunk_pos+len(s)-s_pos] = s[s_pos:]   # s is fit into chunk
            chunk_pos += len(s)-s_pos
        print('Thread ', self.name, ' exited.')

###########################################################################
# main

if __name__ == '__main__':

    # parse input parameters
    import getopt
    param_fmt_short = 'd:n:l:'
    param_fmt_long = ['device=', 'n-ave=', 'fftlen=', 'calib=', 'dBA', 'db-sine']
    options, options_other = getopt.getopt(sys.argv[1:], param_fmt_short, param_fmt_long)

    # default values
    pcm_device = 'default'
    size_chunk = 16384
    n_ave = 1
    calib_path = ''
    use_dBA = False
    RMS_normalize_to_sine = False

    for opt, arg in options:
        if opt in ('-d', '--device'):
            pcm_device = arg
        elif opt in ('-l', '--fftlen'):
            size_chunk = int(arg)
        elif opt in ('-n', '--n-ave'):
            n_ave = int(arg)
        elif opt == '--calib':
            calib_path = arg
        elif opt == '--dBA':
            use_dBA = True
        elif opt == '--db-sine':
            RMS_normalize_to_sine = True

    # buffer that transmit data from recorder to processor
    buf_queue = queue.Queue(10000)

    b_start = True
    while b_start:
        b_start = False
        print("FFT len:", size_chunk)
        print("  n_ave:", n_ave)
        # prepare recorder
        print("using device: ", pcm_device)
        if pcm_device == 'default':
            conf = {
                'sampler_id': 'mic',
                'device'    : 'default',
                'n_channels': 1,
                'sample_rate': 48000,
                'periodsize': 1024,
                'format'    : 'S16_LE',
            }
        elif pcm_device == 'hw:CARD=U18dB,DEV=0':
            conf = {
                'sampler_id': 'mic',
                'device'    : 'hw:CARD=U18dB,DEV=0',
                'n_channels': 2,
                'sample_rate': 48000,
                'periodsize': 1024,
                'format'    : 'S24_LE',
            }
        elif pcm_device == 'ad7606c':
            conf = {
                'sampler_id': 'ad7606c',
                'sample_rate': 48000,
                'periodsize': 4800,
            }
        else:
            conf = {
                'sampler_id': 'mic',
                'device'    : 'default',
                'n_channels': 1,
                'sample_rate': 48000,
                'periodsize': 1024,
                'format'    : 'S16_LE',
            }
        rec_thread = recThread('rec', buf_queue, conf)

        # init analyzer data
        analyzer_data = analyzerData(size_chunk, rec_thread.conf['sample_rate'],
                                     n_ave, RMS_normalize_to_sine)
        analyzer_data.loadCalib(calib_path)
        analyzer_data.use_dBA = use_dBA

        # lock for UI thread
        condition_variable = threading.Condition()

        # init ploter
        plot_audio = plotAudio(analyzer_data, condition_variable)

        func_proc = lambda data_chunk: process_func(analyzer_data, condition_variable, data_chunk)

        # init data dispatcher
        process_thread = sampleChunkThread('dispatch', func_proc, buf_queue, 0,
                                        size_chunk, size_chunk//2)
        process_thread.start()

        rec_thread.start()

        plot_audio.show()

        rec_thread.b_run = False
        process_thread.b_run = False

    print('\nExiting...')

# vim: set expandtab shiftwidth=4 softtabstop=4:
