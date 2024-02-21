import time
import queue

import numpy as np

from record_wave import (
    recThread,
    sampleChunkThread,
)

def log10_(x):
    # log of 0 is -inf, don't raise error or warning
    # no more np.seterr(divide='ignore')       # for God's sake
    with np.errstate(divide='ignore'):
        return np.log10(x)

_t0 = time.time()

def fn_analysis(volt_chunk):
    # Show ASCII dB level
    v = volt_chunk
    n_channel = v.shape[1]
    res_n = 10  # length of each dB bar
    bl = -70
    bu = -20
    print(f'RMS(dB) |', end='')
    print(f't ={time.time() - _t0:6.1f} |', end='')
    for c in range(n_channel):
        u = v[:, c] - np.mean(v[:, c])
        dbu = 20.0 * log10_(np.sqrt(np.mean(u**2)))
        dbu_scaled = int(np.clip((dbu - bl)/(bu-bl) * res_n, 0, res_n))
        print('#'*dbu_scaled + ' '*(res_n - dbu_scaled), end='|')
    print()

if __name__ == "__main__":
    adc_conf = {
        'sampler_id': 'ad7606c',
        'sample_rate': 500000,
        'periodsize': 8192,
        'volt_range': [0, 5]
    }
    data_queue_max_size = 1000

    buf_queue = queue.Queue(data_queue_max_size)
    rec_thread = recThread('recorder', buf_queue, adc_conf)

    sz_chunk = adc_conf['periodsize'] * 2
    sz_hop   = adc_conf['periodsize']
    channel_selected = [0, 1, 2, 3, 4, 5, 6, 7]
    chunk_process_thread = sampleChunkThread('chunking',
        fn_analysis, buf_queue,
        channel_selected,
        sz_chunk, sz_hop)

    # Start threads
    rec_thread.start()
    chunk_process_thread.start()

    # wait until user interrupt
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    rec_thread.b_run = False
    chunk_process_thread.b_run = False

    # Wait for threads to finish
    rec_thread.join()
    chunk_process_thread.join()