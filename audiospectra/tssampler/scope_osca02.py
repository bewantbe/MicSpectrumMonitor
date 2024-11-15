# TO test:
#   python audio_win.py

import os
import sys
import time
import logging
from ctypes import (
    windll,
    POINTER,
    c_ubyte, c_ushort, c_int, c_uint, c_ulong,
    c_double,
)
import numpy as np

logger = logging.getLogger('lotoosc')

if __name__ == '__main__':
    import tssabc     # use as script
    # enable logging to info level
    logging.basicConfig(level=logging.DEBUG)
else:
    from . import tssabc    # use as lib
    logger.setLevel(logging.CRITICAL)

#DLL_ROOT = r'C:\Users\xyy82\soft\LOTO_USB示波器PC软件二次开发SDK_V9\dll\OSCA02_2002_H02\x64'
#OBJdll = windll.LoadLibrary(os.path.join(DLL_ROOT, "USBInterFace.dll"))
DLL_ROOT = os.path.dirname(os.path.abspath(__file__))
OBJdll = windll.LoadLibrary(os.path.join(DLL_ROOT, "USBInterFace_OSCA02.dll"))

# functions to communicate with the oscilloscope
SpecifyDevIdx = OBJdll.SpecifyDevIdx
SpecifyDevIdx.argtypes = [c_int]

DeviceOpen = OBJdll.DeviceOpen
DeviceOpen.restype = c_ulong

GetBuffer4Wr = OBJdll.GetBuffer4Wr
GetBuffer4Wr.argtypes = [c_int]
GetBuffer4Wr.restype = POINTER(c_ubyte)

USBCtrlTrans = OBJdll.USBCtrlTrans
USBCtrlTrans.argtypes = [c_ubyte, c_ushort, c_ulong]
USBCtrlTrans.restype = c_ubyte

SetInfo = OBJdll.SetInfo
SetInfo.argtypes = [c_double, c_double, c_ubyte, c_int, c_uint,  c_uint]

USBCtrlTransSimple = OBJdll.USBCtrlTransSimple
USBCtrlTransSimple.argtypes = [c_ulong]
USBCtrlTransSimple.restype = c_ulong

AiReadBulkData = OBJdll.AiReadBulkData
AiReadBulkData.argtypes = [c_ulong, c_uint, c_ulong, POINTER(c_ubyte), c_ubyte, c_uint]
AiReadBulkData.restype = c_ulong

DeviceClose = OBJdll.DeviceClose
DeviceClose.restype = c_ulong

EventCheck = OBJdll.EventCheck
EventCheck.argtypes = [c_int]
EventCheck.restype = c_int

def sampling_rate_normalization(sr_request, g_CtrlByte0):
    assert sr_request > 0
    sr_list = np.array([100e6]) / np.array([1, 8, 8*16, 8*16*16, 1042])  # [100e6, 12.5e6, 781e3, 49e3, 96e3]
    sr_cmdx = [0x00, 0x08, 0x0c, 0x0e, 0x04]
    # find the closest sampling rate in log scale
    sr_diff = np.abs(np.log(sr_list) - np.log(sr_request))
    sr_idx = np.argmin(sr_diff)
    g_CtrlByte0 &= 0xf0
    g_CtrlByte0 |= sr_cmdx[sr_idx]
    return sr_list[sr_idx], g_CtrlByte0

osc_volt_series = [8.0, 5.0, 2.5, 1.0, 0.5, 0.25, 0.1]

def set_volt_range(va_request, vb_request, g_CtrlByte1):
    v_list = np.array(osc_volt_series)
    mask_a_and = 0xF7
    mask_a_or  = [0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x00]
    mask_a_t   = [0x00, 0x02, 0x04, 0x06, 0x02, 0x04, 0x06]
    mask_b_and = 0xF9
    mask_b_or  = [0x00, 0x02, 0x04, 0x06, 0x02, 0x04, 0x06]
    mask_b_t   = [0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04]
    # if the request is out of range, raise error
    if va_request > v_list.max():
        raise ValueError(f"va_request {va_request} is out of range")
    if vb_request > v_list.max():
        raise ValueError(f"vb_request {vb_request} is out of range")
    # find lowest volt range that can cover the request
    va_idx = len(v_list)-1 - np.argmax(v_list[::-1] >= va_request)
    vb_idx = len(v_list)-1 - np.argmax(v_list[::-1] >= vb_request)
    # set the volt range
    g_CtrlByte1 &= mask_a_and
    g_CtrlByte1 |= mask_a_or[va_idx]
    USBCtrlTrans(c_ubyte(0x22), c_ushort(mask_a_t[va_idx]), c_ulong(1))
    USBCtrlTrans(c_ubyte(0x24), c_ushort(g_CtrlByte1), c_ulong(1))
    time.sleep(0.1)
    g_CtrlByte1 &= mask_b_and
    g_CtrlByte1 |= mask_b_or[vb_idx]
    USBCtrlTrans(c_ubyte(0x23), c_ushort(mask_b_t[vb_idx]), c_ulong(1))
    USBCtrlTrans(c_ubyte(0x24), c_ushort(g_CtrlByte1), c_ulong(1))
    return v_list[va_idx], v_list[vb_idx], va_idx, vb_idx, g_CtrlByte1

def read_volt_calib_data():
    #           A     B
    zvcmd = [0x01, 0x02,  # 8
             0x01, 0x02,  # 5
             0x0E, 0x0F,  # 2.5
             0x14, 0x15,  # 1
             0x12, 0x13,  # 0.5
             0x10, 0x11,  # 0.25
             0xA0, 0xA1]  # 0.1
    zero_volt_offset = np.zeros((7, 2), dtype=np.float64)
    for j, cmd in enumerate(zvcmd):
        zero_volt_offset[j // 2, j % 2] = USBCtrlTrans(0x90, cmd, 1)

    print('volt zero offset:\n', zero_volt_offset)

    #           A     B
    vscmd = [0xC2, 0xD2,  # 8
             0x03, 0x04,  # 5
             0x08, 0x0B,  # 2.5
             0x06, 0x07,  # 1
             0x09, 0x0C,  # 0.5
             0x0A, 0x0D,  # 0.25
             0x2A, 0x2D]  # 0.1
    volt_scale = np.zeros((7, 2), dtype=np.float64)
    for j, cmd in enumerate(vscmd):
        volt_scale[j // 2, j % 2] = USBCtrlTrans(0x90, cmd, 1)
    volt_scale = volt_scale * 2 / 255

    volt_scale[1, :] = [1.0333, 1.0079]
    print('volt scaling:\n', volt_scale)

    return zero_volt_offset, volt_scale

def user_volt_calib_data(raw_volt_offset_list = None, raw_volt_scale_list = None):
    # For calculation of volt_offsets, volt_scale
    # Final volt_true = (volt_raw - volt_offset) * (v_range * 2 / 255 * volt_scale)

    # user calibration point
    
    ## version 1
    j = 3 # 5V chA
    v_range = osc_volt_series[j // 2]

    calib_data_array = np.array([
        # v_ref0, v_raw0, v_ref1, v_raw0
        [0.0, 135.97, 8.000, 261.99],     # 8V chA
        [0.0, 132.78, 8.000, 257.83],     # 8V chB
        [0.0, 136.91, 3.210, 220.18],     # 5V chA
        [0.0, 133.08, 5.000, 259.58],     # 5V chB
        [0.0, 139.67, 2.500, 274.02],     # 2.5V chA
        [0.0, 134.49, 2.500, 268.84],     # 2.5V chB
        [0.0, 140.38, 1.000, 226.85],     # 1V chA
        [0.0, 135.27, 1.000, 221.74],     # 1V chB
        [0.0, 136.95, 0.500, 262.00],     # 0.5V chA
        [0.0, 133.09, 0.500, 257.18],     # 0.5V chB
        [0.0, 138.82, 0.250, 270.98],     # 0.25V chA
        [0.0, 134.46, 0.250, 266.62],     # 0.25V chB
        [0.0, 140.64, 0.100, 226.65],     # 0.1V chA
        [0.0, 135.22, 0.100, 220.33],     # 0.1V chB
    ])

    v_range = np.vstack([osc_volt_series, osc_volt_series]).T.flatten()
    v_ref0 = calib_data_array[:, 0]
    v_raw0 = calib_data_array[:, 1]
    v_ref1 = calib_data_array[:, 2]
    v_raw1 = calib_data_array[:, 3]
    raw_volt_scale_list  = (v_ref1 - v_ref0) / (v_raw1 - v_raw0) / (2 * v_range / 255)
    raw_volt_offset_list = v_raw0 - v_ref0 / (v_ref1 - v_ref0) * (v_raw1 - v_raw0)

    #assert abs((v_raw0 - volt_offset) * (v_range * 2 / 255 * volt_scale) - v_ref0) < 1e-10
    #assert abs((v_raw1 - volt_offset) * (v_range * 2 / 255 * volt_scale) - v_ref1) < 1e-10

    print('volt_offset:\n', raw_volt_offset_list)
    print('volt_scale: \n', raw_volt_scale_list)

    return raw_volt_offset_list, raw_volt_scale_list

class OSCA02Reader(tssabc.SampleReader):
    
    sampler_id = 'osca02'

    def __init__(self):
        self.initilized = False

    def init(self, sample_rate, periodsize, volt_range=5, stream_callback=None, **kwargs):
        self.chunk_size = periodsize * 2

        ## 1. set oscilloscope device model
        SpecifyDevIdx(c_int(6))            # 6: OSCA02
        logger.info("1. 设置当前示波器设备编号为6(OSCA02)!")

        ## 2. open device
        OpenRes = DeviceOpen()
        if OpenRes == 0:
            logger.info("2. 示波器设备连接 成功!")
        else:
            logger.error("2. 示波器设备连接 失败!")
            sys.exit(0)
        
        ## 3. get buffer address
        self.g_pBuffer = GetBuffer4Wr(c_int(-1))
        if 0 == self.g_pBuffer:
            logger.error("GetBuffer Failed!")
            sys.exit(0)
        else:
            logger.info("3. 获取数据缓冲区首地址 成功")

        # record IO control bits
        self.g_CtrlByte0 = 0
        self.g_CtrlByte1 = 0

        ## X: between steps 3 and 4, initialize hardware trigger, if trigger fails, comment out this method and do not call it.
        self.g_CtrlByte1 &= 0xdf
        self.g_CtrlByte1 |= 0x00
        USBCtrlTrans(c_ubyte(0x24), c_ushort(self.g_CtrlByte1), c_ulong(1))

        # set trigger position at 50% of the buffer
        USBCtrlTrans(c_ubyte(0x18), c_ushort(0xff), c_ulong(1))
        USBCtrlTrans(c_ubyte(0x17), c_ushort(0x7f), c_ulong(1))
        logger.info("X: 在步骤3和4之间，初始化硬件触发, 如果触发出问题, 请注释掉这个方法不调用")

        ## 4. set buffer size to 128KB, e.g. 64KB per channel
        SetInfo(c_double(1),  c_double(0),  c_ubyte(0x11),  c_int(0),   c_uint(0),  c_uint(self.chunk_size) )
        logger.info("4. 设置使用的缓冲区为128K字节, 即每个通道64K字节")

        ## 5. set oscilloscope sampling rate to 781kHz
        self.sampling_rate, self.g_CtrlByte0 = sampling_rate_normalization(sample_rate, self.g_CtrlByte0)
        Transres = USBCtrlTrans( c_ubyte(0x94),  c_ushort(self.g_CtrlByte0),  c_ulong(1))
        if 0 == Transres:
            logger.error("5. error")
            sys.exit(0)
        else:
            logger.info("5. 设置示波器采样率781Khz")
        
        # Enable channel B
        time.sleep(0.1)
        self.g_CtrlByte1 &= 0xfe
        self.g_CtrlByte1 |= 0x01
        USBCtrlTrans(c_ubyte(0x24),  c_ushort(self.g_CtrlByte1),  c_ulong(1))
        time.sleep(0.1)

        ## 6. set channel input range
        # chA range：-5V ~ +5V
        #self.g_CtrlByte1 &= 0xF7
        #self.g_CtrlByte1 |= 0x08
        #USBCtrlTrans(c_ubyte(0x22),  c_ushort(0x02),  c_ulong(1))
        #USBCtrlTrans(c_ubyte(0x24),  c_ushort(self.g_CtrlByte1),  c_ulong(1))
        #logger.info("6. 设置通道输入量程 chA 输入量程设置为：-5V ~ +5V")
        #time.sleep(0.1)

        # chB range：-5V ~ +5V
        #self.g_CtrlByte1 &= 0xF9
        #self.g_CtrlByte1 |= 0x02
        #USBCtrlTrans(c_ubyte(0x23),  c_ushort(0x00),  c_ulong(1))
        #USBCtrlTrans(c_ubyte(0x24),  c_ushort(self.g_CtrlByte1),  c_ulong(1))
        #logger.info("6. 设置通道输入量程 chB 输入量程设置为：-5V ~ +5V")

        if isinstance(volt_range, (int, float, np.float64, np.float32)):
            volt_a = volt_b = volt_range
        elif isinstance(volt_range, (list, tuple)):
            volt_a, volt_b = volt_range

        self.volt_a_max, self.volt_b_max, self.volt_ia, self.volt_ib, self.g_CtrlByte1 = \
            set_volt_range(volt_a, volt_b, self.g_CtrlByte1)
        self.volt_offsets, self.volt_scale = read_volt_calib_data()
        self.volt_a_offset = self.volt_offsets[self.volt_ia, 0]
        self.volt_b_offset = self.volt_offsets[self.volt_ib, 1]
        self.volt_a_scale = self.volt_scale[self.volt_ia, 0]
        self.volt_b_scale = self.volt_scale[self.volt_ib, 1]
        if 1:
            self.f_volt_cha = lambda va_bytes: \
                (np.float64(va_bytes) - self.volt_a_offset) * (self.volt_a_max * 2 / 255 * self.volt_a_scale)
            self.f_volt_chb = lambda vb_bytes: \
                (np.float64(vb_bytes) - self.volt_b_offset) * (self.volt_b_max * 2 / 255 * self.volt_b_scale)
        else:
            self.f_volt_cha = lambda va_bytes: \
                (np.float64(va_bytes) - self.volt_a_offset) * (2 * self.volt_a_max / 255)
            self.f_volt_chb = lambda vb_bytes: \
                (np.float64(vb_bytes) - self.volt_b_offset) * (2 * self.volt_b_max / 255)

        ## 7. set AC/DC channel coupling
        self.g_CtrlByte0 &= 0xef # chA DC coupling
        self.g_CtrlByte0 |= 0x10 # chA 0x10:DC coupling, 0x00 AC coupling
        USBCtrlTrans(c_ubyte(0x94),  c_ushort(self.g_CtrlByte0),  c_ulong(1))
        logger.info("7. 设置通道交直流耦合 设置chA为DC耦合")
        time.sleep(0.1)

        self.g_CtrlByte1 &= 0xef # chB DC coupling
        self.g_CtrlByte1 |= 0x10 # chB DC coupling
        Transres = USBCtrlTrans(c_ubyte(0x24),  c_ushort(self.g_CtrlByte1),  c_ulong(1))
        logger.info("7. 设置通道交直流耦合 设置chB为DC耦合")

        ## 8. set trigger mode as non-trigger
        USBCtrlTrans(c_ubyte(0xE7),  c_ushort(0x00),  c_ulong(1))
        logger.info("8. 设置当前触发模式为 无触发")

        if 0:
            # 8. trigger on rising edge, or turn off green LED
            USBCtrlTrans(c_ubyte(0xC5),  c_ushort(0x00),  c_ulong(1))
            logger.info("8. 上升沿，或者设置LED绿色灯灭")
        time.sleep(0.1)

        return self

    def read(self, n_frames = None):
        ## 9. start data acquisition
        USBCtrlTransSimple(c_ulong(0x33))
        logger.info("9. 控制设备开始AD采集")
        t1 = time.time()

        time_sample = self.chunk_size // 2 / self.sampling_rate
        timeout_ms = time_sample * 1000 * 10 + 10
        time.sleep(time_sample)

        ## 10. check if data acquisition and storage is complete, if so, return 33
        rFillUp = USBCtrlTransSimple(c_ulong(0x50))
        while (rFillUp != 33) and (t1 + timeout_ms / 1000 > time.time()):
            time.sleep(0.010)
            rFillUp = USBCtrlTransSimple(c_ulong(0x50))

        if 33 != rFillUp:
            logger.error("10. 缓冲区数据没有蓄满(查询结果)")
            sys.exit(0)
        else:
            logger.info("10. 缓冲区数据已经蓄满(查询结果)")

        rBulkRes = AiReadBulkData( c_ulong(64 * 1024 * 2) ,  c_uint(1),  c_ulong(2000) , self.g_pBuffer,  c_ubyte(0),  c_uint(0))
        if 0 == rBulkRes:
            logger.info("11. 传输获取采集的数据 成功!")
        else:
            logger.error("11. 传输获取采集的数据 失败!")
            #sys.exit(0)

        logger.info("X: waiting data transmition... ")
        ret = EventCheck(c_int(int(timeout_ms)))  # cost about 10~17ms
        t2 = time.time()
        if -1 == ret:
            logger.error("X: wrong calling")
            sys.exit(0)
        elif 0x555 == ret:
            logger.error("X: timeout")
        logger.debug(f"X: sampling data available, ret = {ret}, wait time = {t2 - t1:.3f} s, timeout = {timeout_ms:.3f} ms")

        # note: g_pBuffer is ubyte array
        sample_d = np.ctypeslib.as_array(self.g_pBuffer, shape=(self.chunk_size // 2, 2))
        assert (sample_d.shape[0] == self.chunk_size // 2) and (sample_d.shape[1] == 2)

        return sample_d

        #sample_f = np.zeros_like(sample_d, dtype=np.float32)
        #sample_f[:, 0] = self.f_volt_cha(sample_d[:, 0])
        #sample_f[:, 1] = self.f_volt_chb(sample_d[:, 1])

        #return sample_f

    def read_sleep(self, n_frames = None):
        ## 9. start data acquisition
        USBCtrlTransSimple(c_ulong(0x33))
        logger.info("9. 控制设备开始AD采集")

        logger.info("X: sleep200ms 等待采集.... ")
        time.sleep(0.200)

        ## 10. check if data acquisition and storage is complete, if so, return 33
        rFillUp = USBCtrlTransSimple(c_ulong(0x50))
        if 33 != rFillUp:
            logger.error("10. 缓冲区数据没有蓄满(查询结果)")
            sys.exit(0)
        else:
            logger.info("10. 缓冲区数据已经蓄满(查询结果)")

        ## 11. get samples, where the size is determined by SetInfo()
        rBulkRes = AiReadBulkData( c_ulong(64 * 1024 * 2) ,  c_uint(1),  c_ulong(2000) , self.g_pBuffer,  c_ubyte(0),  c_uint(0))

        if 0 == rBulkRes:
            logger.info("11. 传输获取采集的数据 成功!")
        else:
            logger.error("11. 传输获取采集的数据 失败!")
            sys.exit(0)

        # note: g_pBuffer is ubyte array
        #sample_d = np.ctypeslib.as_array(self.g_pBuffer, shape=(64 * 1024, 2)).astype(np.float32)
        sample_d = np.ctypeslib.as_array(self.g_pBuffer, shape=(64 * 1024, 2))
        assert (sample_d.shape[0] == 64 * 1024) and (sample_d.shape[1] == 2)

        return sample_d

    def close(self):
        logger.info("closing device...")
        DeviceClose()
        self.initilized = False

    def __del__(self):
        if self.initilized:
            self.close()

if __name__ == '__main__':
    # disable logger for matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    acq_dev = OSCA02Reader()
    sample_rate = 781000
    periodsize = 64 * 1024
    volt = 8  # V
    acq_dev.init(sample_rate, periodsize, volt, volt)
    sample_rate = acq_dev.sampling_rate

    user_volt_calib_data()

    if 0:
        d = acq_dev.read()
        v = np.zeros_like(d, dtype=np.float32)
        v[:, 0] = acq_dev.f_volt_cha(d[:, 0])
        v[:, 1] = acq_dev.f_volt_cha(d[:, 1])
        n_ignore = 100

        # plot the data in chA and chB in the same graph
        plt.figure()
        t_s = np.arange(n_ignore, v.shape[0]) / sample_rate
        plt.plot(t_s, v[n_ignore:, 0], '.-', label='chA')
        plt.plot(t_s, v[n_ignore:, 1], '.-', label='chB')
        plt.legend()
        plt.show()
    
    if 1:
        n_ignore = 100
        fig, ax = plt.subplots()
        t_s = np.arange(n_ignore, periodsize) / sample_rate
        line1, = ax.plot(t_s, np.zeros(periodsize - n_ignore), '.-', label='chA')
        line2, = ax.plot(t_s, np.zeros(periodsize - n_ignore), '.-', label='chB')
        ax.legend()

        set_v = []

        def update(frame):
            d = acq_dev.read()
            v = np.zeros_like(d, dtype=np.float32)
            v[:, 0] = acq_dev.f_volt_cha(d[:, 0])
            v[:, 1] = acq_dev.f_volt_cha(d[:, 1])
            print(f'mean volt: {v[n_ignore:, 0].mean():.5g},'
                             f'{v[n_ignore:, 1].mean():.5g}')
            v_m_a = d[n_ignore:, 0].astype(np.float32).mean()
            v_m_b = d[n_ignore:, 1].astype(np.float32).mean()
            if frame > 10:
                set_v.append([v_m_a, v_m_b])
            print(f'mean byte: {v_m_a:.5g},'
                             f'{v_m_b:.5g}')
            line1.set_ydata(v[n_ignore:, 0])
            line2.set_ydata(v[n_ignore:, 1])
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            return line1, line2

        ani = animation.FuncAnimation(fig, update, frames=2**30, blit=True)  # run infinitely
        #ani = animation.FuncAnimation(fig, update, frames=10, blit=True, repeat=False)  # run 10 frames
        plt.show()

        print(np.array(set_v).mean(axis=0))

    acq_dev.close()