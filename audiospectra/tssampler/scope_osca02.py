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

# enable logging to info level
logging.basicConfig(level=logging.DEBUG)

import tssabc     # use as script
#from . import tssabc    # use as lib

DLL_ROOT = r'C:\Users\xyy82\soft\LOTO_USB示波器PC软件二次开发SDK_V9\dll\OSCA02_2002_H02\x64'
OBJdll = windll.LoadLibrary(os.path.join(DLL_ROOT, "USBInterFace.dll"))

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


class OSCA02Reader(tssabc.SampleReader):
    
    sampler_id = 'osca02'

    def __init__(self):
        self.initilized = False

    def init(self, sample_rate, chunk_size, stream_callback=None, **kwargs):
        ## 1. set oscilloscope device model
        SpecifyDevIdx(c_int(6))            # 6: OSCA02
        logging.info("1. 设置当前示波器设备编号为6(OSCA02)!")

        ## 2. open device
        OpenRes = DeviceOpen()
        if OpenRes == 0:
            logging.info("2. 示波器设备连接 成功!")
        else:
            logging.error("2. 示波器设备连接 失败!")
            sys.exit(0)
        
        ## 3. get buffer address
        self.g_pBuffer = GetBuffer4Wr(c_int(-1))
        if 0 == self.g_pBuffer:
            logging.error("GetBuffer Failed!")
            sys.exit(0)
        else:
            logging.info("3. 获取数据缓冲区首地址 成功")

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
        logging.info("X: 在步骤3和4之间，初始化硬件触发, 如果触发出问题, 请注释掉这个方法不调用")

        ## 4. set buffer size to 128KB, e.g. 64KB per channel
        SetInfo(c_double(1),  c_double(0),  c_ubyte(0x11),  c_int(0),   c_uint(0),  c_uint(64 * 1024 * 2) )
        logging.info("4. 设置使用的缓冲区为128K字节, 即每个通道64K字节")

        ## 5. set oscilloscope sampling rate to 781kHz
        self.g_CtrlByte0 &= 0xf0
        self.g_CtrlByte0 |= 0x0c
        Transres = USBCtrlTrans( c_ubyte(0x94),  c_ushort(self.g_CtrlByte0),  c_ulong(1))
        if 0 == Transres:
            logging.error("5. error")
            sys.exit(0)
        else:
            logging.info("5. 设置示波器采样率781Khz")
        
        # Enable channel B
        time.sleep(0.1)
        self.g_CtrlByte1 &= 0xfe
        self.g_CtrlByte1 |= 0x01
        USBCtrlTrans(c_ubyte(0x24),  c_ushort(self.g_CtrlByte1),  c_ulong(1))
        time.sleep(0.1)

        ## 6. set channel input range
        # chA range：-5V ~ +5V
        self.g_CtrlByte1 &= 0xF7
        self.g_CtrlByte1 |= 0x08
        USBCtrlTrans(c_ubyte(0x22),  c_ushort(0x02),  c_ulong(1))
        USBCtrlTrans(c_ubyte(0x24),  c_ushort(self.g_CtrlByte1),  c_ulong(1))
        logging.info("6. 设置通道输入量程 chA 输入量程设置为：-5V ~ +5V")
        time.sleep(0.1)

        # chB range：-5V ~ +5V
        self.g_CtrlByte1 &= 0xF9
        self.g_CtrlByte1 |= 0x02
        USBCtrlTrans(c_ubyte(0x23),  c_ushort(0x00),  c_ulong(1))
        USBCtrlTrans(c_ubyte(0x24),  c_ushort(self.g_CtrlByte1),  c_ulong(1))
        logging.info("6. 设置通道输入量程 chB 输入量程设置为：-5V ~ +5V")

        ## 7. set AC/DC channel coupling
        self.g_CtrlByte0 &= 0xef # chA DC coupling
        self.g_CtrlByte0 |= 0x10 # chA DC coupling
        USBCtrlTrans(c_ubyte(0x94),  c_ushort(self.g_CtrlByte0),  c_ulong(1))
        logging.info("7. 设置通道交直流耦合 设置chA为DC耦合")
        time.sleep(0.1)

        self.g_CtrlByte1 &= 0xef # chB DC coupling
        self.g_CtrlByte1 |= 0x10 # chB DC coupling
        Transres = USBCtrlTrans(c_ubyte(0x24),  c_ushort(self.g_CtrlByte1),  c_ulong(1))
        logging.info("7. 设置通道交直流耦合 设置chB为DC耦合")

        ## 8. set trigger mode as non-trigger
        USBCtrlTrans(c_ubyte(0xE7),  c_ushort(0x00),  c_ulong(1))
        logging.info("8. 设置当前触发模式为 无触发")

        if 0:
            # 8. rising edge, or turn off green LED
            USBCtrlTrans(c_ubyte(0xC5),  c_ushort(0x00),  c_ulong(1))
            logging.info("8. 上升沿，或者设置LED绿色灯灭")
        time.sleep(0.1)

        return self

    def read(self, n_frames = None):
        ## 9. start data acquisition
        USBCtrlTransSimple(c_ulong(0x33))
        logging.info("9. 控制设备开始AD采集")
        t1 = time.time()

        time_sample = 64*1024 / 781000
        timeout_ms = time_sample * 1000 * 10 + 10
        time.sleep(time_sample)

        ## 10. check if data acquisition and storage is complete, if so, return 33
        rFillUp = USBCtrlTransSimple(c_ulong(0x50))
        while (rFillUp != 33) and (t1 + timeout_ms / 1000 > time.time()):
            time.sleep(0.010)
            rFillUp = USBCtrlTransSimple(c_ulong(0x50))

        if 33 != rFillUp:
            logging.error("10. 缓冲区数据没有蓄满(查询结果)")
            sys.exit(0)
        else:
            logging.info("10. 缓冲区数据已经蓄满(查询结果)")

        rBulkRes = AiReadBulkData( c_ulong(64 * 1024 * 2) ,  c_uint(1),  c_ulong(2000) , self.g_pBuffer,  c_ubyte(0),  c_uint(0))
        if 0 == rBulkRes:
            logging.info("11. 传输获取采集的数据 成功!")
        else:
            logging.error("11. 传输获取采集的数据 失败!")
            #sys.exit(0)

        logging.info("X: waiting data transmition... ")
        ret = EventCheck(c_int(int(timeout_ms)))  # cost about 10~17ms
        t2 = time.time()
        if -1 == ret:
            logging.error("X: wrong calling")
            sys.exit(0)
        elif 0x555 == ret:
            logging.error("X: timeout")
        logging.debug(f"X: sampling data available, ret = {ret}, wait time = {t2 - t1:.3f} s, timeout = {timeout_ms:.3f} ms")

        # note: g_pBuffer is ubyte array
        sample_d = np.ctypeslib.as_array(self.g_pBuffer, shape=(64 * 1024, 2))
        assert (sample_d.shape[0] == 64 * 1024) and (sample_d.shape[1] == 2)

        return sample_d

    def read_sleep(self, n_frames = None):
        ## 9. start data acquisition
        USBCtrlTransSimple(c_ulong(0x33))
        logging.info("9. 控制设备开始AD采集")

        logging.info("X: sleep200ms 等待采集.... ")
        time.sleep(0.200)

        ## 10. check if data acquisition and storage is complete, if so, return 33
        rFillUp = USBCtrlTransSimple(c_ulong(0x50))
        if 33 != rFillUp:
            logging.error("10. 缓冲区数据没有蓄满(查询结果)")
            sys.exit(0)
        else:
            logging.info("10. 缓冲区数据已经蓄满(查询结果)")

        ## 11. get samples, where the size is determined by SetInfo()
        rBulkRes = AiReadBulkData( c_ulong(64 * 1024 * 2) ,  c_uint(1),  c_ulong(2000) , self.g_pBuffer,  c_ubyte(0),  c_uint(0))

        if 0 == rBulkRes:
            logging.info("11. 传输获取采集的数据 成功!")
        else:
            logging.error("11. 传输获取采集的数据 失败!")
            sys.exit(0)

        # note: g_pBuffer is ubyte array
        #sample_d = np.ctypeslib.as_array(self.g_pBuffer, shape=(64 * 1024, 2)).astype(np.float32)
        sample_d = np.ctypeslib.as_array(self.g_pBuffer, shape=(64 * 1024, 2))
        assert (sample_d.shape[0] == 64 * 1024) and (sample_d.shape[1] == 2)

        return sample_d

    def close(self):
        logging.info("closing device...")
        DeviceClose()
        self.initilized = False

    def __del__(self):
        if self.initilized:
            self.close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    acq_dev = OSCA02Reader()
    sample_rate = 781000
    chunk_size = 128 * 1024
    acq_dev.init(sample_rate, chunk_size)

    if 0:
        d = acq_dev.read()

        # plot the data in chA and chB in the same graph
        plt.figure()
        t_s = np.arange(0, d.shape[0]) / sample_rate
        plt.plot(t_s, d[:, 0], '.-', label='chA')
        plt.plot(t_s, d[:, 1], '.-', label='chB')
        plt.legend()
        plt.show()
    
    if 1:
        import matplotlib.animation as animation

        fig, ax = plt.subplots()
        t_s = np.arange(0, chunk_size // 2) / sample_rate
        line1, = ax.plot(t_s, np.zeros(chunk_size // 2), '.-', label='chA')
        line2, = ax.plot(t_s, np.zeros(chunk_size // 2), '.-', label='chB')
        ax.legend()

        def update(frame):
            d = acq_dev.read()
            line1.set_ydata(d[:, 0])
            line2.set_ydata(d[:, 1])
            ax.relim()
            ax.autoscale_view()
            return line1, line2

        ani = animation.FuncAnimation(fig, update, blit=True)  # run infinitely
        #ani = animation.FuncAnimation(fig, update, frames=10, blit=True, repeat=False)  # run 10 frames
        plt.show()

    acq_dev.close()