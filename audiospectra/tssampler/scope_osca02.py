# TO test:
#   python scope_osca02.py
# TO debug usb:
#   export PYUSB_DEBUG=debug
#   python scope_osca02.py

import os
import sys
import time
import logging
from ctypes import (
    POINTER,
    c_ubyte, c_ushort, c_int, c_uint, c_ulong,
    c_double,
)
import numpy as np

logger = logging.getLogger('lotoosc')
_t_start = time.time()

# when running as a script, set the logging level to show more
if __name__ == '__main__':
    import tssabc     # use as script
    # enable logging to info level
    #logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
else:
    from . import tssabc    # use as lib
    logger.setLevel(logging.CRITICAL)
    #logging.basicConfig(level=logging.INFO)

if sys.platform == 'win32':
    from ctypes import windll
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
    SetInfo.argtypes = [c_double, c_double, c_ubyte, c_int, c_uint, c_uint]

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
else:  # assume linux
    # ref. https://github.com/pyusb/pyusb/blob/master/docs/tutorial.rst
    import usb.core
    import usb.util
    global _dev, _dev_cfg
    global _osc_id            # probably for indexing oscilloscope models
    global _buffer_byte_size
    _id_vendor_product_map = {
        6: (0x8312, 0x8a02),  # OSCA02
    }
    global _data_endpoint
    _data_endpoint = 0x82

    # debug decorator
    def pdebug(func):
        def wrapper(*args, **kwargs):
            # get current time starting from program start
            t = time.time() - _t_start
            print(f't = {t:7.3f}: calling {func.__name__} with args: {args}, kwargs: {kwargs}')
            ret = func(*args, **kwargs)
            t = time.time() - _t_start
            print(f't = {t:7.3f}: return: {ret}')
            return ret
        return wrapper

    @pdebug
    def SpecifyDevIdx(dev_idx):
        global _osc_id
        _osc_id = int(dev_idx.value)
        assert _osc_id in _id_vendor_product_map

    @pdebug
    def DeviceOpen():
        global _dev, _dev_cfg, _osc_id, _osc_buf_size
        USB_VENDOR_ID = _id_vendor_product_map[_osc_id][0]
        USB_PRODUCT_ID = _id_vendor_product_map[_osc_id][1]
        try:
            _dev = usb.core.find(idVendor=USB_VENDOR_ID, idProduct=USB_PRODUCT_ID)
            _dev.set_configuration()
            _dev_cfg = _dev.get_active_configuration()

            # 0x80: data from device to host, standard request, recipient is dev
            # 0x06: GET_DESCRIPTOR
            dev_id = _dev.ctrl_transfer(0x80, 0x06, 0x0301, 0x0000, 0x0032)
            print('DeviceOpen, dev_ID return:', dev_id)
        except Exception as e:
            print(f'Error: {e}')
            _dev = None
            _dev_cfg = None
            dev_id = -1
            return -1
        return 0
    
    @pdebug
    def DeviceClose():
        global _dev, _dev_cfg
        if _dev is not None:
            #usb.util.dispose_resources(_dev)
            _dev.reset()
            _dev = None
            _dev_cfg = None

    @pdebug
    def USBCtrlTrans(cmd, val, reserved):
        # ctrl_transfer(bmRequestType : uint8_t,
        #               bRequest      : uint8_t,
        #               wValue=0      : uint16_t,
        #               wIndex=0      : uint16_t,
        #               data_or_wLength = None : unsigned char * or uint16_t,
        #               timeout = None : unsigned int)
        # ref. https://github.com/pyusb/pyusb/blob/master/usb/core.py#L1057
        # ref. https://learn.microsoft.com/en-us/windows-hardware/drivers/usbcon/usb-control-transfer
        # ref. file:///usr/share/doc/libusb-1.0-doc/api-1.0/group__libusb__syncio.html#ga2f90957ccc1285475ae96ad2ceb1f58c
        cmd = cmd.value
        val = val.value
        ret = _dev.ctrl_transfer(0x80, cmd, val, 0x0000, 0x0001)
        return ret[0] if ret else None
    
    @pdebug
    def USBCtrlTransSimple(cmd):
        # usually cmd is 0x33 or 0x50 ulong
        cmd = cmd.value
        ret = _dev.ctrl_transfer(0x80, cmd, 0x0000, 0x0000, 0x0001)
        return ret[0] if ret else None

    @pdebug
    def SetInfo(p1, p2, p3, p4, p5, size):
        global _buffer_byte_size
        _buffer_byte_size = size.value
        return 0

    @pdebug
    def GetBuffer4Wr(index):
        # return an ubyte array
        global _buffer_byte_size
        index = index.value
        if index == -1:
            return (c_ubyte * _buffer_byte_size)()
        else:
            raise NotImplementedError(f'GetBuffer4Wr: index={index} not supported')
    
    @pdebug
    def AiReadBulkData(n_sample_byte, n_event, timeout_ms, buffer,
                       flag, reserved):
        # read buffer
        # _dev.read need array.array as buffer
        # timeout_ms can be None
        # always blocking read
        n_sample_byte = n_sample_byte.value
        n_event = n_event.value
        timeout_ms = timeout_ms.value
        assert n_sample_byte <= len(buffer)
        pos = 0
        n_one_read_ave = int(n_sample_byte / n_event + 0.5)
        # the reading of n_sample_byte is divided into n_event reads
        while pos < n_sample_byte:
            n_one_read = min(n_one_read_ave, n_sample_byte - pos)
            print('Trying read n_one_read =', n_one_read)
            r = _dev.read(_data_endpoint, n_one_read, timeout_ms)
            print(f'AiReadBulkData: read {len(r)} bytes')
            buffer[pos:pos+len(r)] = r
            pos += len(r)
        print('AiReadBulkData: read done', 'pos =', pos)
        return 0
    
    @pdebug
    def EventCheck(timeout_ms):
        # return 0x555 if timeout
        #time.sleep(0.02)  # seems _dev.read() will block the program
        return 0


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

#osc_volt_series = [8.0, 5.0, 2.5, 1.0, 0.5, 0.25, 0.1]
osc_volt_series = [10.0, 5.0, 2.5, 1.0, 0.5, 0.25, 0.1]
osc_vdiv_series = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]

def set_volt_range(va_request, vb_request, g_CtrlByte1):
    v_list = np.array(osc_volt_series)
    mask_a_and = 0xF7
    mask_a_or  = [0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x00]
    mask_a_t   = [0x00, 0x02, 0x04, 0x06, 0x02, 0x04, 0x06]
    mask_b_and = 0xF9
    mask_b_or  = [0x00, 0x02, 0x04, 0x06, 0x02, 0x04, 0x06]
    mask_b_t   = [0x00, 0x00, 0x00, 0x00, 0x40, 0x40, 0x40]
    # if the request is out of range, raise error
    if va_request > v_list.max():
        raise ValueError(f"va_request {va_request} is out of range")
    if vb_request > v_list.max():
        raise ValueError(f"vb_request {vb_request} is out of range")
    # find lowest volt range that can cover the request
    va_idx = len(v_list)-1 - np.argmax(v_list[::-1] >= va_request)
    vb_idx = len(v_list)-1 - np.argmax(v_list[::-1] >= vb_request)
    # set chA volt range
    g_CtrlByte1 &= mask_a_and
    g_CtrlByte1 |= mask_a_or[va_idx]
    USBCtrlTrans(c_ubyte(0x22), c_ushort(mask_a_t[va_idx]), c_ulong(1))
    USBCtrlTrans(c_ubyte(0x24), c_ushort(g_CtrlByte1), c_ulong(1))
    time.sleep(0.1)
    # set chB volt range
    g_CtrlByte1 &= mask_b_and
    g_CtrlByte1 |= mask_b_or[vb_idx]
    USBCtrlTrans(c_ubyte(0x23), c_ushort(mask_b_t[vb_idx]), c_ulong(1))
    USBCtrlTrans(c_ubyte(0x24), c_ushort(g_CtrlByte1), c_ulong(1))
    time.sleep(0.1)
    return v_list[va_idx], v_list[vb_idx], va_idx, vb_idx, g_CtrlByte1

def read_volt_calib_data():
    #           A     B
    zvcmd = [0x82, 0x72,  # 8
             0x01, 0x02,  # 5
             0x0E, 0x0F,  # 2.5
             0x14, 0x15,  # 1
             0x12, 0x13,  # 0.5
             0x10, 0x11,  # 0.25
             0xA0, 0xA1]  # 0.1
    zero_volt_offset = np.zeros((7, 2), dtype=np.float32)
    for j, cmd in enumerate(zvcmd):
        zero_volt_offset[j // 2, j % 2] = USBCtrlTrans(0x90, cmd, 1)

    #           A     B
    vscmd = [0xC2, 0xD2,  # 8
             0x03, 0x04,  # 5
             0x08, 0x0B,  # 2.5
             0x06, 0x07,  # 1
             0x09, 0x0C,  # 0.5
             0x0A, 0x0D,  # 0.25
             0x2A, 0x2D]  # 0.1
    volt_scale = np.zeros((7, 2), dtype=np.float32)
    for j, cmd in enumerate(vscmd):
        volt_scale[j // 2, j % 2] = USBCtrlTrans(0x90, cmd, 1)
    volt_scale = volt_scale * 2 / 255

    print('Internal calibration data:')
    print('volt offset: \n', zero_volt_offset)
    print('volt scaling:\n', volt_scale)

    return zero_volt_offset, volt_scale

def user_volt_calib_data(raw_volt_offset_list = None, raw_volt_scale_list = None):
    # For calculation of volt_offsets, volt_scale
    # Final volt_true = (volt_raw - volt_offset) * (v_range * 2 / 255 * volt_scale)

    # user calibration point
    calib_data_array = np.array([
        # v_ref0, v_raw0, v_ref1, v_raw0
        [0.0, 135.97,  4.722, 196.14],     # 8V chA
        [0.0, 132.78,  4.722, 192.54],     # 8V chB
        [0.0, 136.91,  3.144, 218.78],     # 5V chA
        [0.0, 133.08,  3.144, 214.90],     # 5V chB
        [0.0, 138.63,  1.577, 224.49],     # 2.5V chA
        [0.0, 134.49,  1.577, 220.34],     # 2.5V chB
        [0.0, 140.36, 0.7990, 210.84],     # 1V chA
        [0.0, 135.23, 0.7990, 205.65],     # 1V chB
        [0.0, 136.90, 0.4494, 250.97],     # 0.5V chA
        [0.0, 133.01, 0.4494, 246.56],     # 0.5V chB
        [0.0, 138.84, 0.1590, 223.14],     # 0.25V chA
        [0.0, 134.06, 0.1590, 218.30],     # 0.25V chB
        [0.0, 140.73,0.07969, 209.06],     # 0.1V chA
        [0.0, 135.00,0.07969, 202.96],     # 0.1V chB
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

    print('User calibration:')
    print('volt_offset:\n', raw_volt_offset_list.reshape(-1, 2))
    print('volt_scale: \n', raw_volt_scale_list.reshape(-1, 2))

    return raw_volt_offset_list.reshape(-1, 2), raw_volt_scale_list.reshape(-1, 2)

class OSCA02Reader(tssabc.SampleReader):
    
    sampler_id = 'osca02'
    frame_byte_size = 2
    _n_frame_discard = 100

    def __init__(self):
        self.initilized = False

    def init(self, sample_rate, periodsize, volt_range=5, stream_callback=None, 
             indicate_discontinuous = False, **kwargs):
        """Initialize the oscilloscope device
        Parameters:
            sample_rate: will be normalized to the closest value in the list
                         [100e6, 12.5e6, 781e3, 49e3, 96e3]
            periodsize: number of frames (samples) for each period.
                        Internally, the buffer size must be multiple of 8kB,
                        i.e. the periodsize must be of the form (k integer)
                           = 4kB * k - _n_frame_discard
        """
        print("--- periodsize = ", periodsize)
        self.chunk_size = periodsize * self.frame_byte_size  # chunk size for output
        self.chunk_size_raw = self.chunk_size + \
            self._n_frame_discard * self.frame_byte_size  # chunk size for internal

        ## 1. set oscilloscope device model
        SpecifyDevIdx(c_int(6))            # 6: OSCA02
        logger.debug("1. 设置当前示波器设备编号为6(OSCA02)!")

        ## 2. open device
        OpenRes = DeviceOpen()
        if OpenRes == 0:
            logger.debug("2. 示波器设备连接 成功!")
        else:
            logger.error("2. 示波器设备连接 失败!")
            raise ConnectionError("Fail to connect to the device")
        
        ## 4. set buffer size to 128KB, e.g. 64KB per channel
        SetInfo(c_double(1), c_double(0), c_ubyte(0x11), c_int(0), c_uint(0),
                c_uint(self.chunk_size_raw))
        logger.debug("4. 设置使用的缓冲区为128K字节, 如每个通道64K字节")

        ## 3. get buffer address
        self.g_pBuffer = GetBuffer4Wr(c_int(-1))
        if 0 == self.g_pBuffer:
            logger.error("GetBuffer Failed!")
            raise MemoryError("Fail to get buffer address")
        else:
            logger.debug("3. 获取数据缓冲区首地址 成功")

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
        logger.debug("X: 在步骤3和4之间，初始化硬件触发, 如果触发出问题, 请注释掉这个方法不调用")

        ## 5. set oscilloscope sampling rate to 781kHz
        self.sampling_rate, self.g_CtrlByte0 = sampling_rate_normalization(sample_rate, self.g_CtrlByte0)
        Transres = USBCtrlTrans( c_ubyte(0x94),  c_ushort(self.g_CtrlByte0),  c_ulong(1))
        if 0 == Transres:
            logger.error("5. error")
            raise ConnectionError("Fail to set sampling rate")
        else:
            logger.debug("5. 设置示波器采样率781Khz")
        
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
        #self.volt_offsets, self.volt_scale = read_volt_calib_data()
        self.volt_offsets, self.volt_scale = user_volt_calib_data()
        self.volt_a_offset = self.volt_offsets[self.volt_ia, 0]
        self.volt_b_offset = self.volt_offsets[self.volt_ib, 1]
        self.volt_a_scale = self.volt_scale[self.volt_ia, 0]
        self.volt_b_scale = self.volt_scale[self.volt_ib, 1]
        self.f_volt_cha = lambda va_bytes: \
            (np.float32(va_bytes) - self.volt_a_offset) * (self.volt_a_max * 2 / 255 * self.volt_a_scale)
        self.f_volt_chb = lambda vb_bytes: \
            (np.float32(vb_bytes) - self.volt_b_offset) * (self.volt_b_max * 2 / 255 * self.volt_b_scale)
        print(f'volt_a_max: {self.volt_a_max:.2f}, volt_b_max: {self.volt_b_max:.2f}')
        print(f'volt_a_offset: {self.volt_a_offset:.2f}, volt_b_offset: {self.volt_b_offset:.2f}')
        print(f'volt_a_scale: {self.volt_a_scale:.4f}, volt_b_scale: {self.volt_b_scale:.4f}')

        ## 7. set AC/DC channel coupling
        self.g_CtrlByte0 &= 0xef # chA DC coupling
        self.g_CtrlByte0 |= 0x10 # chA 0x10:DC coupling, 0x00 AC coupling
        USBCtrlTrans(c_ubyte(0x94),  c_ushort(self.g_CtrlByte0),  c_ulong(1))
        logger.debug("7. 设置通道交直流耦合 设置chA为DC耦合")
        time.sleep(0.1)

        self.g_CtrlByte1 &= 0xef # chB DC coupling
        self.g_CtrlByte1 |= 0x10 # chB DC coupling
        Transres = USBCtrlTrans(c_ubyte(0x24),  c_ushort(self.g_CtrlByte1),  c_ulong(1))
        logger.debug("7. 设置通道交直流耦合 设置chB为DC耦合")

        ## 8. set trigger mode as non-trigger
        USBCtrlTrans(c_ubyte(0xE7),  c_ushort(0x00),  c_ulong(1))
        logger.debug("8. 设置当前触发模式为 无触发")

        if 1:
            # 8. trigger on rising edge, or turn off green LED
            USBCtrlTrans(c_ubyte(0xC5),  c_ushort(0x00),  c_ulong(1))
            logger.debug("8. 上升沿，或者设置LED绿色灯灭")
        time.sleep(0.1)

        self.indicate_discontinuous = indicate_discontinuous
        self._discontinuity_signal = 0

        self.initilized = True

        return self

    def read_raw(self):
        
        # send discontinuity signal if any
        if self.indicate_discontinuous:
            if self._discontinuity_signal:
                self._discontinuity_signal = 0
                return 0
            self._discontinuity_signal = 1

        ## 9. start data acquisition
        USBCtrlTransSimple(c_ulong(0x33))
        logger.debug("9. 控制设备开始AD采集")
        t1 = time.time()

        time_sample = self.chunk_size_raw // 2 / self.sampling_rate
        timeout_ms = time_sample * 1000 * 10 + 10
        time.sleep(time_sample)

        ## 10. check if data acquisition and storage is complete, if so, return 33
        rFillUp = USBCtrlTransSimple(c_ulong(0x50))
        while (rFillUp != 33) and (t1 + timeout_ms / 1000 > time.time()):
            time.sleep(0.010)
            rFillUp = USBCtrlTransSimple(c_ulong(0x50))

        if 33 != rFillUp:
            logger.error("10. 缓冲区数据没有蓄满(查询结果)")
            raise ValueError("Fail to wait buffer fill up")
        else:
            logger.debug("10. 缓冲区数据已经蓄满(查询结果)")

        rBulkRes = AiReadBulkData(c_ulong(self.chunk_size_raw), c_uint(1),
                                  c_ulong(int(timeout_ms)),
                                  self.g_pBuffer, c_ubyte(0), c_uint(0))
        if 0 == rBulkRes:
            logger.debug("11. 传输获取采集的数据 成功!")
        else:
            logger.error("11. 传输获取采集的数据 失败!")
            raise IOError("Fail to read bulk data")

        logger.debug("X: waiting data transmition... ")
        ret = EventCheck(c_int(int(timeout_ms)))  # cost about 10~17ms
        t2 = time.time()
        if -1 == ret:
            logger.error("X: wrong calling")
            raise RuntimeError("Fail to run EventCheck()")
        elif 0x555 == ret:
            logger.error("X: timeout")
        logger.debug(f"X: sampling data available, ret = {ret}, wait time = {t2 - t1:.3f} s, timeout = {timeout_ms:.3f} ms")

        # note: g_pBuffer is ubyte array
        sample_d = np.ctypeslib.as_array(self.g_pBuffer, shape=(self.chunk_size_raw,))
        sample_d = sample_d.reshape(-1, 2)
        print(f'sample_d.shape = {sample_d.shape}')
        print(f'  Test dim1:', (sample_d.shape[1] == 2))
        print(f'  Test dim0: sample_d.shape[0] = {sample_d.shape[0]}', 'chunk_size_raw // 2 =', self.chunk_size_raw // 2, ' test', (sample_d.shape[0] == self.chunk_size_raw // 2))
        assert (sample_d.shape[0] == self.chunk_size_raw // 2) and (sample_d.shape[1] == 2)

        return sample_d[self._n_frame_discard:, :]

    def read(self):
        """Read sampling data, and return normalized voltage data"""
        d = self.read_raw()
        if isinstance(d, int):  # special signal
            return d
        a = np.array(d, dtype=np.float32)
        a[:, 0] = (a[:,0] - self.volt_a_offset) / 128.0
        a[:, 1] = (a[:,1] - self.volt_b_offset) / 128.0
        return a

    def raw_to_physical_value(self, d):
        """Convert the raw data to physical value"""
        a = np.array(d, dtype=np.float32)
        a[:, 0] = (a[:, 0] - self.volt_a_offset) * (self.volt_a_max * 2 / 255 * self.volt_a_scale)
        a[:, 1] = (a[:, 1] - self.volt_b_offset) * (self.volt_b_max * 2 / 255 * self.volt_b_scale)
        return a

    def read_physical(self):
        """Read sampling data, and return physical voltage data"""
        d = self.read_raw()
        if isinstance(d, int):  # special signal
            return d
        return self.raw_to_physical_value(d)

    def close(self):
        logger.debug("closing device...")
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
    periodsize = 64 * 1024 - acq_dev._n_frame_discard
    volt = 5  # V
    acq_dev.init(sample_rate, periodsize, volt)
    sample_rate = acq_dev.sampling_rate

    if 0:
        d = acq_dev.read_raw()
        v = acq_dev.raw_to_physical_value(d)

        # plot the data in chA and chB in the same graph
        plt.figure()
        t_s = np.arange(v.shape[0]) / sample_rate
        plt.plot(t_s, v[:, 0], '.-', label='chA')
        plt.plot(t_s, v[:, 1], '.-', label='chB')
        plt.legend()
        plt.show()
    
    if 1:
        fig, ax = plt.subplots()
        t_s = np.arange(periodsize) / sample_rate
        line1, = ax.plot(t_s, np.zeros(periodsize), '.-', label='chA')
        line2, = ax.plot(t_s, np.zeros(periodsize), '.-', label='chB')
        ax.legend()

        set_v = []

        def update(frame):
            d = acq_dev.read_raw()
            v = acq_dev.raw_to_physical_value(d)
            print(f'mean volt: {v[:, 0].mean():.5g},'
                             f'{v[:, 1].mean():.5g}')
            v_m_a = d[:, 0].astype(np.float32).mean()
            v_m_b = d[:, 1].astype(np.float32).mean()
            if frame > 5:
                set_v.append([v_m_a, v_m_b])
            print(f'mean byte: {v_m_a:.5g},'
                             f'{v_m_b:.5g}')
            line1.set_ydata(v[:, 0])
            line2.set_ydata(v[:, 1])
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            return line1, line2

        ani = animation.FuncAnimation(fig, update, frames=2**30, blit=True)  # run infinitely
        #ani = animation.FuncAnimation(fig, update, frames=10, blit=True, repeat=False)  # run 10 frames
        plt.show()

        v_raw_mean = np.array(set_v).mean(axis=0)
        print(f'v_raw:  {v_raw_mean[0]:.2f}  {v_raw_mean[1]:.2f}')
        v_cal_mean = [acq_dev.f_volt_cha(v_raw_mean[0]), acq_dev.f_volt_chb(v_raw_mean[1])]
        print(f'v_raw_cal:  {v_cal_mean[0]:.3f}  {v_cal_mean[1]:.3f}')

    acq_dev.close()
