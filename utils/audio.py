import struct
import math

def calc_rms(data, sample_width=2):
    """计算音频块的 RMS 能量值"""
    count = len(data) // sample_width
    if count == 0:
        return 0
    shorts = struct.unpack(f'{count}h', data)
    sum_squares = sum(s * s for s in shorts)
    return math.sqrt(sum_squares / count)
