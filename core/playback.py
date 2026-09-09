import os
import time
import logging
import queue
import subprocess
import threading
import shutil
from config import HW_SAMPLE_RATE, PLAYBACK_CHANNELS, DING_PCM_PATH

logger = logging.getLogger("LocalVoiceGateway")

GLOBAL_AUDIO_QUEUE = queue.Queue()
_is_playing = False
_playing_lock = threading.Lock()
_current_player_proc = None
_player_proc_lock = threading.Lock()

class PyAudioPlayer:
    """跨平台音频播放器包装 (统一包装为与 subprocess.Popen 兼容的对象)"""
    def __init__(self, sample_rate: int, channels: int):
        import pyaudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            output=True
        )
        self.closed = False

    class _StdinWrapper:
        def __init__(self, parent):
            self.parent = parent

        def write(self, data: bytes):
            if not self.parent.closed:
                try:
                    self.parent.stream.write(data)
                except Exception as e:
                    logger.debug(f"PyAudio 播放写入异常: {e}")

        def flush(self):
            pass

        def close(self):
            self.parent.close()

    @property
    def stdin(self):
        return self._StdinWrapper(self)

    def wait(self, timeout=None):
        self.close()

    def kill(self):
        self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            try:
                self.p.terminate()
            except Exception:
                pass

_playback_callback = None

def set_playback_callback(callback):
    """注册扬声器播放状态改变回调函数"""
    global _playback_callback
    _playback_callback = callback

def get_is_playing():
    global _is_playing
    return _is_playing

def set_is_playing(status: bool):
    global _is_playing
    changed = False
    with _playing_lock:
        if _is_playing != status:
            _is_playing = status
            changed = True
    if changed and _playback_callback:
        try:
            _playback_callback(status)
        except Exception as e:
            logger.debug(f"播放状态回调通知异常: {e}")

def stop_playback():
    """清空音频队列并立即终止当前音频播报 (闭嘴功能)"""
    global _current_player_proc
    while not GLOBAL_AUDIO_QUEUE.empty():
        try:
            GLOBAL_AUDIO_QUEUE.get_nowait()
            GLOBAL_AUDIO_QUEUE.task_done()
        except Exception:
            break

    with _player_proc_lock:
        if _current_player_proc:
            try:
                _current_player_proc.kill()
            except Exception:
                pass
            _current_player_proc = None

    if os.name != 'nt' and shutil.which("pkill"):
        try:
            subprocess.run(["pkill", "-9", "-f", "aplay -D default"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    set_is_playing(False)
    logger.info("🛑 [音频播报]: 已收到停止指令，清空队列并中断扬声器")

def generate_chime_pcm(output_path: str, sample_rate: int = HW_SAMPLE_RATE, channels: int = PLAYBACK_CHANNELS):
    """
    纯算法合成科技感清脆双音琶音 (High C 1046Hz -> High G 1568Hz)，
    专门针对微型会议扬声器与音箱中高频优化的极速提示音，响度充足、穿透力极强且无爆音。
    """
    import math
    import struct

    duration = 0.20  # 200ms
    num_samples = int(sample_rate * duration)
    frames = bytearray()

    split_sample = int(sample_rate * 0.065) # 前 65ms 为第一个高音
    fade_in_len = int(sample_rate * 0.005)   # 5ms 平滑淡入防止开关破音

    for i in range(num_samples):
        t = i / sample_rate
        if i < split_sample:
            # 阶段 1: 1046.5 Hz (High C) + 2093 Hz 微量倍频
            f1, f2 = 1046.5, 2093.0
            raw = 0.85 * math.sin(2 * math.pi * f1 * t) + 0.15 * math.sin(2 * math.pi * f2 * t)
            if i < fade_in_len:
                env = 0.5 * (1 - math.cos(math.pi * i / fade_in_len))
            else:
                env = 1.0 - 0.1 * ((i - fade_in_len) / (split_sample - fade_in_len))
        else:
            # 阶段 2: 1567.98 Hz (High G) + 3135.96 Hz
            f1, f2 = 1567.98, 3135.96
            raw = 0.88 * math.sin(2 * math.pi * f1 * t) + 0.12 * math.sin(2 * math.pi * f2 * t)
            decay_len = num_samples - split_sample
            prog = (i - split_sample) / decay_len
            env = math.exp(-prog * 4.0)

        # 峰值拉满至 31500 (满幅 32767 的 96%)，获得极度清晰通透的听觉响度
        val = int(raw * env * 31500)
        val = max(-32767, min(32767, val))
        sample_bytes = struct.pack("<h", val)
        for _ in range(channels):
            frames.extend(sample_bytes)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(frames)
    return output_path

def ensure_ding_pcm():
    """确保 assets/ding.pcm 存在且声道/采样率与当前硬件配置完全对齐"""
    expected_bytes = int(HW_SAMPLE_RATE * 0.20) * PLAYBACK_CHANNELS * 2
    if os.path.exists(DING_PCM_PATH):
        try:
            if os.path.getsize(DING_PCM_PATH) == expected_bytes:
                return
            logger.info(f"🔔 硬件采样率或声道发生变更 (HW_SAMPLE_RATE={HW_SAMPLE_RATE}, CHANNELS={PLAYBACK_CHANNELS})，重新合成自适应 ding.pcm")
        except Exception:
            pass

    os.makedirs(os.path.dirname(DING_PCM_PATH), exist_ok=True)
    try:
        generate_chime_pcm(DING_PCM_PATH, sample_rate=HW_SAMPLE_RATE, channels=PLAYBACK_CHANNELS)
        logger.info(f"🔔 已生成自适应硬件规格的提示音: ding.pcm ({HW_SAMPLE_RATE}Hz, {PLAYBACK_CHANNELS}声道)")
    except Exception as e:
        logger.error(f"生成提示音失败: {e}")

# 模块加载时确保提示音就绪
ensure_ding_pcm()

def play_ding():
    """播放 ding.pcm 提示音"""
    if not os.path.exists(DING_PCM_PATH):
        ensure_ding_pcm()

    if os.path.exists(DING_PCM_PATH):
        try:
            with open(DING_PCM_PATH, "rb") as f:
                pcm_data = f.read()
            GLOBAL_AUDIO_QUEUE.put(pcm_data)
        except Exception as e:
            logger.error(f"播放 ding 失败: {e}")

def create_audio_player():
    """创建音频输出流进程或对象 (优先 ALSA aplay，Windows/非 ALSA 环境自动回退至 PyAudio)"""
    if os.name != "nt" and shutil.which("aplay"):
        return subprocess.Popen(
            ["aplay", "-D", "default", "-f", "S16_LE", "-r", str(HW_SAMPLE_RATE), 
             "-c", str(PLAYBACK_CHANNELS), "-t", "raw"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        return PyAudioPlayer(HW_SAMPLE_RATE, PLAYBACK_CHANNELS)

def playback_worker(audio_queue: queue.Queue):
    """音频播放工作线程：从队列拉取原始双声道 PCM 字节并直写播放管道"""
    global _current_player_proc
    player_proc = None

    try:
        while True:
            try:
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                if player_proc:
                    set_is_playing(False)
                    try:
                        player_proc.stdin.close()
                        player_proc.wait(timeout=1)
                    except:
                        try:
                            player_proc.kill()
                        except:
                            pass
                    with _player_proc_lock:
                        if _current_player_proc is player_proc:
                            _current_player_proc = None
                    player_proc = None
                continue

            set_is_playing(True)

            if chunk is None:
                if player_proc:
                    try:
                        player_proc.stdin.close()
                        player_proc.wait(timeout=1)
                    except:
                        player_proc.kill()
                    with _player_proc_lock:
                        if _current_player_proc is player_proc:
                            _current_player_proc = None
                    player_proc = None
                set_is_playing(False)
                audio_queue.task_done()
                continue

            if player_proc is None:
                try:
                    player_proc = create_audio_player()
                    with _player_proc_lock:
                        _current_player_proc = player_proc

                    # 硬件 DAC 建立缓冲：先注入 150ms 静音防止开头音节被吃
                    silence_padding = b'\x00' * int(HW_SAMPLE_RATE * PLAYBACK_CHANNELS * 2 * 0.15)
                    player_proc.stdin.write(silence_padding)
                    player_proc.stdin.flush()
                except Exception as e:
                    logger.error(f"启动音频播放进程失败: {e}")
                    player_proc = None
                    with _player_proc_lock:
                        _current_player_proc = None
                    audio_queue.task_done()
                    continue

            try:
                player_proc.stdin.write(chunk)
                player_proc.stdin.flush()
            except Exception as e:
                logger.error(f"播放器写入音频失败: {e}")
                try:
                    player_proc.kill()
                except:
                    pass
                with _player_proc_lock:
                    if _current_player_proc is player_proc:
                        _current_player_proc = None
                player_proc = None

            audio_queue.task_done()
    finally:
        set_is_playing(False)
        if player_proc:
            try:
                player_proc.kill()
            except:
                pass
        with _player_proc_lock:
            _current_player_proc = None
