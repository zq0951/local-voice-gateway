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
            GLOBAL_AUDIO_QUEUE.put(None)
        except Exception as e:
            logger.error(f"播放 ding 失败: {e}")

def _close_player(player):
    """安全关闭音频播放器"""
    if player is None:
        return
    try:
        if hasattr(player, "stdin") and player.stdin:
            try:
                player.stdin.close()
            except Exception:
                pass
        if hasattr(player, "wait"):
            try:
                player.wait(timeout=0.3)
            except Exception:
                if hasattr(player, "kill"):
                    try:
                        player.kill()
                    except Exception:
                        pass
        elif hasattr(player, "close"):
            player.close()
    except Exception:
        pass

def create_audio_player():
    """创建音频输出流进程或对象 (优先物理硬件直连 plughw:X,Y -> ALSA default -> PyAudio 兜底)"""
    if os.name != "nt" and shutil.which("aplay"):
        candidate_devices = []

        # 1. 动态探测真实物理播放声卡并加入候选列表 (纯数字编号 plughw:X,Y 最优先，完全不依赖 /proc)
        try:
            from core.audio_device import AudioDeviceManager
            p_devs = AudioDeviceManager.get_playback_devices()
            best = AudioDeviceManager.select_best_device(p_devs)
            if best:
                candidate_devices.append(f"plughw:{best['card_num']},{best['device_num']}")
                candidate_devices.append(f"plughw:CARD={best['card_id']},DEV={best['device_num']}")
                candidate_devices.append(f"hw:{best['card_num']},{best['device_num']}")
        except Exception as e:
            logger.debug(f"探测物理声卡列表异常: {e}")

        # 2. 扫描 /dev/snd/ 目录下的所有硬件播放节点 (作为双重保险)
        if os.path.exists("/dev/snd"):
            try:
                import glob
                import re
                for p in sorted(glob.glob("/dev/snd/pcmC*D*p"), reverse=True):
                    m = re.search(r"pcmC(\d+)D(\d+)p", p)
                    if m:
                        target = f"plughw:{m.group(1)},{m.group(2)}"
                        if target not in candidate_devices:
                            candidate_devices.append(target)
            except Exception:
                pass

        # 3. 兜底加入 ALSA default 虚拟设备
        if "default" not in candidate_devices:
            candidate_devices.append("default")

        for dev_name in candidate_devices:
            try:
                proc = subprocess.Popen(
                    ["aplay", "-D", dev_name, "-f", "S16_LE", "-r", str(HW_SAMPLE_RATE), 
                     "-c", str(PLAYBACK_CHANNELS), "-t", "raw", "-q"],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
                time.sleep(0.05)
                if proc.poll() is None:
                    logger.info(f"🔊 已成功连接音频输出设备: aplay -D {dev_name}")
                    return proc
                err = proc.stderr.read().decode("utf-8", errors="ignore")
                logger.warning(f"⚠️ aplay 设备 '{dev_name}' 启动异常 ({err.strip()})，尝试下一候选...")
            except Exception as e:
                logger.debug(f"尝试 aplay 设备 {dev_name} 失败: {e}")

        # 4. 所有 aplay 设备均不可用时，平滑回退至 PyAudio
        logger.warning("⚠️ 所有 aplay 设备均不可用，回退至系统底层 PyAudio 播放器")
        try:
            return PyAudioPlayer(HW_SAMPLE_RATE, PLAYBACK_CHANNELS)
        except Exception as pe:
            logger.error(f"❌ PyAudio 播放器初始化亦失败: {pe}")
            return None
    else:
        try:
            return PyAudioPlayer(HW_SAMPLE_RATE, PLAYBACK_CHANNELS)
        except Exception as pe:
            logger.error(f"❌ PyAudio 播放器初始化失败: {pe}")
            return None

def playback_worker(audio_queue: queue.Queue):
    """音频播放工作线程：从队列拉取原始双声道 PCM 字节并直写播放管道，支持平滑播放与即时打断"""
    global _current_player_proc
    player_proc = None

    bytes_per_sec = HW_SAMPLE_RATE * PLAYBACK_CHANNELS * 2  # 16000 * 2 * 2 = 64000 字节/秒

    while True:
        try:
            chunk = audio_queue.get(timeout=0.2)
        except queue.Empty:
            if player_proc:
                _close_player(player_proc)
                with _player_proc_lock:
                    if _current_player_proc is player_proc:
                        _current_player_proc = None
                player_proc = None
                set_is_playing(False)
            continue

        if chunk is None:
            if player_proc:
                _close_player(player_proc)
                with _player_proc_lock:
                    if _current_player_proc is player_proc:
                        _current_player_proc = None
                player_proc = None
            set_is_playing(False)
            audio_queue.task_done()
            continue

        chunk_len = len(chunk)
        if chunk_len == 0:
            audio_queue.task_done()
            continue

        total_audio_sec = chunk_len / bytes_per_sec
        set_is_playing(True)
        logger.info(f"🔊 [扬声器开始发声]: 时长 {total_audio_sec:.1f}s, 写入播放通道...")

        just_created = False
        if player_proc is None or (hasattr(player_proc, "poll") and player_proc.poll() is not None):
            player_proc = create_audio_player()
            with _player_proc_lock:
                _current_player_proc = player_proc

            if player_proc is None:
                logger.error("❌ 无法创建音频播放器，丢弃该段音频")
                set_is_playing(False)
                audio_queue.task_done()
                continue

            # 硬件 DAC 建立缓冲：首段先注入 100ms 静音防止开头音节被硬件吃掉
            silence_padding = b'\x00' * int(bytes_per_sec * 0.10)
            try:
                player_proc.stdin.write(silence_padding)
                player_proc.stdin.flush()
                just_created = True
            except Exception as e:
                logger.debug(f"写入初始静音缓冲: {e}")

        # 分片流式写入播放器 (每片 80ms = 5120 字节)
        slice_size = int(bytes_per_sec * 0.08)
        offset = 0
        t_start = time.monotonic()
        interrupted = False

        try:
            while offset < chunk_len:
                with _player_proc_lock:
                    if _current_player_proc is not player_proc or player_proc is None:
                        interrupted = True
                        break

                end_pos = min(offset + slice_size, chunk_len)
                sub = chunk[offset:end_pos]
                player_proc.stdin.write(sub)
                player_proc.stdin.flush()
                offset = end_pos

                # 动态控制写入节奏：让管道预缓冲保持在 0.8s~1.2s 充裕区间，彻底杜绝 Docker 调度抖动引发的欠载破音 (Buffer Underrun)
                written_sec = offset / bytes_per_sec
                elapsed_sec = time.monotonic() - t_start
                lead_sec = written_sec - elapsed_sec
                if lead_sec > 1.2:
                    time.sleep(lead_sec - 0.8)
                else:
                    time.sleep(0.005)

            # 数据全部注入管道后，等待硬件自然播放完毕剩余缓冲
            if not interrupted:
                # 总播放等待时长需计入: 音频本身时长 + 首次打开设备的前置静音补偿(100ms) + 声卡硬件缓冲区排空余量(80ms)
                hardware_drain_margin = 0.08 + (0.10 if just_created else 0.0)
                total_wait_sec = (chunk_len / bytes_per_sec) + hardware_drain_margin
                while True:
                    with _player_proc_lock:
                        if _current_player_proc is not player_proc or player_proc is None:
                            interrupted = True
                            break
                    elapsed_sec = time.monotonic() - t_start
                    if elapsed_sec >= total_wait_sec:
                        logger.info("✅ [扬声器播放完毕]: 音频已完整播放")
                        break
                    time.sleep(min(0.05, max(0.01, total_wait_sec - elapsed_sec)))

        except (BrokenPipeError, OSError) as e:
            err_details = ""
            if player_proc and hasattr(player_proc, "stderr") and player_proc.stderr:
                try:
                    err_details = player_proc.stderr.read().decode("utf-8", errors="ignore").strip()
                except Exception:
                    pass
            logger.error(f"❌ 播放器音频通道异常关闭: {e} {f'(底层报错: {err_details})' if err_details else ''}")
            _close_player(player_proc)
            player_proc = None
            with _player_proc_lock:
                if _current_player_proc is player_proc:
                    _current_player_proc = None
        except Exception as e:
            logger.error(f"❌ 播放器流式写入异常: {e}")
            _close_player(player_proc)
            player_proc = None
            with _player_proc_lock:
                if _current_player_proc is player_proc:
                    _current_player_proc = None

        audio_queue.task_done()
        if audio_queue.empty():
            set_is_playing(False)
