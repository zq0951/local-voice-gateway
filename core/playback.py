import os
import time
import logging
import queue
import subprocess
import threading
from config import HW_SAMPLE_RATE, PLAYBACK_CHANNELS, DING_PCM_PATH

logger = logging.getLogger("LocalVoiceGateway")

GLOBAL_AUDIO_QUEUE = queue.Queue()
_is_playing = False
_playing_lock = threading.Lock()

def get_is_playing():
    global _is_playing
    return _is_playing

def set_is_playing(status: bool):
    global _is_playing
    with _playing_lock:
        _is_playing = status

def stop_playback():
    """清空音频队列并立即终止当前 aplay 播报 (闭嘴功能)"""
    while not GLOBAL_AUDIO_QUEUE.empty():
        try:
            GLOBAL_AUDIO_QUEUE.get_nowait()
            GLOBAL_AUDIO_QUEUE.task_done()
        except Exception:
            break
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

def playback_worker(audio_queue: queue.Queue):
    """音频播放工作线程：从队列拉取原始双声道 PCM 字节并直写 aplay 管道"""
    aplay_proc = None

    try:
        while True:
            try:
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                if aplay_proc:
                    set_is_playing(False)
                    try:
                        aplay_proc.stdin.close()
                        aplay_proc.wait(timeout=1)
                    except:
                        try:
                            aplay_proc.kill()
                        except:
                            pass
                    aplay_proc = None
                continue

            set_is_playing(True)

            if chunk is None:
                if aplay_proc:
                    try:
                        aplay_proc.stdin.close()
                        aplay_proc.wait(timeout=1)
                    except:
                        aplay_proc.kill()
                    aplay_proc = None
                set_is_playing(False)
                audio_queue.task_done()
                continue

            if aplay_proc is None:
                try:
                    aplay_proc = subprocess.Popen(
                        ["aplay", "-D", "default", "-f", "S16_LE", "-r", str(HW_SAMPLE_RATE), 
                         "-c", str(PLAYBACK_CHANNELS), "-t", "raw"],
                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    # 硬件 DAC 建立缓冲：先注入 150ms 静音防止开头音节被吃
                    silence_padding = b'\x00' * int(HW_SAMPLE_RATE * PLAYBACK_CHANNELS * 2 * 0.15)
                    aplay_proc.stdin.write(silence_padding)
                    aplay_proc.stdin.flush()
                except Exception as e:
                    logger.error(f"启动 aplay 播放进程失败: {e}")
                    aplay_proc = None
                    audio_queue.task_done()
                    continue

            try:
                aplay_proc.stdin.write(chunk)
                aplay_proc.stdin.flush()
            except Exception as e:
                logger.error(f"aplay 写入音频失败: {e}")
                try:
                    aplay_proc.kill()
                except:
                    pass
                aplay_proc = None

            audio_queue.task_done()
    finally:
        set_is_playing(False)
        if aplay_proc:
            try:
                aplay_proc.kill()
            except:
                pass
