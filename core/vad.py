import subprocess
import time
import wave
import logging
import os
from config import (
    SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH, CHUNK_SIZE, 
    ENERGY_THRESHOLD, MIN_ENERGY_THRESHOLD, VAD_MULTIPLIER, 
    SILENCE_TIMEOUT, MAX_RECORD_SECONDS, PRE_SPEECH_BUFFER, 
    CHUNK_DURATION_MS, AUDIO_DUPLEX_MODE
)
from utils.audio import calc_rms
from core.playback import get_is_playing

logger = logging.getLogger("LocalVoiceGateway")

_global_noise_floor = 0.0
_global_calibrated_threshold = 0.0
is_enrolling_event = __import__('threading').Event()

def cleanup_arecord():
    """安全回收遗留的 arecord 麦克风录音进程"""
    try:
        subprocess.run(["pkill", "-9", "-f", "arecord -q -f S16_LE"], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.debug(f"pkill arecord 清理异常: {e}")

def create_microphone_stream():
    """创建一个连续读取麦克风 16kHz PCM 的子进程管道"""
    cleanup_arecord()
    proc = subprocess.Popen(
        ["arecord", "-q", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", str(CHANNELS), "-t", "raw"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    return proc

def record_audio_until_silence(output_filename="temp_recorded.wav", allow_enrolling=False, listen_timeout=None):
    """
    使用自适应动态阈值 VAD 录音直到静音
    listen_timeout: 尚未开始说话前的最大等待时长(秒)，超时返回 None
    返回: (output_path, trigger_rms)
    """
    cleanup_arecord()
    
    proc = create_microphone_stream()
    
    frames = []
    pre_buffer = []
    is_speaking = False
    trigger_rms = 0
    silence_start = None
    record_start = None
    wait_start = None
    bytes_per_chunk = CHUNK_SIZE * SAMPLE_WIDTH
    
    global _global_noise_floor, _global_calibrated_threshold
    
    noise_floor = _global_noise_floor
    alpha = 0.05
    calibrated_threshold = _global_calibrated_threshold if _global_calibrated_threshold > 0 else ENERGY_THRESHOLD

    try:
        while True:
            if not allow_enrolling and is_enrolling_event.is_set():
                logger.info("⏸️ [麦克风互斥]: 正在录入声纹，主循环让出声卡")
                break

            now = time.time()
            if not is_speaking:
                if listen_timeout is not None:
                    if wait_start is None:
                        wait_start = now
                    elif now - wait_start > listen_timeout:
                        break

                if AUDIO_DUPLEX_MODE == "half" and get_is_playing():
                    time.sleep(0.04)
                    continue
            else:
                if AUDIO_DUPLEX_MODE == "half" and get_is_playing():
                    # 已在说话中途若被扬声器抢断则截断保存
                    break

            data = proc.stdout.read(bytes_per_chunk)
            if not data or len(data) < bytes_per_chunk:
                break
            rms = calc_rms(data, SAMPLE_WIDTH)
            
            current_multiplier = VAD_MULTIPLIER
            if get_is_playing():
                current_multiplier = VAD_MULTIPLIER * 2.5 # 播报时抑制回声
            
            if not is_speaking:
                if noise_floor == 0:
                    noise_floor = rms
                else:
                    noise_floor = (1 - alpha) * noise_floor + alpha * rms
                
                calibrated_threshold = max(MIN_ENERGY_THRESHOLD, noise_floor * current_multiplier)
                
                pre_buffer.append(data)
                if len(pre_buffer) > PRE_SPEECH_BUFFER:
                    pre_buffer.pop(0)
                
                if rms > calibrated_threshold and rms > MIN_ENERGY_THRESHOLD:
                    if AUDIO_DUPLEX_MODE == "half" and get_is_playing():
                        continue

                    is_speaking = True
                    trigger_rms = rms
                    record_start = time.time()
                    silence_start = None
                    frames.extend(pre_buffer)
                    pre_buffer = []
                    logger.info(f"🎤 [VAD] 检测到人声 (RMS={int(rms)} > 阈值={int(calibrated_threshold)})，开始录制!")
            else:
                frames.append(data)
                if rms < (calibrated_threshold * 0.8):
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        logger.info(f"🛑 [VAD] 静默持续 {SILENCE_TIMEOUT}s，录音结束 (共 {len(frames)} 帧)")
                        break
                else:
                    silence_start = None
                
                if time.time() - record_start > MAX_RECORD_SECONDS:
                    logger.info(f"⏰ [VAD] 达到最大录音时长 {MAX_RECORD_SECONDS}s，强制截断")
                    break
    finally:
        _global_noise_floor = noise_floor
        _global_calibrated_threshold = calibrated_threshold
        
        was_running = proc.poll() is None
        if was_running:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=0.5)
                except Exception:
                    pass

    if not frames:
        return None, 0

    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

    duration = len(frames) * CHUNK_DURATION_MS / 1000
    logger.info(f"✅ 录音完成并保存: {output_filename} ({duration:.1f}s), 触发 RMS: {int(trigger_rms)}")
    return output_filename, trigger_rms
