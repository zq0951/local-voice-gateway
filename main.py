import os
import sys
import time
import queue
import logging
import asyncio
import threading
import uvicorn
import httpx

import wave
import config
from utils.audio import calc_rms
from core.playback import GLOBAL_AUDIO_QUEUE, playback_worker, play_ding, get_is_playing
from core.vad import record_audio_until_silence, create_microphone_stream, is_enrolling_event
from core.speaker_verifier import SPEAKER_VERIFIER
from core.wakeword import WAKEWORD_DETECTOR
from core.stt import transcribe_audio, init_stt_engine
from core.tts import init_tts_engine, synthesize_and_enqueue, enqueue_tts, is_tts_active
from core.audio_device import AudioDeviceManager
from core.enrollment import VOICEPRINT_MANAGER
from api.server import app, broadcast_event

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LocalVoiceGateway")

_llm_client = None

def get_llm_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
    return _llm_client

async def query_agent_and_speak(user_text: str, speaker_name: str, emotion: str = None):
    """处理识别结果：事件已由 broadcast_event 派发至 WebSocket，供 DSH/外部 Agent 消费"""
    enable_local_echo = os.getenv("ENABLE_LOCAL_ECHO", "false").lower() == "true"
    if not config.AGENT_API_URL or config.AGENT_API_URL.lower() in ["", "none", "echo"]:
        if enable_local_echo:
            speaker_display = f"{speaker_name}，" if speaker_name and speaker_name not in ["unknown", "authorized_user"] else ""
            reply = f"{speaker_display}听到你说：{user_text}"
            logger.info(f"🔁 [本地复述模式]: {reply}")
            if getattr(config, "AUTO_SPEAK", True):
                enqueue_tts(reply)
            else:
                logger.info("🔇 [AUTO_SPEAK 关闭]: 跳过本地复述语音合成")
        else:
            logger.info(f"✨ 语音已广播至事件流，等待 DSH/Agent 回复: [{speaker_name}]: {user_text}")
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.AGENT_TOKEN}"
    }

    full_prompt = f"[说话人:{speaker_name}] {user_text}"
    if emotion and emotion != "NEUTRAL":
        full_prompt = f"(说话人情绪:{emotion}) " + full_prompt

    payload = {
        "model": "openclaw:main",
        "messages": [{"role": "user", "content": full_prompt}],
        "stream": False
    }

    try:
        client = get_llm_client()
        logger.info(f"🚀 发送给 Agent: {full_prompt}")
        resp = await client.post(f"{config.AGENT_API_URL}/v1/chat/completions", headers=headers, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
            logger.info(f"🤖 Agent 回复: {reply}")
            if getattr(config, "AUTO_SPEAK", True):
                enqueue_tts(reply)
            else:
                logger.info("🔇 [AUTO_SPEAK 关闭]: 跳过大模型回复语音合成")
        else:
            logger.error(f"Agent 请求失败: HTTP {resp.status_code}")
    except Exception as e:
        logger.error(f"调用 Agent 出错: {e}")

def run_dialog_session(loop, temp_wav_path, initial_speaker="authorized_user"):
    """持续对话交互窗口：会话保持 active_window 秒，期间免唤醒直接交流"""
    active_window = getattr(config, "WAKE_WORD_ACTIVE_WINDOW", 25)
    logger.info(f"✨ 进入持续对话窗口 (会话保持 {active_window}s，在此期间可免唤醒直接交流)...")

    # 1. 缓冲等待 ding 提示音播放完毕，确保录音不被自身音效干扰
    t_wait = time.time()
    while get_is_playing() and (time.time() - t_wait < 1.2):
        time.sleep(0.03)
    time.sleep(0.12) # 避开扬声器余震

    while True:
        if is_enrolling_event.is_set():
            break

        # 半双工保护：若扬声器正在准备或朗读回复，等待其播报完毕
        if config.AUDIO_DUPLEX_MODE == "half" and is_tts_active():
            while is_tts_active():
                time.sleep(0.05)
            time.sleep(0.2)

        # 2. 录制用户语音指令（等待开口超时为 active_window 秒）
        audio_path, _ = record_audio_until_silence(temp_wav_path, listen_timeout=active_window)
        if not audio_path:
            logger.info(f"💤 [{active_window}s 静默超时]: 未检测到新语音指令，退出持续对话，返回待机。")
            break

        # 3. 语音识别
        stt_res = loop.run_until_complete(transcribe_audio(audio_path))
        text = stt_res.get("text", "").strip()
        is_speech = stt_res.get("is_speech", True)
        if not text or not is_speech:
            logger.info("💭 [未识别到有效人声内容/非语音事件]，保持会话继续聆听...")
            continue

        # 4. 广播事件并交由 Agent 思考回复
        speaker_display = initial_speaker or "authorized_user"
        logger.info(f"🗣️ 用户输入: '{text}' (情绪={stt_res.get('emotion')})")
        loop.run_until_complete(broadcast_event({
            "event": "speech_recognized",
            "speaker": speaker_display,
            "text": text,
            "emotion": stt_res.get("emotion")
        }))
        loop.run_until_complete(query_agent_and_speak(text, speaker_display, stt_res.get("emotion")))

def run_api_server():
    """在后台独立线程启动 FastAPI 服务"""
    uvicorn.run(app, host="0.0.0.0", port=config.GATEWAY_PORT, log_level="warning")

def main_voice_loop():
    """主事件循环：双模状态机 (关键词唤醒 vs 声纹被动常开 vs 混合模式)"""
    logger.info(f"🚀 Local Voice Gateway 启动中...")
    
    # 0. 自动探测并自愈音频设备配置
    AudioDeviceManager.auto_configure_alsa(force=False)

    # 0.5 清理进程重启丢下的声纹录入临时孤儿目录
    VOICEPRINT_MANAGER.clean_orphaned_dirs()

    logger.info(f"📌 当前触发模式: {config.TRIGGER_MODE}")
    logger.info(f"🎙️ 双工模式: {config.AUDIO_DUPLEX_MODE}")
    
    # 预载声纹与唤醒模型
    SPEAKER_VERIFIER.reload_profiles()
    WAKEWORD_DETECTOR.load_model()
    init_tts_engine()
    init_stt_engine()

    available_models = WAKEWORD_DETECTOR.get_available_models()
    logger.info(f"🎯 唤醒词生效: [{config.WAKE_WORD_MODEL}] (阈值={config.WAKE_WORD_THRESHOLD})")
    logger.info(f"💡 本地支持唤醒词: {available_models} (支持命令行 --wakeword 或 API 实时切换)")

    temp_wav_path = os.path.join(config.BASE_DIR, "temp_current_voice.wav")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    last_mode = None
    was_playing = False

    while True:
        try:
            if is_enrolling_event.is_set():
                time.sleep(0.3)
                continue

            # 半双工防打架：扬声器播报或 TTS 合成期间优雅等待，杜绝抢占声卡导致发声失败
            if config.AUDIO_DUPLEX_MODE == "half" and is_tts_active():
                if not was_playing:
                    logger.info("⏸️ [半双工]: 扬声器正在准备/播报回复，暂停麦克风录制...")
                    was_playing = True
                time.sleep(0.2)
                continue

            if was_playing:
                logger.info("▶️ [半双工]: 扬声器播报完毕，恢复麦克风监听")
                was_playing = False
                time.sleep(0.3) # 避开扬声器尾音余震

            configured_mode = config.TRIGGER_MODE
            has_profiles = bool(SPEAKER_VERIFIER.centroids)

            # 无声纹样本时，纯声纹被动常开自动安全降级为关键词唤醒模式
            effective_mode = configured_mode
            if configured_mode == "voiceprint_passive" and not has_profiles:
                effective_mode = "wake_word"
                if last_mode != "fallback_wake_word":
                    wake_hint = "Hey Jarvis" if config.WAKE_WORD_MODEL == "hey_jarvis" else config.WAKE_WORD_MODEL
                    logger.warning(f"⚠️ [自动保护降级]: 当前声纹库为空，纯声纹被动常开已自动切换为【关键词唤醒模式】（请清晰说英文唤醒词 '{wake_hint}'，或在面板录入声纹）")
                    last_mode = "fallback_wake_word"

            # -------------------------------------------------------------
            # 模式 1: 关键词唤醒模式 (配置为 wake_word 或无声纹时安全降级)
            # -------------------------------------------------------------
            if effective_mode == "wake_word":
                if last_mode != effective_mode and last_mode != "fallback_wake_word":
                    logger.info(f"🎧 [关键词唤醒模式]: 正在监听唤醒词 '{config.WAKE_WORD_MODEL}' ...")
                    last_mode = effective_mode

                proc = create_microphone_stream()
                chunk_bytes = 1280 * config.SAMPLE_WIDTH # 80ms chunk for openwakeword
                triggered = False

                try:
                    while True:
                        if is_enrolling_event.is_set():
                            time.sleep(0.2)
                            break

                        # 动态感知模式切换与声纹入库，若已不再是 wake_word 模式，立即退出内部循环切换
                        curr_cfg = config.TRIGGER_MODE
                        curr_has_profiles = bool(SPEAKER_VERIFIER.centroids)
                        curr_eff = "wake_word" if (curr_cfg == "voiceprint_passive" and not curr_has_profiles) else curr_cfg
                        if curr_eff != "wake_word":
                            logger.info(f"🔄 检测到触发模式已热切换为 [{curr_eff}]，唤醒监听器退出并切换")
                            break

                        if config.AUDIO_DUPLEX_MODE == "half" and is_tts_active():
                            break

                        raw = proc.stdout.read(chunk_bytes)
                        if not raw or len(raw) < chunk_bytes:
                            break
                        
                        is_hit, model_name, score = WAKEWORD_DETECTOR.process_chunk(raw)
                        if is_hit:
                            triggered = True
                            logger.info(f"🎯 唤醒词命中! ({model_name}, 置信度={score:.2f})")
                            play_ding() # 毫秒级反馈
                            loop.run_until_complete(broadcast_event({
                                "event": "wake_word_detected", 
                                "model": model_name, 
                                "score": score
                            }))
                            break
                finally:
                    try:
                        proc.terminate()
                        proc.wait(timeout=0.5)
                    except:
                        try: proc.kill()
                        except: pass

                if triggered:
                    run_dialog_session(loop, temp_wav_path, initial_speaker="authorized_user")

            # -------------------------------------------------------------
            # 模式 2: 声纹被动常开模式 (单兵书房沉浸首选，需有已注册声纹)
            # -------------------------------------------------------------
            elif effective_mode == "voiceprint_passive":
                if last_mode != effective_mode:
                    logger.info("🎧 [声纹被动模式]: 麦克风常开，等待主人说话...")
                    last_mode = effective_mode

                audio_path, rms = record_audio_until_silence(temp_wav_path)
                if not audio_path:
                    time.sleep(0.1)
                    continue

                # 提取声纹特征并与库中比对
                is_matched, speaker, score = SPEAKER_VERIFIER.verify(audio_path)
                if not is_matched:
                    logger.info(f"🚫 [声纹过滤]: 未匹配到已注册主人声纹 (最佳: {speaker}, 分数: {score:.2f} < 门限)，静默丢弃。")
                    continue

                # 声纹匹配成功，直写 ding 提示音消除等待感
                play_ding()
                logger.info(f"✨ 主人身份确认: {speaker} (置信度: {score:.2f})")

                stt_res = loop.run_until_complete(transcribe_audio(audio_path))
                text = stt_res.get("text", "").strip()
                is_speech = stt_res.get("is_speech", True)
                if text and is_speech:
                    loop.run_until_complete(broadcast_event({
                        "event": "speech_recognized",
                        "speaker": speaker,
                        "text": text,
                        "emotion": stt_res.get("emotion")
                    }))
                    loop.run_until_complete(query_agent_and_speak(text, speaker, stt_res.get("emotion")))

            # -------------------------------------------------------------
            # 模式 3: 混合模式 (二选一触发: 唤醒词命中 OR 主人声纹免唤醒直接触发)
            # -------------------------------------------------------------
            else: # hybrid
                if last_mode != effective_mode:
                    logger.info(f"🎧 [混合双模 (二选一触发)]: 正在并发监听【唤醒词 '{config.WAKE_WORD_MODEL}'】与【主人声纹免唤醒】...")
                    last_mode = effective_mode

                proc = create_microphone_stream()
                chunk_bytes = 1280 * config.SAMPLE_WIDTH # 80ms chunk
                trigger_type = None
                matched_speaker = None

                vad_frames = []
                pre_buffer = []
                is_speaking = False
                silence_start = None
                speech_start = None
                noise_floor = 0.0
                alpha = 0.05
                calibrated_threshold = config.MIN_ENERGY_THRESHOLD

                try:
                    while True:
                        if is_enrolling_event.is_set():
                            time.sleep(0.2)
                            break

                        if config.TRIGGER_MODE != "hybrid":
                            break

                        if config.AUDIO_DUPLEX_MODE == "half" and is_tts_active():
                            break

                        raw = proc.stdout.read(chunk_bytes)
                        if not raw or len(raw) < chunk_bytes:
                            break

                        # -------------------------------------------------
                        # 触发路径 1: 关键词唤醒检测 (OpenWakeWord)
                        # -------------------------------------------------
                        is_hit, model_name, score = WAKEWORD_DETECTOR.process_chunk(raw)
                        if is_hit:
                            trigger_type = "wakeword"
                            logger.info(f"🎯 [混合模式 - 路径1] 唤醒词命中! ({model_name}, 置信度={score:.2f})")
                            play_ding()
                            loop.run_until_complete(broadcast_event({
                                "event": "wake_word_detected", 
                                "model": model_name, 
                                "score": score
                            }))
                            break

                        # -------------------------------------------------
                        # 触发路径 2: 主人声纹免唤醒检测 (VAD 人声切片 + CAM++ 声纹)
                        # -------------------------------------------------
                        rms = calc_rms(raw, config.SAMPLE_WIDTH)
                        if noise_floor == 0:
                            noise_floor = rms
                        else:
                            noise_floor = (1 - alpha) * noise_floor + alpha * rms
                        calibrated_threshold = max(config.MIN_ENERGY_THRESHOLD, noise_floor * config.VAD_MULTIPLIER)

                        now = time.time()
                        if not is_speaking:
                            pre_buffer.append(raw)
                            if len(pre_buffer) > 4: # 保留 320ms 前置静音/辅音缓冲
                                pre_buffer.pop(0)

                            if rms > calibrated_threshold and rms > config.MIN_ENERGY_THRESHOLD:
                                is_speaking = True
                                speech_start = now
                                silence_start = None
                                vad_frames.extend(pre_buffer)
                                pre_buffer = []
                        else:
                            vad_frames.append(raw)
                            if rms < (calibrated_threshold * 0.8):
                                if silence_start is None:
                                    silence_start = now
                                elif now - silence_start > config.SILENCE_TIMEOUT:
                                    # 用户整句话讲完了！
                                    total_duration = len(vad_frames) * 0.08
                                    if total_duration >= 0.8: # 至少 0.8s 有效音频段
                                        with wave.open(temp_wav_path, 'wb') as wf:
                                            wf.setnchannels(config.CHANNELS)
                                            wf.setsampwidth(config.SAMPLE_WIDTH)
                                            wf.setframerate(config.SAMPLE_RATE)
                                            wf.writeframes(b''.join(vad_frames))

                                        is_matched, speaker, vp_score = SPEAKER_VERIFIER.verify(temp_wav_path)
                                        if is_matched:
                                            trigger_type = "voiceprint"
                                            matched_speaker = speaker
                                            logger.info(f"✨ [混合模式 - 路径2] 主人声纹免唤醒直接触发! ({speaker}, 置信度={vp_score:.2f})")
                                            play_ding()
                                            break
                                        else:
                                            logger.debug(f"混合模式: 说话人非主人 (最佳: {speaker}, 分数={vp_score:.2f})，继续监听")
                                    # 重置 VAD 状态
                                    is_speaking = False
                                    vad_frames = []
                                    silence_start = None
                            else:
                                silence_start = None

                            if speech_start and (now - speech_start > config.MAX_RECORD_SECONDS):
                                is_speaking = False
                                vad_frames = []
                                silence_start = None

                finally:
                    try:
                        proc.terminate()
                        proc.wait(timeout=0.5)
                    except:
                        try: proc.kill()
                        except: pass

                # ---------------------------------------------------------
                # 触发后交互流 (二选一)
                # ---------------------------------------------------------
                if trigger_type == "wakeword":
                    # 路径 1: 喊了唤醒词 -> 无论声纹直接进入持续对话交互
                    run_dialog_session(loop, temp_wav_path, initial_speaker="authorized_user")

                elif trigger_type == "voiceprint":
                    # 路径 2: 声纹直接触发 -> 识别刚才录下的整句话并直接回复
                    stt_res = loop.run_until_complete(transcribe_audio(temp_wav_path))
                    text = stt_res.get("text", "").strip()
                    is_speech = stt_res.get("is_speech", True)
                    if text and is_speech:
                        logger.info(f"🗣️ [主人免唤醒指令]: '{text}' (说话人={matched_speaker})")
                        loop.run_until_complete(broadcast_event({
                            "event": "speech_recognized",
                            "speaker": matched_speaker,
                            "text": text,
                            "emotion": stt_res.get("emotion")
                        }))
                        loop.run_until_complete(query_agent_and_speak(text, matched_speaker, stt_res.get("emotion")))
                        # 声纹免唤醒一问一答完成后直接回归待机，不进入 25s 静默超时

        except KeyboardInterrupt:
            logger.info("👋 收到退出信号")
            break
        except Exception as e:
            logger.error(f"主监听循环异常: {e}")
            time.sleep(1.0)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local Voice Gateway - 物理语音交互网关")
    parser.add_argument("-w", "--wakeword", type=str, default=None,
                        help="配置唤醒词 (支持单个或逗号分隔，如: hey_jarvis 或 alexa 或 hey_jarvis,alexa)")
    parser.add_argument("-t", "--threshold", type=float, default=None,
                        help="唤醒词灵敏度阈值 (0.0~1.0, 默认 0.5)")
    parser.add_argument("-m", "--mode", type=str, choices=["hybrid", "wake_word", "voiceprint_passive"], default=None,
                        help="触发模式: hybrid(混合), wake_word(仅关键词), voiceprint_passive(仅声纹常开)")
    parser.add_argument("--list-wakewords", action="store_true",
                        help="列出本地所有可用唤醒词模型并退出")
    args, _ = parser.parse_known_args()

    if args.list_wakewords:
        available = WAKEWORD_DETECTOR.get_available_models()
        print("\n📦 [Local Voice Gateway] 本地可用的唤醒词模型列表:")
        for m in available:
            print(f"  - {m}")
        print("\n💡 切换示例: python main.py --wakeword alexa\n")
        sys.exit(0)

    # 动态覆盖命令行传入的配置
    if args.wakeword:
        config.WAKE_WORD_MODEL = args.wakeword
        WAKEWORD_DETECTOR.model_names = [w.strip() for w in args.wakeword.split(",") if w.strip()]
    if args.threshold is not None:
        config.WAKE_WORD_THRESHOLD = args.threshold
        WAKEWORD_DETECTOR.threshold = args.threshold
    if args.mode:
        config.TRIGGER_MODE = args.mode

    # 1. 启动音频播放工作线程
    playback_th = threading.Thread(target=playback_worker, args=(GLOBAL_AUDIO_QUEUE,), daemon=True)
    playback_th.start()

    # 2. 启动 FastAPI 服务线程
    api_th = threading.Thread(target=run_api_server, daemon=True)
    api_th.start()
    logger.info(f"🌐 REST API 与 WebSocket 事件流已启动: http://0.0.0.0:{config.GATEWAY_PORT}")

    # 3. 运行主语音识别与状态机循环
    main_voice_loop()

if __name__ == "__main__":
    main()
