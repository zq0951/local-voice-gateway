import os
import json
import asyncio
import logging
from typing import List, Set, Optional
from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import config
from core.playback import play_ding, stop_playback, set_playback_callback
from core.tts import synthesize_and_enqueue, enqueue_tts, stop_tts, set_tts_event_broadcaster
from core.speaker_verifier import SPEAKER_VERIFIER
from core.wakeword import WAKEWORD_DETECTOR
from core.enrollment import VOICEPRINT_MANAGER

logger = logging.getLogger("LocalVoiceGateway")

RUNTIME_CONFIG_FILE = os.path.join(config.BASE_DIR, "models", "runtime_config.json")

def save_runtime_config():
    try:
        data = {
            "trigger_mode": config.TRIGGER_MODE,
            "auto_speak": config.AUTO_SPEAK,
            "wake_word_model": config.WAKE_WORD_MODEL,
        }
        with open(RUNTIME_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存运行时配置失败: {e}")

def load_runtime_config():
    if os.path.exists(RUNTIME_CONFIG_FILE):
        try:
            with open(RUNTIME_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "trigger_mode" in data:
                config.TRIGGER_MODE = data["trigger_mode"]
            if "auto_speak" in data:
                config.AUTO_SPEAK = data["auto_speak"]
            if "wake_word_model" in data:
                config.WAKE_WORD_MODEL = data["wake_word_model"]
            logger.info(f"💾 已从本地存储恢复运行时配置: mode={config.TRIGGER_MODE}, auto_speak={config.AUTO_SPEAK}")
        except Exception as e:
            logger.warning(f"读取运行时配置失败: {e}")

load_runtime_config()

app = FastAPI(title="Local Voice Gateway", version="1.0.0")

cors_origins = [o.strip() for o in getattr(config, "CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 活跃 WebSocket 客户端集合，用于广播事件
connected_clients: Set[WebSocket] = set()
_main_async_loop: Optional[asyncio.AbstractEventLoop] = None

@app.on_event("startup")
async def _on_fastapi_startup():
    global _main_async_loop
    _main_async_loop = asyncio.get_running_loop()

async def broadcast_event(event_data: dict):
    """向所有连接的外部 Agent 或 Web UI 客户端广播事件"""
    if not connected_clients:
        return
    disconnected = set()
    for ws in list(connected_clients):
        try:
            await ws.send_json(event_data)
        except Exception:
            disconnected.add(ws)
    for ws in disconnected:
        if ws in connected_clients:
            connected_clients.remove(ws)

def sync_broadcast_event(event_data: dict):
    """从普通同步后台线程向全局 WebSocket 客户端安全广播事件"""
    global _main_async_loop
    if _main_async_loop is not None and _main_async_loop.is_running() and connected_clients:
        try:
            asyncio.run_coroutine_threadsafe(broadcast_event(event_data), _main_async_loop)
        except Exception as e:
            logger.debug(f"同步广播事件调度异常: {e}")

# 将同步广播接入 TTS 与扬声器状态机
set_tts_event_broadcaster(sync_broadcast_event)

def _on_playback_state_changed(is_playing: bool):
    if not is_playing:
        sync_broadcast_event({"event": "tts_idle"})
    else:
        sync_broadcast_event({"event": "playback_started"})

set_playback_callback(_on_playback_state_changed)

@app.get("/v1/system/status")
async def get_status():
    has_profiles = bool(SPEAKER_VERIFIER.centroids)
    effective_mode = "wake_word" if (config.TRIGGER_MODE == "voiceprint_passive" and not has_profiles) else config.TRIGGER_MODE
    return {
        "status": "running",
        "trigger_mode": config.TRIGGER_MODE,
        "effective_trigger_mode": effective_mode,
        "audio_duplex_mode": config.AUDIO_DUPLEX_MODE,
        "auto_speak": config.AUTO_SPEAK,
        "wake_word_model": config.WAKE_WORD_MODEL,
        "voiceprint_threshold": config.VOICEPRINT_THRESHOLD,
        "registered_speakers": list(SPEAKER_VERIFIER.centroids.keys())
    }

@app.post("/v1/system/mode")
async def set_trigger_mode(mode: str = Body(..., embed=True)):
    """
    动态切换触发模式:
    - wake_word (仅关键词唤醒，防打游戏误触)
    - voiceprint_passive (纯声纹被动常开)
    - hybrid (关键词激活 + 声纹识别)
    """
    valid_modes = ["wake_word", "voiceprint_passive", "hybrid"]
    if mode not in valid_modes:
        return JSONResponse(status_code=400, content={"error": f"无效模式，允许的值为: {valid_modes}"})
    
    config.TRIGGER_MODE = mode
    save_runtime_config()
    logger.info(f"🔄 触发模式已动态切换为: {mode}")
    await broadcast_event({"event": "mode_changed", "new_mode": mode})
    return {"status": "ok", "current_mode": mode}

@app.get("/v1/system/wakeword")
async def get_wakeword_info():
    """获取可用唤醒词列表及当前配置"""
    return {
        "current_model": config.WAKE_WORD_MODEL,
        "threshold": config.WAKE_WORD_THRESHOLD,
        "available_models": WAKEWORD_DETECTOR.get_available_models()
    }

@app.post("/v1/system/wakeword")
async def set_wakeword(model: str = Body(..., embed=True), threshold: float = Body(None, embed=True)):
    """动态热切换唤醒词模型（无需重启进程）"""
    if threshold is not None:
        config.WAKE_WORD_THRESHOLD = float(threshold)
        WAKEWORD_DETECTOR.threshold = float(threshold)
    config.WAKE_WORD_MODEL = model
    success = await asyncio.to_thread(WAKEWORD_DETECTOR.reload_models, model)
    save_runtime_config()
    logger.info(f"🔄 唤醒词已动态热重载为: {model} (成功={success})")
    await broadcast_event({"event": "wakeword_changed", "model": model, "threshold": config.WAKE_WORD_THRESHOLD})
    return {"status": "ok" if success else "failed", "current_model": model, "threshold": config.WAKE_WORD_THRESHOLD}

@app.post("/v1/system/autospeak")
async def set_autospeak(enabled: bool = Body(..., embed=True)):
    """动态开启/关闭大模型回复自动朗读 (闭嘴开关联动)"""
    config.AUTO_SPEAK = bool(enabled)
    save_runtime_config()
    logger.info(f"🔄 自动朗读回复已设为: {config.AUTO_SPEAK}")
    await broadcast_event({"event": "autospeak_changed", "enabled": config.AUTO_SPEAK})
    return {"status": "ok", "auto_speak": config.AUTO_SPEAK}

@app.post("/v1/audio/ding")
async def trigger_ding():
    """主动播放一次低延迟 ding 提示音"""
    play_ding()
    return {"status": "ok"}

@app.post("/v1/audio/speak")
async def speak_text(text: str = Body(..., embed=True)):
    """让本地音箱主动合成并播报一段文本 (丢入专用抢占式任务队列，耗时 < 1ms，绝不阻塞网络事件循环)"""
    clean = text.strip()
    if not clean:
        return JSONResponse(status_code=400, content={"error": "播报文本不能为空"})

    task_id = enqueue_tts(clean, interrupt_previous=True)
    return {"status": "queued", "task_id": task_id}

@app.post("/v1/audio/stop")
async def stop_audio():
    """立即打断当前合成与播报并清空任务队列 (闭嘴开关)"""
    stop_tts()
    return {"status": "ok"}

# ==============================================================================
# 👤 声纹识别与多步引导录入 API (B1, B2, B3, B4)
# ==============================================================================

@app.get("/v1/voiceprint/profiles")
async def get_voiceprint_profiles():
    """B2: 获取声纹库中已录入的所有说话人信息"""
    profiles = await asyncio.to_thread(VOICEPRINT_MANAGER.list_profiles)
    return {"status": "ok", "profiles": profiles}

@app.delete("/v1/voiceprint/{name}")
async def delete_voiceprint_profile(name: str):
    """B3: 删除指定说话人的声纹库并热重载 (丢入线程池，避免重新计算质心阻塞事件循环)"""
    deleted = await asyncio.to_thread(VOICEPRINT_MANAGER.delete_profile, name)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": f"说话人 '{name}' 不存在"})
    await broadcast_event({"event": "voice_profile_deleted", "speaker": name})
    return {"status": "ok", "deleted": name}

@app.post("/v1/voiceprint/enroll/start")
async def enroll_start(name: str = Body(..., embed=True), steps: int = Body(5, embed=True)):
    """B1 & B4: 开启多步短语引导录入会话，通知主循环挂起麦克风占用"""
    try:
        session_info = await asyncio.to_thread(VOICEPRINT_MANAGER.start_session, name, steps)
        await broadcast_event({"event": "enroll_started", "session": session_info})
        return {"status": "ok", "session": session_info}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/v1/voiceprint/enroll/record_step")
async def enroll_record_step(session_id: str = Body(..., embed=True)):
    """B1: 执行单步麦克风录制，提取声纹并推进引导步骤 (异步线程池执行，不卡死其他API)"""
    try:
        step_res = await asyncio.to_thread(VOICEPRINT_MANAGER.record_step, session_id)
        await broadcast_event({"event": "enroll_step_recorded", "step_result": step_res})
        return {"status": "ok" if step_res.get("success") else "failed", **step_res}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/v1/voiceprint/enroll/finish")
async def enroll_finish(session_id: str = Body(..., embed=True)):
    """B1: 完成录入，离群检验与剔除，正式存盘入库并热重载声纹库 (异步线程池执行)"""
    try:
        finish_res = await asyncio.to_thread(VOICEPRINT_MANAGER.finish_session, session_id)
        await broadcast_event({"event": "enroll_finished", "result": finish_res})
        return {"status": "ok", **finish_res}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/v1/voiceprint/enroll/abort")
async def enroll_abort(session_id: str = Body(..., embed=True)):
    """中止录入并释放声卡"""
    VOICEPRINT_MANAGER.abort_session(session_id)
    await broadcast_event({"event": "enroll_aborted", "session_id": session_id})
    return {"status": "ok"}

@app.websocket("/v1/events")
async def websocket_events(websocket: WebSocket):
    """事件流 WebSocket：供外部 Agent (如 DSH 插件) 实时订阅唤醒与识别事件"""
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"🔌 外部客户端已连接事件流 ({len(connected_clients)} 个客户端在线)")
    try:
        while True:
            # 保持长连接并接收客户端控制消息
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info("🔌 外部客户端已断开事件流")
    except Exception as e:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
