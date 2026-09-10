import json
import ssl
import logging
import asyncio
import websockets
from config import STT_WS_URL
from utils.text import simple_t2s
from utils.funasr_parser import parse_funasr_tags

logger = logging.getLogger("LocalVoiceGateway")

async def transcribe_audio(audio_file_path: str):
    """
    通过 WebSocket 请求本地 SenseVoice / FunASR 服务
    返回: dict {"text": str, "emotion": str, "lang": str, "is_speech": bool}
    具备 wss 与 ws 协议自适应重试容错机制，无缝兼容自签名 SSL 与普通端口
    """
    candidate_urls = [STT_WS_URL]
    if STT_WS_URL.startswith("wss://"):
        candidate_urls.append(STT_WS_URL.replace("wss://", "ws://", 1))
    elif STT_WS_URL.startswith("ws://"):
        candidate_urls.append(STT_WS_URL.replace("ws://", "wss://", 1))

    last_error = None
    for url in candidate_urls:
        is_wss = url.startswith("wss://")
        ssl_context = None
        if is_wss:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        try:
            connect_kwargs = {"open_timeout": 3.0}
            if ssl_context:
                connect_kwargs["ssl"] = ssl_context
            async with websockets.connect(url, **connect_kwargs) as websocket:
                config = {
                    "mode": "offline",
                    "chunk_size": [5, 10, 5],
                    "chunk_interval": 10,
                    "wav_name": "stt_request",
                    "is_speaking": True,
                    "hotwords": ""
                }
                await websocket.send(json.dumps(config))

                with open(audio_file_path, "rb") as f:
                    audio_data = f.read()
                    await websocket.send(audio_data)

                await websocket.send(json.dumps({"is_speaking": False}))

                raw_text = ""
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    res_data = json.loads(response)
                    raw_text = res_data.get("text", "")
                except asyncio.TimeoutError:
                    logger.warning("FunASR 接收超时")

                parsed = parse_funasr_tags(raw_text)
                parsed["text"] = simple_t2s(parsed["text"])
                logger.info(f"📝 [STT 识别结果]: '{parsed['text']}' (情绪={parsed['emotion']}, 语言={parsed['lang']})")
                return parsed
        except Exception as e:
            last_error = e
            continue

    logger.error(f"连接 STT 服务失败 ({STT_WS_URL}): {last_error}")
    return {"text": "", "emotion": None, "lang": None, "is_speech": False}
