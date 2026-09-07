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
    """
    is_wss = STT_WS_URL.startswith("wss://")
    ssl_context = None
    if is_wss:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    try:
        connect_kwargs = {"open_timeout": 3.0}
        if ssl_context:
            connect_kwargs["ssl"] = ssl_context
        async with websockets.connect(STT_WS_URL, **connect_kwargs) as websocket:
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
        logger.error(f"连接 STT 服务失败 ({STT_WS_URL}): {e}")
        return {"text": "", "emotion": None, "lang": None, "is_speech": False}
