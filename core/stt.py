import os
import logging
import asyncio
import threading
import config
from utils.text import simple_t2s
from utils.funasr_parser import parse_funasr_tags

logger = logging.getLogger("LocalVoiceGateway")

MIN_CUDA_FREE_BYTES = 1_500_000_000


class LocalSenseVoiceRecognizer:
    """
    FunASR / SenseVoiceSmall 纯本地直接推理单例引擎
    完全基于 Python 进程内运行，跨平台支持 Windows、macOS 与 Linux，彻底免除 Docker 与外部网络端口依赖
    """
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.device = "cpu"
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_device() -> str:
        requested = str(getattr(config, "STT_DEVICE", "auto")).strip().lower()
        if requested == "cpu":
            return "cpu"
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
            if requested in ("cuda", "gpu"):
                return "cuda:0"
            if requested.startswith("cuda:"):
                return requested
            # auto 模式: 检查显存余量，避免与本地 LLM 争抢显存触发 CUDA OOM
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                if free_bytes < MIN_CUDA_FREE_BYTES:
                    logger.info(
                        f"ℹ️ 当前空闲显存仅 {free_bytes / (1024 ** 3):.2f} GiB (< 1.5 GiB)，STT 自动使用 CPU 推理"
                    )
                    return "cpu"
            except Exception:
                pass
            return "cuda:0"
        except Exception:
            return "cpu"

    def init_model(self) -> bool:
        if self.is_initialized:
            return True

        with self._lock:
            if self.is_initialized:
                return True

            try:
                from funasr import AutoModel
            except ImportError:
                logger.error("❌ 未检测到 funasr 库，请先执行: pip install funasr")
                return False

            model_base = config.FUNASR_MODEL_DIR
            sensevoice_dir = os.path.join(model_base, "SenseVoiceSmall")

            # 检查核心模型目录是否存在
            if not os.path.exists(sensevoice_dir):
                logger.error(f"❌ 未找到本地 SenseVoiceSmall 模型目录: {sensevoice_dir}")
                return False

            target_device = self._resolve_device()
            devices_to_try = [target_device] if target_device == "cpu" else [target_device, "cpu"]

            for dev in devices_to_try:
                logger.info(f"⏳ 正在加载 FunASR 本地直接推理引擎 (设备: {dev}, 模型: {sensevoice_dir}) ...")
                try:
                    self.model = AutoModel(
                        model=sensevoice_dir,
                        disable_update=True,
                        device=dev,
                        log_level="ERROR",
                    )
                    self.device = dev
                    self.is_initialized = True
                    logger.info("✅ FunASR 本地直接推理引擎加载完成 (原生进程内推理，免 Docker/免网络端口)")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ FunASR 在设备 {dev} 上加载失败: {e}")
                    self.model = None
                    self.is_initialized = False

            logger.error("❌ 本地 FunASR 引擎加载失败 (所有候选设备均不可用)")
            return False

    def transcribe_file(self, audio_file_path: str, language: str = "auto", use_itn: bool = True) -> dict:
        """同步推理音频文件并解析富文本情绪标签"""
        if not self.is_initialized or self.model is None:
            logger.warning("STT 引擎尚未初始化成功，跳过识别")
            return {"text": "", "emotion": None, "lang": None, "event": None, "is_speech": False}

        lang_opt = (language or "auto").strip() or "auto"
        try:
            with self._lock:
                res = self.model.generate(
                    input=audio_file_path,
                    cache={},
                    language=lang_opt,
                    use_itn=bool(use_itn),
                    batch_size_s=60,
                )
            raw_text = ""
            if isinstance(res, list) and len(res) > 0:
                first_item = res[0]
                if isinstance(first_item, dict):
                    raw_text = first_item.get("text", "")
                elif isinstance(first_item, str):
                    raw_text = first_item
            elif isinstance(res, dict):
                raw_text = res.get("text", "")

            parsed = parse_funasr_tags(raw_text)
            parsed["text"] = simple_t2s(parsed["text"])
            return parsed
        except Exception as e:
            logger.error(f"本地 STT 推理执行失败: {e}")
            return {"text": "", "emotion": None, "lang": None, "event": None, "is_speech": False}


LOCAL_STT_ENGINE = LocalSenseVoiceRecognizer()


def init_stt_engine():
    """网关启动时统一预载 STT 引擎"""
    return LOCAL_STT_ENGINE.init_model()


async def transcribe_audio(audio_file_path: str, language: str = "auto", use_itn: bool = True) -> dict:
    """
    纯本地语音识别接口：在异步线程池中调用本地直接推理引擎，避免阻塞 asyncio 事件循环
    """
    if not LOCAL_STT_ENGINE.is_initialized:
        await asyncio.to_thread(LOCAL_STT_ENGINE.init_model)

    parsed = await asyncio.to_thread(LOCAL_STT_ENGINE.transcribe_file, audio_file_path, language, use_itn)
    logger.info(f"📝 [STT 本地直接识别]: '{parsed['text']}' (情绪={parsed['emotion']}, 语言={parsed['lang']})")
    return parsed
