import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 自动加载本地 .env 文件 (无需外部第三方库)
env_file = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_file):
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# ==============================================================================
# 🎯 触发模式控制 (Trigger Modes)
# ==============================================================================
# 可选值:
# - "wake_word": 仅关键词唤醒模式 (适合打游戏连麦、看电影、多人嘈杂环境)
# - "voiceprint_passive": 纯声纹被动监听模式 (常开麦，仅认主人声纹，张嘴即触发)
# - "hybrid": 混合模式 (听到唤醒词后激活，同时识别声纹身份)
TRIGGER_MODE = os.getenv("TRIGGER_MODE", "voiceprint_passive")

# 自动朗读回复 (Auto-Speak) 开关 (可通过 API 或前端闭嘴开关动态控制)
AUTO_SPEAK = os.getenv("AUTO_SPEAK", "true").lower() in ("true", "1", "yes")

# 唤醒词配置 (OpenWakeWord)
WAKE_WORD_MODEL = os.getenv("WAKE_WORD_MODEL", "hey_jarvis")
WAKE_WORD_THRESHOLD = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))
WAKE_WORD_ACTIVE_WINDOW = int(os.getenv("WAKE_WORD_ACTIVE_WINDOW", "25")) # 唤醒后免唤醒持续对话窗口(秒)

# 声纹识别配置 (CAM++)
SPEAKER_MODEL_PATH = os.path.join(BASE_DIR, "models/campplus.onnx")
VOICE_PROFILES_DIR = os.path.join(BASE_DIR, "models/voice_profiles")
VOICEPRINT_THRESHOLD = float(os.getenv("VOICEPRINT_THRESHOLD", "0.50"))

# ==============================================================================
# 🎙️ 音频采集与播放配置
# ==============================================================================
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# 双工模式: "full" (全双工，支持打断) 或 "half" (半双工，播报时麦克风静音)
AUDIO_DUPLEX_MODE = os.getenv("AUDIO_DUPLEX_MODE", "half")

# 硬件播报采样率与声道 (默认 44100Hz 对齐 EMEET M1A 等绝大多数 USB 音箱物理 DAC，通过 torchaudio 高保真重采样)
HW_SAMPLE_RATE = int(os.getenv("HW_SAMPLE_RATE", "44100"))
PLAYBACK_CHANNELS = int(os.getenv("PLAYBACK_CHANNELS", "2"))

# 极速提示音路径
DING_PCM_PATH = os.path.join(BASE_DIR, "assets/ding.pcm")

# TTS 本地模型与代码路径 (完全收敛在项目内部 models/moss_tts，支持外部通过环境变量覆盖)
MOSS_MODEL_DIR = os.getenv("MOSS_MODEL_DIR", os.path.join(BASE_DIR, "models/moss_tts"))
MOSS_CACHE_DIR = os.getenv("MOSS_CACHE_DIR", os.path.join(MOSS_MODEL_DIR, "hf_cache"))

# ==============================================================================
# 🎛️ VAD (人声活动检测) 动态阈值配置
# ==============================================================================
ENERGY_THRESHOLD = int(os.getenv("ENERGY_THRESHOLD", "600"))
MIN_ENERGY_THRESHOLD = int(os.getenv("MIN_ENERGY_THRESHOLD", "450"))
VAD_MULTIPLIER = float(os.getenv("VAD_MULTIPLIER", "1.5"))
SILENCE_TIMEOUT = float(os.getenv("SILENCE_TIMEOUT", "1.5"))
MAX_RECORD_SECONDS = int(os.getenv("MAX_RECORD_SECONDS", "30"))
PRE_SPEECH_BUFFER = int(os.getenv("PRE_SPEECH_BUFFER", "10"))

# ==============================================================================
# 🌐 服务与后端连接端点
# ==============================================================================
# 本地网关监听地址与端口 (默认 127.0.0.1 安全收敛，若 Docker/远程调试可设 GATEWAY_HOST=0.0.0.0)
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8765"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# 本地 FunASR / SenseVoice 离线模型路径 (纯本地进程内推理，零网络开销与端口占用)
FUNASR_MODEL_DIR = os.getenv("FUNASR_MODEL_DIR", os.path.join(BASE_DIR, "models/funasr"))

# 后端 Agent 接口 (留空或设为 "echo" 时，网关进入纯本地回环复述模式，零配置开箱即测)
AGENT_API_URL = os.getenv("AGENT_API_URL", "echo")
AGENT_TOKEN = os.getenv("AGENT_TOKEN", "")
