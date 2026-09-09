import os
import sys
import logging
import types
import torch
import torchaudio
import soundfile as sf
from config import HW_SAMPLE_RATE, PLAYBACK_CHANNELS, MOSS_MODEL_DIR, MOSS_CACHE_DIR
from utils.text import filter_symbols
from core.playback import GLOBAL_AUDIO_QUEUE

logger = logging.getLogger("LocalVoiceGateway")

# 强制使用 soundfile 替代 torchaudio.load / torchaudio.save，彻底消除 TorchCodec 依赖
def _patched_torchaudio_load(filepath, **kwargs):
    data, samplerate = sf.read(filepath)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    return torch.from_numpy(data.T).float(), samplerate

def _patched_torchaudio_save(filepath, src, sample_rate, **kwargs):
    data = src.detach().cpu().numpy()
    if data.ndim == 2:
        data = data.T
    sf.write(filepath, data, sample_rate)

torchaudio.load = _patched_torchaudio_load
torchaudio.save = _patched_torchaudio_save

MOSS_SERVICE = None

def apply_offline_patches(moss_path, hf_cache_path):
    """为 CPU 离线环境注入精准补丁，确保 MOSS-TTS 无网无显卡完美运行"""
    if moss_path and moss_path not in sys.path:
        sys.path.append(moss_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    if hf_cache_path:
        os.environ["HF_HOME"] = hf_cache_path

    try:
        import transformers
        # 1. 强制关闭 low_cpu_mem_usage，防止 torch-cpu 进入不支持的 meta tensor copy
        for auto_cls in [transformers.AutoModel, transformers.AutoModelForCausalLM]:
            if not hasattr(auto_cls, "_is_patched"):
                _orig_fp = auto_cls.from_pretrained
                @classmethod
                def _patched_fp(cls, *args, **kwargs):
                    kwargs["low_cpu_mem_usage"] = False
                    kwargs["device_map"] = None
                    if "device" in kwargs:
                        del kwargs["device"]
                    return _orig_fp.__func__(cls, *args, **kwargs)
                auto_cls.from_pretrained = _patched_fp
                auto_cls._is_patched = True

        # 2. 注入 MOSS 虚拟配置模块支持
        for m in ["configuration_moss_audio_tokenizer", "configuration_moss_tts_nano", 
                  "modeling_moss_audio_tokenizer", "modeling_moss_tts_nano"]:
            if m not in sys.modules:
                sys.modules[m] = types.ModuleType(m)

        # 3. 禁用 accelerate 探测以防进入 meta 加载
        import transformers.utils.import_utils as import_utils
        import_utils.is_accelerate_available = lambda: False

        # 4. 劫持 torch.load 保持 CPU 重定向
        _orig_torch_load = torch.load
        def _patched_torch_load(*args, **kwargs):
            ml = kwargs.get("map_location")
            if ml == "meta" or getattr(ml, "type", "") == "meta":
                kwargs["map_location"] = "cpu"
            return _orig_torch_load(*args, **kwargs)
        torch.load = _patched_torch_load

        # 5. 兼容新版 transformers 中 _keys_to_ignore_* 为 list 时引发的 unsupported operand type(s) for |: 'list' and 'set'
        from transformers.modeling_utils import PreTrainedModel
        _orig_pt_from_pretrained = PreTrainedModel.from_pretrained
        @classmethod
        def _patched_pt_from_pretrained(cls, *args, **kwargs):
            for attr in ["_keys_to_ignore_on_load_unexpected", "_keys_to_ignore_on_load_missing"]:
                val = getattr(cls, attr, None)
                if isinstance(val, (list, tuple)):
                    setattr(cls, attr, set(val))
            res = _orig_pt_from_pretrained.__func__(cls, *args, **kwargs)
            for attr in ["_keys_to_ignore_on_load_unexpected", "_keys_to_ignore_on_load_missing"]:
                val = getattr(res, attr, None)
                if isinstance(val, (list, tuple)):
                    setattr(res, attr, set(val))
            return res
        PreTrainedModel.from_pretrained = _patched_pt_from_pretrained

        # 6. 修复部分 transformers 版本中 safetensors metadata 为 None 时 load_state_dict 崩溃 ('NoneType' object has no attribute 'get')
        try:
            import safetensors
            _orig_safe_open = safetensors.safe_open

            class _SafeOpenProxy:
                def __init__(self, *args, **kwargs):
                    self._f = _orig_safe_open(*args, **kwargs)
                    self._ctx = None

                def __enter__(self):
                    self._ctx = self._f.__enter__()
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    if hasattr(self._f, "__exit__"):
                        return self._f.__exit__(exc_type, exc_val, exc_tb)
                    return False

                def metadata(self):
                    target = self._ctx or self._f
                    meta = target.metadata() if hasattr(target, "metadata") else None
                    return meta if meta is not None else {"format": "pt"}

                def __getattr__(self, name):
                    target = self._ctx or self._f
                    return getattr(target, name)

            safetensors.safe_open = _SafeOpenProxy
            import transformers.modeling_utils as modeling_utils
            if hasattr(modeling_utils, "safe_open"):
                modeling_utils.safe_open = _SafeOpenProxy

            # 7. 补充 load_state_dict 双重容错：若遇任何元数据提取异常，自动退回直接读取 safetensors
            if hasattr(modeling_utils, "load_state_dict"):
                _orig_load_state_dict = modeling_utils.load_state_dict
                def _patched_load_state_dict(checkpoint_file, *args, **kwargs):
                    try:
                        return _orig_load_state_dict(checkpoint_file, *args, **kwargs)
                    except AttributeError as ae:
                        if "has no attribute 'get'" in str(ae):
                            from safetensors.torch import load_file
                            return load_file(str(checkpoint_file), device="cpu")
                        raise
                modeling_utils.load_state_dict = _patched_load_state_dict
        except Exception as e:
            logger.warning(f"safetensors metadata 兼容性补丁注入提示: {e}")

        logger.info(f"🛠️ [CPU-Patch] MOSS-TTS 离线补丁已成功注入 (HF_HOME={hf_cache_path})")
    except Exception as e:
        logger.error(f"注入离线补丁失败: {e}")

def _find_model_directory(root_dir, keyword, indicator_files=("config.json", "model.safetensors")):
    """在 root_dir 下智能递归查找包含特定关键字及标志性权重文件的真实物理目录"""
    if not root_dir or not os.path.exists(root_dir):
        return None

    keyword_lower = keyword.lower()
    candidates = []

    for root, dirs, files in os.walk(root_dir):
        path_lower = root.lower()
        if keyword_lower in path_lower:
            if any(f in files for f in indicator_files):
                candidates.append(root)

    if candidates:
        # 优先匹配 snapshots 下的具体哈希版本目录
        snapshot_cands = [c for c in candidates if "snapshots" in c.lower()]
        if snapshot_cands:
            return sorted(snapshot_cands)[-1]
        return sorted(candidates)[-1]
    return None

def init_tts_engine():
    global MOSS_SERVICE
    if MOSS_SERVICE is not None:
        return MOSS_SERVICE

    # 1. 优先定位项目配置 models/moss_tts 及其缓存目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        MOSS_MODEL_DIR,
        os.path.join(base_dir, "models", "moss_tts"),
        os.path.join(base_dir, "MOSS-TTS-Nano")
    ]
    
    moss_dir = None
    hf_cache_dir = MOSS_CACHE_DIR if (MOSS_CACHE_DIR and os.path.exists(MOSS_CACHE_DIR)) else None
    for c in candidates:
        if c and os.path.exists(c):
            moss_dir = c
            if not hf_cache_dir:
                cache_p = os.path.join(c, "hf_cache")
                if os.path.exists(cache_p):
                    hf_cache_dir = cache_p
            break

    # 2. 激活离线补丁与环境变量
    apply_offline_patches(moss_dir, hf_cache_dir)

    try:
        from moss_tts_nano_runtime import NanoTTSService
        import threading
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
        
        # 智能动态定位本地 snapshot 真实物理路径 (自适应 ModelScope 与 HuggingFace 结构)
        search_root = hf_cache_dir or moss_dir
        ckpt_path = _find_model_directory(search_root, "MOSS-TTS-Nano", ("config.json", "model.safetensors"))
        tok_path = _find_model_directory(search_root, "Audio-Tokenizer", ("config.json", "model.safetensors"))

        kwargs = {
            "device": "cpu",
            "dtype": "float32",
            "attn_implementation": "sdpa"
        }
        if ckpt_path and os.path.exists(ckpt_path):
            kwargs["checkpoint_path"] = ckpt_path
        if tok_path and os.path.exists(tok_path):
            kwargs["audio_tokenizer_path"] = tok_path

        logger.info(f"⏳ 正在加载 MOSS-TTS (模型: {ckpt_path or '默认'}, Tokenizer: {tok_path or '默认'}) ...")
        MOSS_SERVICE = NanoTTSService(**kwargs)

        # 异步预热模型，避免第一次语音合成时出现长延迟卡顿
        def _async_warmup():
            try:
                MOSS_SERVICE.get_model()
                MOSS_SERVICE._load_audio_tokenizer_locked(tts_attn_implementation="sdpa")
                logger.info("⚡ MOSS-TTS 内存预热完成，已进入毫秒发音状态")
            except Exception as ex:
                logger.warning(f"MOSS-TTS 预热异常: {ex}", exc_info=True)
        threading.Thread(target=_async_warmup, daemon=True).start()

        logger.info("✅ MOSS-TTS 引擎初始化成功")
    except ImportError:
        logger.warning("未检测到 moss_tts_nano_runtime 库，TTS 暂处于仿真模式")
    except Exception as e:
        logger.error(f"初始化 MOSS TTS 失败: {e}")
    return MOSS_SERVICE

def synthesize_and_enqueue(text: str):
    """合成语音并将生成的双声道 PCM 推入播放队列"""
    clean_text = filter_symbols(text)
    if not clean_text:
        return False

    logger.info(f"🔊 [TTS 请求]: {clean_text}")

    if MOSS_SERVICE is None:
        init_tts_engine()

    if MOSS_SERVICE:
        try:
            result = MOSS_SERVICE.synthesize(text=clean_text)
            waveform = result["waveform"]
            sr = result["sample_rate"]

            if sr != HW_SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, HW_SAMPLE_RATE)

            # 异常长度截断保护 (防止 MOSS-TTS 偶发幻觉产生超长尾音)
            max_allowed_seconds = len(clean_text) * 0.4 + 2.0
            max_allowed_frames = int(max_allowed_seconds * HW_SAMPLE_RATE)
            if waveform.shape[-1] > max_allowed_frames:
                waveform = waveform[..., :max_allowed_frames]

            if waveform.shape[0] == 1 and PLAYBACK_CHANNELS == 2:
                waveform = waveform.repeat(2, 1)

            pcm_data = (torch.clamp(waveform, -1.0, 1.0) * 32767).to(torch.int16).cpu().numpy().T.tobytes()
            GLOBAL_AUDIO_QUEUE.put(pcm_data)
            return True
        except Exception as e:
            logger.error(f"TTS 合成出错: {e}")
            return False
    else:
        logger.warning(f"TTS 服务未就绪，模拟播报: {clean_text}")
        return False
