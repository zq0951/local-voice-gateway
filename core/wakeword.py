import os
import logging
import numpy as np
from config import WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD

logger = logging.getLogger("LocalVoiceGateway")

class WakeWordDetector:
    def __init__(self, model_names=None, threshold=WAKE_WORD_THRESHOLD):
        if model_names is None:
            model_names = WAKE_WORD_MODEL
        if isinstance(model_names, str):
            model_names = [w.strip() for w in model_names.split(",") if w.strip()]
        self.model_names = model_names
        self.threshold = threshold
        self.model = None
        self._is_loaded = False
        self._load_failed = False

    @staticmethod
    def get_available_models():
        """扫描本地 models/wakeword 目录，返回所有支持的内置唤醒词别名"""
        local_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/wakeword")
        available = set()
        if os.path.exists(local_dir):
            for f in os.listdir(local_dir):
                if f.endswith(".onnx") and not f.startswith("melspectrogram") and not f.startswith("embedding") and not f.startswith("silero"):
                    clean_name = f.replace(".onnx", "").replace("_v0.1", "").replace("_v0.2", "")
                    available.add(clean_name)
        return sorted(list(available))

    def reload_models(self, new_models):
        """动态热重载唤醒词模型"""
        if isinstance(new_models, str):
            new_models = [w.strip() for w in new_models.split(",") if w.strip()]
        self.model_names = new_models
        self._is_loaded = False
        self._load_failed = False
        return self.load_model()

    def _resolve_model_paths(self):
        """将模型别名（如 hey_jarvis）自动解析为本地 models/wakeword 目录下的具体 onnx 路径"""
        local_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/wakeword")
        resolved = []
        for name in self.model_names:
            if os.path.exists(name):
                resolved.append(name)
                continue
            
            # 常见候选命名
            candidates = [
                os.path.join(local_dir, f"{name}.onnx"),
                os.path.join(local_dir, f"{name}_v0.1.onnx"),
                os.path.join(local_dir, f"{name}_v0.2.onnx"),
            ]
            found = False
            for c in candidates:
                if os.path.exists(c):
                    resolved.append(c)
                    found = True
                    break
            if not found:
                # 保持原名称让库自己尝试查找
                resolved.append(name)
        return resolved

    def load_model(self):
        if self._is_loaded:
            return True
        if self._load_failed:
            return False

        try:
            import openwakeword
            from openwakeword.model import Model
            
            local_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "wakeword")
            melspec_path = os.path.join(local_dir, "melspectrogram.onnx")
            embedding_path = os.path.join(local_dir, "embedding_model.onnx")

            resolved_paths = self._resolve_model_paths()
            logger.info(f"⏳ 正在加载 OpenWakeWord 模型: {resolved_paths} ...")

            model_kwargs = {
                "wakeword_models": resolved_paths,
                "inference_framework": "onnx"
            }
            if os.path.exists(melspec_path):
                model_kwargs["melspec_model_path"] = melspec_path
            if os.path.exists(embedding_path):
                model_kwargs["embedding_model_path"] = embedding_path

            self.model = Model(**model_kwargs)
            self._is_loaded = True
            logger.info(f"✅ OpenWakeWord 唤醒引擎加载完成 (阈值: {self.threshold})")
            return True
        except ImportError:
            logger.warning("未安装 openwakeword 库，关键词唤醒模式将不可用")
            self._load_failed = True
            return False
        except Exception as e:
            logger.error(f"加载 openwakeword 失败: {e}")
            self._load_failed = True
            return False

    def reset(self):
        if self._is_loaded and self.model:
            self.model.reset()

    def process_chunk(self, pcm_chunk: bytes):
        """
        处理单帧 16-bit 16kHz PCM 音频块 (通常 1280 样本 = 80ms)
        返回: (is_triggered: bool, model_name: str, score: float)
        """
        if not self._is_loaded:
            if not self.load_model():
                return False, None, 0.0

        audio_data = np.frombuffer(pcm_chunk, dtype=np.int16)
        predictions = self.model.predict(audio_data)

        for model_name, score in predictions.items():
            if score >= self.threshold:
                logger.info(f"🎯 关键词触发! 模型={model_name}, 置信度={score:.4f}")
                self.reset()
                return True, model_name, float(score)

        return False, None, 0.0

WAKEWORD_DETECTOR = WakeWordDetector()
