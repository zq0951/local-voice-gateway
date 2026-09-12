import os
import logging
import numpy as np
from config import SPEAKER_MODEL_PATH, VOICE_PROFILES_DIR, VOICEPRINT_THRESHOLD

logger = logging.getLogger("LocalVoiceGateway")

class SpeakerVerifier:
    def __init__(self, model_path=SPEAKER_MODEL_PATH, profile_dir=VOICE_PROFILES_DIR, threshold=VOICEPRINT_THRESHOLD):
        self.model_path = model_path
        self.profile_dir = profile_dir
        self._threshold = float(threshold)
        self.session = None
        
        os.makedirs(self.profile_dir, exist_ok=True)
        
        self.ort = None
        self.torch = None
        self.torchaudio = None
        self.centroids = {}

    @property
    def threshold(self) -> float:
        import config
        return getattr(config, "VOICEPRINT_THRESHOLD", self._threshold)

    @threshold.setter
    def threshold(self, val: float):
        import config
        self._threshold = float(val)
        config.VOICEPRINT_THRESHOLD = float(val)

    def _safe_load_audio(self, audio_path):
        import soundfile as sf
        import torch
        data, sr = sf.read(audio_path)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        waveform = torch.from_numpy(data.T).float()
        return waveform, sr

    def _load_model(self):
        if self.session is not None:
            return True
            
        if os.path.isdir(self.model_path):
            logger.critical(
                f"❌ 严重错误: {self.model_path} 是一个目录而非 ONNX 模型文件！"
                "这通常是因为在未下载模型前直接执行了 docker compose up 触发了 Docker bind-mount 自动建目录坑。"
                f"请先在宿主机删除该目录 ('rm -rf {self.model_path}')，并运行 'python utils/download_models.py --campplus' 拉取真实模型！"
            )
            return False

        if not os.path.exists(self.model_path):
            logger.error(f"Speaker verification model not found at {self.model_path}")
            return False
            
        try:
            import onnxruntime as ort
            import torch
            import torchaudio
            
            # 解决 torchaudio 新版本强制依赖 torchcodec 的问题
            torchaudio.load = lambda p, **kw: self._safe_load_audio(p)
            
            self.ort = ort
            self.torch = torch
            self.torchaudio = torchaudio
            
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            logger.info(f"✅ Speaker Verification Model (CAM++) loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load SpeakerVerifier ONNX model: {e}")
            return False

    def reload_profiles(self):
        """加载已注册说话人的声纹并计算特征质心 (Centroids)"""
        if not os.path.exists(self.profile_dir):
            return
            
        self.centroids.clear()
        
        for user_name in sorted(os.listdir(self.profile_dir)):
            if user_name.startswith("_") or user_name.startswith("."):
                continue
            user_dir = os.path.join(self.profile_dir, user_name)
            if not os.path.isdir(user_dir):
                continue
                
            embeddings = []
            files = [f for f in os.listdir(user_dir) if f.endswith('.wav')]
            
            for f in files:
                file_path = os.path.join(user_dir, f)
                emb = self.extract_embedding(file_path)
                if emb is not None:
                    if isinstance(emb, list):
                        embeddings.extend(emb)
                    else:
                        embeddings.append(emb)
            
            if embeddings:
                embeddings = np.array(embeddings)
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                self.centroids[user_name] = centroid
                logger.info(f"Loaded voice profile for '{user_name}': {len(embeddings)} samples.")

    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.duration
        except Exception:
            waveform, sr = self.torchaudio.load(audio_path)
            return waveform.shape[-1] / sr

    def _extract_embedding_from_waveform(self, waveform_tensor, sample_rate):
        waveform = waveform_tensor * 32768.0
        fbank = self.torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            energy_floor=0.0,
            dither=0.0
        )
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        fbank_np = fbank.unsqueeze(0).numpy().astype(np.float32)
        ort_inputs = {self.session.get_inputs()[0].name: fbank_np}
        embeddings = self.session.run(None, ort_inputs)[0]
        return embeddings[0]

    def _trim_silence(self, waveform):
        import torch
        w = waveform.squeeze(0)
        window = 1600
        if w.shape[0] < window * 2:
            return waveform
            
        sq = w.pow(2)
        cumsum = torch.cumsum(sq, dim=0)
        ma_energy = (cumsum[window:] - cumsum[:-window]) / window
        
        max_energy = ma_energy.max()
        threshold = max_energy * 0.01
        
        mask = ma_energy > threshold
        if not mask.any():
            return waveform
            
        indices = mask.nonzero(as_tuple=True)[0]
        start_idx = indices[0].item()
        end_idx = indices[-1].item() + window
        
        buffer = 1600
        start_idx = max(0, start_idx - buffer)
        end_idx = min(w.shape[0], end_idx + buffer)
        
        return waveform[:, start_idx:end_idx]

    def extract_embedding(self, audio_path: str):
        self._load_model()
        if self.session is None:
            return None

        try:
            waveform, sample_rate = self.torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = self.torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            waveform = self._trim_silence(waveform)
            return self._extract_embedding_from_waveform(waveform, 16000)

        except Exception as e:
            logger.error(f"Error extracting embedding from {audio_path}: {e}")
            return None

    def _get_effective_threshold(self, duration: float) -> float:
        if duration < 0.8:
            return 0.45
        if duration < 1.5:
            ratio = (duration - 0.8) / 0.7
            return 0.45 + ratio * (self.threshold - 0.45)
        if duration >= 5.0:
            return max(0.45, self.threshold - 0.10)
        return self.threshold

    def verify(self, audio_path: str):
        """
        验证音频说话人身份
        返回: (is_matched: bool, best_speaker: str, score: float)
        """
        if not self._load_model():
            logger.error("❌ 声纹验证模型加载失败或文件缺失，安全拦截 (Fail-closed)")
            return False, "unknown", 0.0
            
        if not self.centroids:
            logger.warning("⚠️ [声纹核对]: 当前声纹库为空，已拒绝未认证说话人。")
            return False, "unknown", 0.0
        
        duration = self._get_audio_duration(audio_path)
        target_embedding = self.extract_embedding(audio_path)
        if target_embedding is None:
            return False, "unknown", 0.0
            
        if isinstance(target_embedding, list):
            target_embeddings = [t / np.linalg.norm(t) for t in target_embedding]
        else:
            target_embeddings = [target_embedding / np.linalg.norm(target_embedding)]
        
        best_score = -1.0
        best_speaker = "unknown"
        
        for speaker_id, centroid in self.centroids.items():
            for t_emb in target_embeddings:
                score = np.dot(t_emb, centroid)
                if score > best_score:
                    best_score = score
                    best_speaker = speaker_id
        
        effective = self._get_effective_threshold(duration)
        logger.info(f"🎙️ 声纹核对: 最佳匹配 '{best_speaker}' 分数={best_score:.4f} "
                    f"时长={duration:.1f}s 门限={effective:.2f} (基准={self.threshold})")
        
        if best_score >= effective:
            return True, best_speaker, float(best_score)
            
        return False, best_speaker, float(best_score)

SPEAKER_VERIFIER = SpeakerVerifier()
