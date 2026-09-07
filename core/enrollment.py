import os
import time
import shutil
import logging
import uuid
import numpy as np
import threading
from typing import Dict, List, Optional

import config
from core.vad import record_audio_until_silence, cleanup_arecord, is_enrolling_event
from core.speaker_verifier import SPEAKER_VERIFIER

logger = logging.getLogger("LocalVoiceGateway")

# 推荐经典诗词名句 (平仄起伏分明、开闭音节兼备、声母韵母全面覆盖，极利于 CAM++ 提取高辨识度声纹特征)
DEFAULT_ENROLL_PROMPTS = [
    "请清晰朗读：大鹏一日同风起，扶摇直上九万里",
    "请清晰朗读：海上生明月，天涯共此时",
    "请清晰朗读：会当凌绝顶，一览众山小",
    "请清晰朗读：长风破浪会有时，直挂云帆济沧海",
    "请清晰朗读：莫道前路无知己，天下谁人不识君"
]

class EnrollmentSession:
    def __init__(self, session_id: str, speaker_name: str, total_steps: int = 5):
        self.session_id = session_id
        self.speaker_name = speaker_name
        self.total_steps = min(max(total_steps, 3), len(DEFAULT_ENROLL_PROMPTS))
        self.prompts = DEFAULT_ENROLL_PROMPTS[:self.total_steps]
        self.current_step = 0
        self.wav_paths: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.created_at = time.time()
        self.temp_dir = os.path.join(config.BASE_DIR, "models", "voice_profiles", f"_temp_enroll_{session_id}")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.lock = threading.Lock() # 会话级互斥锁，防止并发 record_step 与 finish/abort 竞态破坏

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "speaker_name": self.speaker_name,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_prompt": self.prompts[self.current_step] if self.current_step < self.total_steps else None,
            "all_prompts": self.prompts,
            "is_completed": self.current_step >= self.total_steps
        }

class VoiceprintManager:
    def __init__(self):
        self.sessions: Dict[str, EnrollmentSession] = {}
        self._lock = threading.Lock()
        self.is_enrolling_event = is_enrolling_event
        self.clean_orphaned_dirs()

    def clean_orphaned_dirs(self) -> int:
        """清理历史遗留或非正常退出的 _temp_enroll_* 临时孤儿目录"""
        if not os.path.exists(config.VOICE_PROFILES_DIR):
            return 0
        cleaned = 0
        for item in os.listdir(config.VOICE_PROFILES_DIR):
            if item.startswith("_temp_enroll_"):
                p = os.path.join(config.VOICE_PROFILES_DIR, item)
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
                        cleaned += 1
                except Exception as e:
                    logger.warning(f"清理临时录音孤儿目录失败 {p}: {e}")
        if cleaned > 0:
            logger.info(f"🧹 [声纹目录清扫] 已清理 {cleaned} 个残留的临时声纹录入孤儿目录")
        return cleaned

    def start_session(self, speaker_name: str, total_steps: int = 5) -> dict:
        """开启引导录入会话，通知主循环挂起麦克风占用"""
        speaker_name = speaker_name.strip()
        if not speaker_name or speaker_name.startswith(("_", ".")) or any(c in speaker_name for c in r'\/:*?"<>|'):
            raise ValueError(f"说话人名称非法或不能以 '_' / '.' 开头: '{speaker_name}'")

        with self._lock:
            # 清理超时的陈旧会话 (> 10 分钟)
            now = time.time()
            expired = [sid for sid, s in self.sessions.items() if now - s.created_at > 600]
            for sid in expired:
                self.abort_session(sid)

            session_id = str(uuid.uuid4())[:8]
            session = EnrollmentSession(session_id, speaker_name, total_steps)
            self.sessions[session_id] = session

            # 激活录音互斥锁，通知主循环让出声卡
            is_enrolling_event.set()
            cleanup_arecord()
            time.sleep(0.2)

            logger.info(f"🎙️ [声纹录入] 会话启动: {session_id}, 说话人: {speaker_name}, 总步骤: {session.total_steps}")
            return session.to_dict()

    def record_step(self, session_id: str) -> dict:
        """执行单步短语麦克风录制并提取声纹特征 (会话全程持锁防止并发重入)"""
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError("录入会话不存在或已过期")

        with session.lock:
            if session.current_step >= session.total_steps:
                return {"status": "already_completed", **session.to_dict()}

            # 录制目标音频文件 (持有会话锁，确保并发请求严格排队，且 abort 必须等待录音收尾)
            step_idx = session.current_step
            out_wav = os.path.join(session.temp_dir, f"step_{step_idx + 1}.wav")

            logger.info(f"🎤 [声纹录入] 开始录制第 {step_idx + 1}/{session.total_steps} 句: {session.prompts[step_idx]}")
            recorded_path, rms = record_audio_until_silence(out_wav, allow_enrolling=True)

            if not recorded_path or not os.path.exists(recorded_path):
                return {
                    "success": False,
                    "error": "未采集到有效声音，请靠近麦克风重新朗读",
                    "message": "未采集到有效声音，请靠近麦克风重新朗读",
                    **session.to_dict()
                }

            # 提取 embedding
            emb = SPEAKER_VERIFIER.extract_embedding(recorded_path)
            if emb is None:
                return {
                    "success": False,
                    "error": "特征提取失败，请重新录制该短语",
                    "message": "特征提取失败，请重新录制该短语",
                    **session.to_dict()
                }

            if isinstance(emb, list):
                emb = emb[0]

            # 归一化特征
            emb = emb / np.linalg.norm(emb)

            session.wav_paths.append(recorded_path)
            session.embeddings.append(emb)
            session.current_step += 1

            logger.info(f"✅ [声纹录入] 第 {step_idx + 1} 步成功采集 (RMS={int(rms)})")

            is_final = session.current_step >= session.total_steps
            return {
                "success": True,
                "step": step_idx + 1,
                "rms": int(rms),
                "is_final_step": is_final,
                "next_step": session.current_step,
                "next_prompt": session.prompts[session.current_step] if not is_final else None,
                **session.to_dict()
            }

    def finish_session(self, session_id: str) -> dict:
        """完成录入，执行离群样本检测与剔除，保存至正式声纹库并热重载 (会话持锁)"""
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError("录入会话不存在或已过期")

        try:
            with session.lock:
                if len(session.embeddings) < 3:
                    raise ValueError(f"已录制样本不足 3 句 (当前 {len(session.embeddings)} 句)，无法完成声纹构建")

                embeddings = session.embeddings
                wav_paths = session.wav_paths
                n = len(embeddings)

                # 计算样本间两两余弦相似度矩阵
                sim_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        sim_matrix[i][j] = np.dot(embeddings[i], embeddings[j])

                # 计算每个样本与其他样本的平均内聚相似度
                mean_sims = []
                for i in range(n):
                    others = [sim_matrix[i][j] for j in range(n) if j != i]
                    mean_sims.append(float(np.mean(others)))

                logger.info(f"📊 [离群检验] 各样本平均相似度: {[round(s, 4) for s in mean_sims]}")

                # 识别离群样本 (相似度异常偏低且样本数 > 3)
                min_idx = int(np.argmin(mean_sims))
                discarded_idx = None
                if mean_sims[min_idx] < 0.60 and n > 3:
                    discarded_idx = min_idx
                    logger.warning(f"⚠️ [离群剔除] 发现样本 {min_idx + 1} 相似度较低 ({mean_sims[min_idx]:.4f} < 0.60)，已自动剔除")

                # 建立正式说话人目录 (若已存在，彻底清理旧样本防止新旧混杂导致质心污染)
                target_dir = os.path.join(config.VOICE_PROFILES_DIR, session.speaker_name)
                if os.path.exists(target_dir):
                    for old_f in os.listdir(target_dir):
                        if old_f.endswith(".wav"):
                            try:
                                os.remove(os.path.join(target_dir, old_f))
                            except Exception as e:
                                logger.warning(f"清理旧样本失败: {e}")
                os.makedirs(target_dir, exist_ok=True)

                saved_count = 0
                for idx, wav_p in enumerate(wav_paths):
                    if idx == discarded_idx:
                        continue
                    saved_count += 1
                    dst = os.path.join(target_dir, f"sample_{saved_count}.wav")
                    shutil.copy2(wav_p, dst)

                # 触发声纹库热重载
                SPEAKER_VERIFIER.reload_profiles()
                logger.info(f"🎉 [声纹录入完成] 说话人 '{session.speaker_name}' 已成功入库 ({saved_count} 个合格样本)")

                return {
                    "status": "ok",
                    "speaker": session.speaker_name,
                    "saved_samples": saved_count,
                    "discarded_outlier": discarded_idx is not None,
                    "similarity_scores": mean_sims
                }
        finally:
            self.cleanup_session(session_id)

    def cleanup_session(self, session_id: str):
        """清理会话临时文件并释放麦克风互斥锁 (会话持锁保证安全清理)"""
        with self._lock:
            session = self.sessions.pop(session_id, None)
            # 若无其他录制会话进行中，恢复主监听循环
            if not self.sessions:
                is_enrolling_event.clear()
                logger.info("🔓 [麦克风锁] 录入结束，主监听循环恢复")

        if session:
            with session.lock:
                if os.path.exists(session.temp_dir):
                    shutil.rmtree(session.temp_dir, ignore_errors=True)

        # 重新加载声纹库，确保内存中的声纹质心与磁盘目录严格同步 (剔除中止/超时会话残留的幽灵质心)
        SPEAKER_VERIFIER.reload_profiles()

    def abort_session(self, session_id: str):
        """强制中止录入会话"""
        self.cleanup_session(session_id)

    def list_profiles(self) -> List[dict]:
        """列出所有已持久化保存的说话人声纹库信息"""
        profiles = []
        if not os.path.exists(config.VOICE_PROFILES_DIR):
            return profiles

        for user_name in sorted(os.listdir(config.VOICE_PROFILES_DIR)):
            if user_name.startswith("_"):
                continue
            user_dir = os.path.join(config.VOICE_PROFILES_DIR, user_name)
            if not os.path.isdir(user_dir):
                continue

            wavs = [f for f in os.listdir(user_dir) if f.endswith(".wav")]
            mtime = os.path.getmtime(user_dir)
            created_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))

            profiles.append({
                "name": user_name,
                "sample_count": len(wavs),
                "created_at": created_str,
                "is_active": user_name in SPEAKER_VERIFIER.centroids
            })
        return profiles

    def delete_profile(self, speaker_name: str) -> bool:
        """删除指定说话人的声纹数据并热重载"""
        user_dir = os.path.join(config.VOICE_PROFILES_DIR, speaker_name)
        if not os.path.exists(user_dir):
            return False

        shutil.rmtree(user_dir, ignore_errors=True)
        SPEAKER_VERIFIER.reload_profiles()
        logger.info(f"🗑️ [声纹删除] 已删除说话人: '{speaker_name}'，声纹库已重载")
        return True

VOICEPRINT_MANAGER = VoiceprintManager()
