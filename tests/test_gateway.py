import os
import sys
import io
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

# Ensure local-voice-gateway root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
import api.server as api_server
from api.server import app, sync_broadcast_event
from core.enrollment import VOICEPRINT_MANAGER, EnrollmentSession


class TestGatewayConfig(unittest.TestCase):
    """测试配置管理与环境变量解析"""

    def test_default_config_values(self):
        self.assertIn(config.GATEWAY_HOST, ["127.0.0.1", "0.0.0.0"])
        self.assertEqual(config.GATEWAY_PORT, 8765)
        self.assertIsInstance(config.VOICEPRINT_THRESHOLD, float)
        self.assertGreaterEqual(config.VOICEPRINT_THRESHOLD, 0.0)
        self.assertLessEqual(config.VOICEPRINT_THRESHOLD, 1.0)
        self.assertIsInstance(config.ENERGY_THRESHOLD, int)
        self.assertIsInstance(config.SILENCE_TIMEOUT, float)


class TestGatewayAPI(unittest.TestCase):
    """测试核心 HTTP REST API 端点"""

    @classmethod
    def setUpClass(cls):
        cls.orig_mode = config.TRIGGER_MODE
        cls.orig_auto_speak = config.AUTO_SPEAK

    @classmethod
    def tearDownClass(cls):
        config.TRIGGER_MODE = cls.orig_mode
        config.AUTO_SPEAK = cls.orig_auto_speak

    def setUp(self):
        self.client = TestClient(app)

    def test_get_system_status(self):
        resp = self.client.get("/v1/system/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "running")
        self.assertIn("trigger_mode", data)
        self.assertIn("audio_duplex_mode", data)
        self.assertIn("auto_speak", data)
        self.assertIn("registered_speakers", data)

    def test_set_system_mode_valid(self):
        for mode in ["wake_word", "voiceprint_passive", "hybrid"]:
            resp = self.client.post("/v1/system/mode", json={"mode": mode})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "ok")
            self.assertEqual(resp.json()["current_mode"], mode)

    def test_set_system_mode_invalid(self):
        resp = self.client.post("/v1/system/mode", json={"mode": "invalid_mode_xyz"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    def test_toggle_autospeak(self):
        resp = self.client.post("/v1/system/autospeak", json={"enabled": False})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["auto_speak"], False)

        resp = self.client.post("/v1/system/autospeak", json={"enabled": True})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["auto_speak"], True)

    def test_speak_empty_text_validation(self):
        resp = self.client.post("/v1/audio/speak", json={"text": "   "})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    def test_audio_stop(self):
        resp = self.client.post("/v1/audio/stop")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_list_voiceprint_profiles(self):
        resp = self.client.get("/v1/voiceprint/profiles")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("profiles", resp.json())
        self.assertIsInstance(resp.json()["profiles"], list)

    def test_stt_transcriptions_openai_file_field(self):
        captured = {}

        async def fake_transcribe(path, language="auto", use_itn=True):
            captured["path"] = path
            captured["exists_during_call"] = os.path.exists(path)
            captured["size"] = os.path.getsize(path)
            captured["language"] = language
            captured["use_itn"] = use_itn
            return {
                "text": "打开客厅灯",
                "emotion": "NEUTRAL",
                "lang": "zh",
                "event": "Speech",
                "is_speech": True,
            }

        with patch.object(api_server.LOCAL_STT_ENGINE, "is_initialized", True), \
                patch("api.server.transcribe_audio", side_effect=fake_transcribe):
            resp = self.client.post(
                "/v1/audio/transcriptions",
                files={"file": ("query.wav", io.BytesIO(b"RIFFdemo_wav"), "audio/wav")},
                data={"language": "zh", "model": "sensevoice-small"},
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["text"], "打开客厅灯")
        self.assertEqual(body["language"], "zh")
        self.assertEqual(body["emotion"], "NEUTRAL")
        self.assertTrue(body["is_speech"])
        self.assertTrue(captured["exists_during_call"])
        self.assertTrue(captured["path"].endswith(".wav"))
        self.assertEqual(captured["size"], len(b"RIFFdemo_wav"))
        self.assertFalse(os.path.exists(captured["path"]))

    def test_stt_alias_audio_field_and_text_format(self):
        mock_stt = AsyncMock(return_value={
            "text": "hello jarvis",
            "emotion": None,
            "lang": "en",
            "event": "Speech",
            "is_speech": True,
        })
        with patch.object(api_server.LOCAL_STT_ENGINE, "is_initialized", True), \
                patch("api.server.transcribe_audio", mock_stt):
            resp = self.client.post(
                "/v1/audio/stt",
                files={"audio": ("clip.m4a", io.BytesIO(b"m4abytes"), "audio/mp4")},
                data={"response_format": "text"},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.text, "hello jarvis")
        self.assertTrue(resp.headers["content-type"].startswith("text/plain"))

    def test_stt_validation_and_unavailable(self):
        # Missing file/audio field -> 400
        resp_missing = self.client.post("/v1/audio/transcriptions")
        self.assertEqual(resp_missing.status_code, 400)

        # Empty file -> 400
        resp_empty = self.client.post(
            "/v1/audio/transcribe",
            files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
        )
        self.assertEqual(resp_empty.status_code, 400)

        # Uninitialized engine when init fails -> 503
        with patch.object(api_server.LOCAL_STT_ENGINE, "is_initialized", False), \
                patch("api.server.init_stt_engine", return_value=False):
            resp_503 = self.client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", io.BytesIO(b"abc"), "audio/wav")},
            )
        self.assertEqual(resp_503.status_code, 503)

        # Exceeding max upload MB -> 413
        orig_max = config.STT_MAX_AUDIO_MB
        try:
            config.STT_MAX_AUDIO_MB = 1
            oversize = b"a" * (1024 * 1024 + 10)
            resp_413 = self.client.post(
                "/v1/audio/stt",
                files={"file": ("big.wav", io.BytesIO(oversize), "audio/wav")},
            )
            self.assertEqual(resp_413.status_code, 413)
        finally:
            config.STT_MAX_AUDIO_MB = orig_max


class TestVoiceprintEnrollment(unittest.TestCase):
    """测试声纹录入引导状态机与契约"""

    def setUp(self):
        self.speaker = "test_user_ci"
        self.session_id = None

    def tearDown(self):
        if self.session_id and self.session_id in VOICEPRINT_MANAGER.sessions:
            VOICEPRINT_MANAGER.abort_session(self.session_id)
        # Cleanup mock speaker profile if created
        VOICEPRINT_MANAGER.delete_profile(self.speaker)

    def test_enrollment_lifecycle(self):
        # 1. Start enrollment
        session_info = VOICEPRINT_MANAGER.start_session(self.speaker, total_steps=3)
        self.session_id = session_info["session_id"]
        self.assertEqual(session_info["speaker_name"], self.speaker)
        self.assertEqual(session_info["total_steps"], 3)
        self.assertEqual(session_info["current_step"], 0)

        # 2. Check session retrieval
        session = VOICEPRINT_MANAGER.sessions.get(self.session_id)
        self.assertIsNotNone(session)

        # 3. Simulate step completion by populating mock embedding
        session.embeddings.append([0.1] * 192)
        session.current_step = 1

        # 4. Verify abort cleans up directory
        temp_dir = session.temp_dir
        VOICEPRINT_MANAGER.abort_session(self.session_id)
        self.assertNotIn(self.session_id, VOICEPRINT_MANAGER.sessions)
        self.assertFalse(os.path.exists(temp_dir))

    def test_already_completed_contract(self):
        """验证录制超出总步骤时返回的契约包含 success: True 与 is_completed: True"""
        session_info = VOICEPRINT_MANAGER.start_session(self.speaker, total_steps=3)
        self.session_id = session_info["session_id"]
        session = VOICEPRINT_MANAGER.sessions[self.session_id]

        # Manually set current_step to total_steps
        session.current_step = session.total_steps
        result = VOICEPRINT_MANAGER.record_step(self.session_id)

        self.assertEqual(result.get("status"), "ok")
        self.assertTrue(result.get("success"))
        self.assertTrue(result.get("is_completed"))
        self.assertTrue(result.get("already_completed"))


import core.tts as core_tts
import torch
import wave


class TestGatewayTts(unittest.TestCase):
    """测试 HTTP 整段单次 TTS 接口与 WAV 编码"""

    def setUp(self):
        self.client = TestClient(app)

    def test_speech_endpoint_single_pass_wav(self):
        multi_sentence = "第一句话，检查一下网关连接。第二句话，音色必须保持一致！第三句话 Hello."
        fake_wave = torch.tensor([[0.1, -0.2, 0.3, -0.15] * 100], dtype=torch.float32)
        fake_service = MagicMock()
        fake_service.synthesize.return_value = {
            "sample_rate": 24000,
            "waveform": fake_wave,
            "audio_path": "",
        }

        with patch.object(core_tts, "MOSS_SERVICE", fake_service):
            resp = self.client.post("/v1/audio/speech", json={"input": multi_sentence})

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.headers["content-type"].startswith("audio/wav"))
        self.assertTrue(resp.content.startswith(b"RIFF"))

        with wave.open(io.BytesIO(resp.content), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 24000)
            self.assertEqual(wf.getnframes(), 400)

        # 关键约束：整段文本必须一次送入 synthesize，不能分句多次调用
        fake_service.synthesize.assert_called_once()
        called_kwargs = fake_service.synthesize.call_args.kwargs
        called_args = fake_service.synthesize.call_args.args
        passed_text = called_kwargs.get("text") if "text" in called_kwargs else called_args[0]
        self.assertEqual(passed_text, core_tts.filter_symbols(multi_sentence))

    def test_tts_alias_and_voice_forwarding(self):
        fake_wave = torch.tensor([0.05, -0.05] * 50, dtype=torch.float32)
        fake_service = MagicMock()
        fake_service.synthesize.return_value = {
            "sample_rate": 24000,
            "waveform": fake_wave,
        }

        with patch.object(core_tts, "MOSS_SERVICE", fake_service):
            resp = self.client.post("/v1/audio/tts", json={"text": "测试别名与音色透传", "voice": "female_soft"})

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.content.startswith(b"RIFF"))
        fake_service.synthesize.assert_called_once()
        kwargs = fake_service.synthesize.call_args.kwargs
        self.assertEqual(kwargs.get("voice"), "female_soft")

    def test_speech_validation_unavailable_and_error(self):
        with patch.object(core_tts, "MOSS_SERVICE", MagicMock()):
            empty_resp = self.client.post("/v1/audio/speech", json={"input": "   "})
            self.assertEqual(empty_resp.status_code, 400)

            long_resp = self.client.post("/v1/audio/speech", json={"text": "字" * 1201})
            self.assertEqual(long_resp.status_code, 400)

        with patch.object(core_tts, "MOSS_SERVICE", None), \
             patch.object(core_tts, "init_tts_engine", return_value=None):
            unavail_resp = self.client.post("/v1/audio/speech", json={"input": "你好"})
            self.assertEqual(unavail_resp.status_code, 503)

        bad_service = MagicMock()
        bad_service.synthesize.side_effect = RuntimeError("mock infer boom")
        with patch.object(core_tts, "MOSS_SERVICE", bad_service):
            err_resp = self.client.post("/v1/audio/speech", json={"input": "你好"})
            self.assertEqual(err_resp.status_code, 500)

    def test_status_reports_tts_ready(self):
        with patch.object(core_tts, "MOSS_SERVICE", MagicMock()):
            resp = self.client.get("/v1/system/status")
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json().get("tts_ready"))


class TestSyncBroadcast(unittest.TestCase):
    """测试同步广播跨线程安全性"""

    def test_sync_broadcast_no_exception_when_idle(self):
        # Even without running event loop or connected clients, it should safely return
        try:
            sync_broadcast_event({"event": "test_ping", "data": 123})
        except Exception as e:
            self.fail(f"sync_broadcast_event raised an unexpected exception: {e}")


if __name__ == "__main__":
    unittest.main()
