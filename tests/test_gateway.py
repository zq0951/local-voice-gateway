import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Ensure local-voice-gateway root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
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
