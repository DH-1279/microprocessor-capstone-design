import tempfile
import threading
import time
import unittest
import wave
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from types import SimpleNamespace

from KHS_KMJ_vision.audio import AudioService, build_whisper_command
from KHS_KMJ_vision.dialogue import Announcements, HAPTIC_HELP, target_announcement
from KHS_KMJ_vision.speech import match_wake, normalize_transcript, parse
from ble_sender.controller import Controller
from ble_sender.link import Link
from test_sender import packet


TARGETS = {
    "cup": {"name": "컵", "aliases": ["컵", "머그컵"]},
    "Wallet": {"name": "지갑", "aliases": ["지갑", "카드지갑"]},
    "cell phone": {"name": "휴대폰", "aliases": ["휴대폰", "스마트폰"]},
}
WAKE = ("캡스톤", "갭스톤", "캡손", "캐스톤", "캡스", "캡스턴")


class SpeechQATests(unittest.TestCase):
    def test_known_english_wake_and_spacing_variants(self):
        for text in ("캡스톤", "캡 스톤", "갭스톤", "캡스톤아", "Kept stone!", "CAPSTONE", "cap-stone", "hey capstone"):
            self.assertTrue(match_wake(text, WAKE), text)

    def test_noncommands_and_negation_do_not_wake(self):
        for text in ("", "stone", "kept", "I kept stone yesterday", "캡스톤 말고", "캡스톤 정지", "캡스톤아니야", "돌을 찾았어"):
            self.assertFalse(match_wake(text, WAKE), text)

    def test_multilingual_objects_and_help_keep_grammar(self):
        for text, expected in (("WALLET", "Wallet"), ("cell phone 찾아줘", "cell phone"), ("CELL PHONE을 찾아줘", "cell phone")):
            intent = parse(text, TARGETS, state="scanning")
            self.assertEqual(intent.name, expected, (text, intent))
        for text in ("컵 말고", "cup wallet", "책상 정리했어"):
            self.assertFalse(parse(text, TARGETS, state="scanning"))
        for state in ("scanning", "guiding", "waiting"):
            self.assertEqual(parse("설명", TARGETS, state).name, "help")
        self.assertEqual(normalize_transcript("  CAP-STONE!! "), "cap stone")

    def test_short_instruction_and_retry_when_stt_is_busy(self):
        self.assertIn(HAPTIC_HELP, target_announcement("지갑", True))
        self.assertNotIn(HAPTIC_HELP, target_announcement("컵", False))
        audio = Mock()
        audio.say.side_effect = [False, True]
        pending = Announcements(audio)
        self.assertTrue(pending.say("지갑을 찾습니다."))
        self.assertEqual(len(pending.pending), 1)
        pending.flush()
        self.assertFalse(pending.pending)


class ReadyToneTests(unittest.TestCase):
    def test_queued_tts_then_one_tone_then_input_ready(self):
        service = AudioService({"echo_cooldown_ms": 0}, lambda _: None)
        service._started = True
        trace = []
        def speak(text):
            self.assertFalse(service.input_ready)
            trace.append(text)
        def tone():
            self.assertFalse(service.input_ready)
            service._capture(b"\x00\x10" * 320, 320, None, None)
            self.assertTrue(service._pcm.empty())
            trace.append("TONE")
        service._play, service._play_ready_tone = speak, tone
        self.assertTrue(service.say("첫 안내"))
        self.assertTrue(service.say("연결 안내"))
        self.assertTrue(service.say("연결 안내"))  # Deduplicated.
        worker = threading.Thread(target=service._speak_worker)
        worker.start()
        try:
            deadline = time.monotonic() + 2
            while not service.input_ready and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(service.input_ready)
            self.assertEqual(trace, ["첫 안내", "연결 안내", "TONE"])
            service._capture(b"\x00\x10" * 320, 320, None, None)
            self.assertFalse(service._pcm.empty())
        finally:
            service._stop.set()
            worker.join(1)
        self.assertFalse(worker.is_alive())

    def test_tone_wav_and_prompt_are_local_and_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            service = AudioService({"cache_dir": directory}, lambda _: None)
            service._run = Mock()
            service._play_ready_tone()
            with wave.open(str(Path(directory) / "input-ready-v1.wav")) as audio:
                self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()),
                                 (1, 2, 16000, 1280))
            base = build_whisper_command({}, Path("in.wav"), Path("out"))
            hinted = build_whisper_command({}, Path("in.wav"), Path("out"), prompt="캡스톤 지갑")
            self.assertNotIn("--prompt", base)
            self.assertEqual(hinted[-2:], ["--prompt", "캡스톤 지갑"])
            self.assertEqual(hinted[hinted.index("-l") + 1], "ko")

    def test_context_invalidates_old_input_once_and_bounds_tts_queue(self):
        service = AudioService({}, lambda _: None)
        service.set_context("WAIT_WAKE")
        generation = service._input_generation
        service.set_context("WAIT_WAKE")
        self.assertEqual(generation, service._input_generation)
        service._listening = service._vad.active = True
        service._pcm.put_nowait((time.monotonic(), b"\x00\x10"))
        service.set_context("WAIT_TARGET", ("컵", "지갑"))
        self.assertFalse(service._listening or service._vad.active)
        self.assertTrue(service._pcm.empty())
        self.assertGreater(service._input_generation, generation)
        self.assertIn("지갑", service._prompt)
        service._started = True
        for n in range(8):
            self.assertTrue(service.say(f"안내{n}"))
        self.assertFalse(service.say("추가 안내"))


class RecoveryGateTests(unittest.TestCase):
    def test_offline_cues_not_replayed_and_explicit_stop_survives_reconnect(self):
        clock = [0.0]
        c = Controller(mode=3, stable_frames=1, clock=lambda: clock[0])
        c.set_link_ready(False)
        clock[0] = 1
        c.receive(packet(1, captured=1, event="arrival"))
        self.assertIsNone(c.cue())
        clock[0] = 1.1
        c.set_link_ready(True)
        c.receive(packet(2, captured=1.2, event="arrival"), now=1.2)
        self.assertIsNone(c.cue(now=1.2))
        c.command("stop")
        c.set_link_ready(False)
        clock[0] = 2
        c.set_link_ready(True)
        c.receive(packet(3, captured=2.1, target_id="fresh"), now=2.1)
        self.assertTrue(c.paused)
        self.assertIsNone(c.cue(now=2.1))
        c.command("resume")
        self.assertIsNone(c.cue())
        c.receive(packet(4, captured=2.2, target_id="fresh"), now=2.2)
        self.assertIsNotNone(c.cue(now=2.2))


class BackendErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_backend_error_is_retryable_connection_error(self):
        class AdapterError(Exception):
            pass
        module = SimpleNamespace(BleakScanner=SimpleNamespace(
            find_device_by_address=AsyncMock(side_effect=AdapterError("adapter off"))), BleakClient=Mock())
        with patch.dict(sys.modules, bleak=module), patch("builtins.print"):
            with self.assertRaisesRegex(ConnectionError, "BLE 검색 실패"):
                async with Link("address"):
                    self.fail("not connected")
