"""USB microphone reconnect and one-shot BLE connection feedback."""

from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from KHS_KMJ_vision.alsa_input import AlsaMicrophone
from KHS_KMJ_vision.audio import AudioService
from ble_sender.link import Link
from ble_sender.protocol import PACKET


class MicrophoneRecoveryTests(unittest.TestCase):
    def test_absent_then_on_off_on_reopens_by_card_id_and_silence_is_healthy(self):
        blocks, states = [], []
        def process():
            child = Mock()
            child.poll.return_value = None
            child.terminate.side_effect = lambda: setattr(child.poll, "return_value", 0)
            return child
        children = [process(), process()]
        silence = b"\x00" * 640
        mic = AlsaMicrophone("arecord", "plughw:CARD=GM20U,DEV=0", 640,
                             blocks.append, lambda *args: states.append(args), retry_s=0.001)
        def receive(fd, size):
            # Two silent PCM blocks, USB loss, then one block after reappearance.
            if len(blocks) == 2 and len(states) == 2:
                return b""
            return silence
        def captured(block):
            blocks.append(block)
            if len(blocks) == 3:
                mic._stop.set()
        mic.on_pcm = captured
        selector = Mock()
        selector.__enter__ = Mock(return_value=selector)
        selector.__exit__ = Mock(return_value=False)
        selector.select.return_value = [object()]
        with patch("KHS_KMJ_vision.alsa_input.subprocess.Popen",
                   side_effect=[OSError("USB absent"), *children]) as popen, \
                patch("KHS_KMJ_vision.alsa_input.selectors.DefaultSelector", return_value=selector), \
                patch("KHS_KMJ_vision.alsa_input.os.read", side_effect=receive):
            mic.start()
            mic._thread.join(1)
            self.assertFalse(mic._thread.is_alive())
            self.assertEqual([online for online, _ in states], [False, True, False, True])
            self.assertEqual(blocks, [silence] * 3)
            self.assertEqual(popen.call_count, 3)
            for call in popen.call_args_list:
                self.assertIn("plughw:CARD=GM20U,DEV=0", call.args[0])
            for child in children:
                child.stdout.close.assert_called_once()
            self.assertIsNone(mic._process)

    def test_stalled_pcm_reopens_and_shutdown_interrupts_retry(self):
        mic = AlsaMicrophone("arecord", "stable-id", 640, lambda _: None, lambda *a: None,
                             retry_s=0.001, stall_s=0.5)
        child = Mock()
        child.poll.return_value = None
        selector = Mock()
        selector.__enter__ = Mock(return_value=selector)
        selector.__exit__ = Mock(return_value=False)
        selector.select.return_value = []
        with patch("KHS_KMJ_vision.alsa_input.subprocess.Popen", return_value=child), \
                patch("KHS_KMJ_vision.alsa_input.selectors.DefaultSelector", return_value=selector), \
                patch("KHS_KMJ_vision.alsa_input.monotonic", side_effect=[0, 1]):
            with self.assertRaisesRegex(ConnectionError, "PCM"):
                mic._session()
        self.assertIsNone(mic._process)
        waiting = threading.Event()
        mic = AlsaMicrophone("arecord", "stable-id", 640, lambda _: None,
                             lambda *a: None, retry_s=30)
        def absent():
            waiting.set()
            raise OSError("off")
        mic._session = absent
        mic.start()
        self.assertTrue(waiting.wait(1))
        mic.close()
        self.assertFalse(mic._thread.is_alive())

    def test_disconnect_invalidates_old_audio_without_stopping_tts(self):
        service = AudioService({}, lambda _: None)
        service._ready = service._listening = True
        service._pcm.put_nowait((time.monotonic(), b"old"))
        generation = service._input_generation
        service._microphone_state(False, "USB removed")
        self.assertEqual(service.input_state, "WAITING_MIC")
        self.assertTrue(service._pcm.empty())
        self.assertGreater(service._input_generation, generation)
        self.assertFalse(service._stop.is_set())
        service._started = True
        self.assertTrue(service.say("마이크가 없어도 음성 출력은 유지"))
        service._microphone_state(True)
        self.assertFalse(service.input_ready)

    def test_alsa_start_accepts_microphone_off_without_opening_portaudio(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "ggml-tiny.bin"
            model.write_bytes(b"test")
            service = AudioService({"enabled": True, "input_backend": "alsa",
                                    "whisper_model": str(model), "cache_dir": directory}, lambda _: None)
            with patch("KHS_KMJ_vision.audio.shutil.which", side_effect=lambda value: value), \
                    patch("KHS_KMJ_vision.audio.subprocess.run", return_value=SimpleNamespace(stdout="", stderr="")), \
                    patch.object(service, "_init_supertonic"), \
                    patch("KHS_KMJ_vision.alsa_input.AlsaMicrophone") as microphone:
                service.start()
                try:
                    self.assertEqual(service.input_state, "WAITING_MIC")
                    self.assertFalse(service._stop.is_set())
                    microphone.return_value.start.assert_called_once()
                finally:
                    service.close()
                microphone.return_value.close.assert_called_once()

    def test_offline_ble_suppresses_tones_but_keeps_voice_input(self):
        service = AudioService({"echo_cooldown_ms": 0}, lambda _: None)
        service._play_ready_tone = Mock()
        service.set_ready_tone_enabled(False)
        for _ in range(2):
            service._speaking = True
            service._finish_output()
            self.assertTrue(service.input_ready)
        service._play_ready_tone.assert_not_called()
        service.set_ready_tone_enabled(True)
        service._play_ready_tone.assert_not_called()  # Connection alone never queues a beep.
        service._finish_output()
        service._play_ready_tone.assert_called_once()


class ConnectionFeedbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_bounded_all_finger_pulse_then_stop_and_no_dry_run_output(self):
        link = Link("address")
        link.client = SimpleNamespace(is_connected=True, write_gatt_char=AsyncMock())
        link.initial_status = SimpleNamespace(flags=1)
        with patch("ble_sender.link.asyncio.sleep", new_callable=AsyncMock):
            self.assertTrue(await link.connection_feedback())
        calls = link.client.write_gatt_char.call_args_list
        self.assertEqual(len(calls), 2)
        first, second = (PACKET.unpack(call.args[1]) for call in calls)
        self.assertEqual(first[0:4], (2, 2, 0, 300))
        self.assertEqual(first[4:9], (2, 0, 0, 0, 180))
        self.assertEqual(first[8:13], (180,) * 5)
        self.assertEqual(second[1], 1)  # STOP
        link.client.write_gatt_char.reset_mock()
        link.initial_status.flags = 17
        self.assertFalse(await link.connection_feedback())
        link.client.write_gatt_char.assert_not_called()


if __name__ == "__main__":
    unittest.main()
