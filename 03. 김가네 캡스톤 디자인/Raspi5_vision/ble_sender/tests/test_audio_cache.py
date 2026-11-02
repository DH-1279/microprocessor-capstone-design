import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from KHS_KMJ_vision.audio import AudioService, audio_config, tts_cache_path
from KHS_KMJ_vision.dialogue import Announcements, HAPTIC_HELP, target_announcement_parts
from KHS_KMJ_vision.prepare_audio import main, prompt_catalog
from KHS_KMJ_vision.voice_profile import TARGET_MAPPING


def write_wav(path):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(b"\x01\x00" * 160)


class CachedVoiceTests(unittest.TestCase):
    def test_prepare_command_reuses_all_clips_without_opening_hardware(self):
        engine = Mock(voice_style_names=["M1"])
        engine.synthesize.return_value = (b"unused", 0.01)
        engine.save_audio.side_effect = lambda data, path: write_wav(path)
        tts = Mock(return_value=engine)
        with tempfile.TemporaryDirectory() as directory, \
                patch("KHS_KMJ_vision.prepare_audio.detector_audio_config", return_value={"cache_dir": directory}), \
                patch.dict(sys.modules, supertonic=SimpleNamespace(TTS=tts)), \
                patch.object(AudioService, "start", side_effect=AssertionError("hardware opened")), \
                patch.object(AudioService, "_run", side_effect=AssertionError("playback opened")), \
                patch("builtins.print"):
            self.assertEqual(main(["--check"]), 1)
            tts.assert_not_called()
            self.assertEqual(main([]), 0)
            count = len(prompt_catalog())
            self.assertEqual(engine.synthesize.call_count, count)
            self.assertEqual(main(["--check"]), 0)
            self.assertEqual(main([]), 0)
            self.assertEqual(engine.synthesize.call_count, count)
            # A damaged entry must be reported and regenerated, not reused.
            path = tts_cache_path(audio_config({"cache_dir": directory}), HAPTIC_HELP)
            path.write_bytes(b"corrupt" * 20)
            self.assertEqual(main(["--check"]), 1)
            self.assertEqual(main([]), 0)
            self.assertEqual(engine.synthesize.call_count, count + 1)

    def test_prepared_target_turn_plays_without_synthesis_and_one_ready_tone(self):
        with tempfile.TemporaryDirectory() as directory:
            service = AudioService({"cache_dir": directory, "echo_cooldown_ms": 0}, lambda _: None)
            parts = target_announcement_parts("지갑", True)
            for text in parts:
                write_wav(tts_cache_path(service.config, text))
            trace = []
            service._init_supertonic = Mock(side_effect=AssertionError("unexpected synthesis"))
            def play(command, timeout):
                self.assertFalse(service.input_ready)
                trace.append(command[-1])
            service._run = play
            def tone():
                self.assertFalse(service.input_ready)
                trace.append("TONE")
            service._play_ready_tone = tone
            service._started = True
            self.assertTrue(service.say_parts(parts))
            self.assertTrue(service.say_parts(parts))
            worker = threading.Thread(target=service._speak_worker)
            worker.start()
            try:
                deadline = time.monotonic() + 2
                while not service.input_ready and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertTrue(service.input_ready)
                self.assertEqual(trace, [str(tts_cache_path(service.config, t)) for t in parts] + ["TONE"])
            finally:
                service._stop.set()
                worker.join(1)
            self.assertFalse(worker.is_alive())

    def test_catalog_covers_each_target_and_group_retries_atomically(self):
        catalog = set(prompt_catalog())
        for target in TARGET_MAPPING.values():
            for first in (True, False):
                self.assertTrue(set(target_announcement_parts(target["name"], first)) <= catalog)
        audio = Mock()
        audio.say_parts.side_effect = [False, True]
        announcements = Announcements(audio)
        parts = target_announcement_parts("지갑", True)
        announcements.say_parts(parts)
        announcements.flush()
        self.assertFalse(announcements.pending)
        self.assertEqual(audio.say_parts.call_count, 2)
        audio.say.assert_not_called()

    def test_tone_settings_have_separate_cache_and_validated_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            quiet = AudioService({"cache_dir": directory, "ready_tone_ms": 100, "ready_tone_amplitude": 3000}, lambda _: None)
            normal = AudioService({"cache_dir": directory}, lambda _: None)
            self.assertNotEqual(quiet.prepare_ready_tone(), normal.prepare_ready_tone())
            with wave.open(str(quiet.ready_tone_path())) as sound:
                self.assertEqual(sound.getnframes(), 1600)
        for settings in ({"ready_tone_ms": 1}, {"ready_tone_ms": 250.5},
                         {"ready_tone_amplitude": 40000}, {"ready_tone_amplitude": True}):
            with self.assertRaises(ValueError):
                audio_config(settings)


if __name__ == "__main__":
    unittest.main()
