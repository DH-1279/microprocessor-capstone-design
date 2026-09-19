"""Audio contracts tested without a microphone, speaker, Whisper model or numpy."""

from array import array
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import wave

from haptic_mvp.audio import (
    AudioService, UtteranceDetector, audio_config, build_whisper_command,
    pcm16_rms, tts_cache_path,
)


def pcm(value=4000, milliseconds=20):
    samples = array("h", [value] * (16 * milliseconds))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def write_wav(path, rate=16000, channels=1, seconds=0.1):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\0\0" * int(rate * seconds) * channels)


class VadTests(unittest.TestCase):
    def setUp(self):
        self.vad = UtteranceDetector({"min_speech_ms": 100, "silence_ms": 100,
                                       "max_utterance_ms": 500, "pre_roll_ms": 40})

    def test_pcm16_rms_and_invalid_input(self):
        self.assertEqual(pcm16_rms(b""), 0)
        self.assertEqual(pcm16_rms(pcm(0)), 0)
        self.assertEqual(pcm16_rms(pcm(-32768)), 1)
        self.assertAlmostEqual(pcm16_rms(pcm(16384)), 0.5)
        with self.assertRaises(ValueError):
            pcm16_rms(b"\0")

    def test_silence_never_generates_an_utterance(self):
        for _ in range(1000):
            self.assertIsNone(self.vad.feed(pcm(0)))
        self.assertFalse(self.vad.active)
        self.assertLessEqual(self.vad._pre_samples, 640)

    def test_speech_requires_trailing_silence_and_preserves_preroll(self):
        for _ in range(8):
            self.vad.feed(pcm(0))
        for _ in range(5):
            self.assertIsNone(self.vad.feed(pcm()))
        self.assertTrue(self.vad.active)
        for _ in range(4):
            self.assertIsNone(self.vad.feed(pcm(0)))
        result = self.vad.feed(pcm(0))
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 16 * (40 + 100 + 100) * 2)
        self.assertFalse(self.vad.active)

    def test_short_click_is_not_a_command(self):
        self.vad.feed(pcm())
        for _ in range(5):
            self.assertIsNone(self.vad.feed(pcm(0)))
        self.assertFalse(self.vad.active)

    def test_silence_between_clicks_does_not_count_as_speech(self):
        self.vad.feed(pcm())
        for _ in range(4):
            self.vad.feed(pcm(0))
        self.vad.feed(pcm())
        for _ in range(5):
            self.assertIsNone(self.vad.feed(pcm(0)))

    def test_continuous_noise_has_a_finite_buffer(self):
        result = None
        for _ in range(25):
            result = self.vad.feed(pcm())
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 16000)
        self.assertFalse(self.vad.active)

    def test_mute_drops_partial_utterance_and_preroll(self):
        for _ in range(5):
            self.vad.feed(pcm())
        self.assertTrue(self.vad.active)
        self.assertIsNone(self.vad.feed(pcm(), muted=True))
        self.assertFalse(self.vad.active)
        for _ in range(10):
            self.assertIsNone(self.vad.feed(pcm(0)))


class ConfigurationTests(unittest.TestCase):
    def test_reject_english_only_whisper_models(self):
        for filename in ("ggml-base.en.bin", "ggml-tiny.en-q5_1.bin", "small.en"):
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "multilingual"):
                audio_config({"whisper_model": filename})
        audio_config({"whisper_model": "models/ggml-tiny.bin"})

    def test_limits_reject_unbounded_or_invalid_recordings(self):
        for settings in ({"rms_threshold": float("nan")}, {"max_utterance_ms": 999999},
                         {"sample_rate": 44100}, {"whisper_threads": 2.5},
                         {"min_speech_ms": 1000, "max_utterance_ms": 1000},
                         {"enabled": "false"}, {"stt_timeout_s": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                audio_config(settings)

    def test_command_handles_spaces_as_single_arguments_and_korean_language(self):
        cfg = {"whisper_executable": "/home/pi/my tools/whisper-cli",
               "whisper_model": "/home/pi/models/ggml tiny.bin"}
        command = build_whisper_command(cfg, Path("recording 한글.wav"), Path("result"))
        self.assertEqual(command[0], str(Path(cfg["whisper_executable"]).expanduser()))
        self.assertEqual(command[command.index("-m") + 1], str(Path(cfg["whisper_model"]).expanduser()))
        self.assertEqual(command[command.index("-l") + 1], "ko")
        self.assertIn("recording 한글.wav", command)
        self.assertNotIn("-tr", command)

    def test_cache_is_path_safe_and_changes_with_voice_rate(self):
        cfg = audio_config({"cache_dir": "cache"})
        path = tts_cache_path(cfg, "../../$(touch danger); 컵")
        self.assertEqual(path.parent, Path("cache"))
        self.assertEqual(len(path.stem), 64)
        self.assertNotEqual(path, tts_cache_path({**cfg, "tts_rate": 160}, "../../$(touch danger); 컵"))


class AudioServiceTests(unittest.TestCase):
    def test_disabled_audio_never_imports_hardware(self):
        service = AudioService({"enabled": False}, lambda text: None)
        with patch("haptic_mvp.audio.importlib.import_module") as importer:
            service.start()
        importer.assert_not_called()
        self.assertFalse(service.say("컵"))
        self.assertFalse(service.listening)
        service.close()

    def test_start_reports_missing_binary_before_opening_mic(self):
        service = AudioService({"enabled": True}, lambda text: None)
        with patch("haptic_mvp.audio.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "executable missing"):
                service.start()

    def test_bounded_speech_respects_user_voice_and_busy_output(self):
        service = AudioService({}, lambda text: None)
        service._started = True
        service._listening = True
        self.assertFalse(service.say("컵이 보입니다"))
        service._listening = False
        service._transcribing = True
        self.assertFalse(service.say("컵이 보입니다"))
        service._transcribing = False
        self.assertTrue(service.say("컵이 보입니다"))
        self.assertTrue(service.speaking)
        self.assertFalse(service.say("다른 반복 안내"))
        self.assertEqual(service._speech.qsize(), 1)
        service.close()

    def test_capture_gate_discards_echo_and_cooldown(self):
        service = AudioService({}, lambda text: None)
        service._speaking = True
        service._capture(pcm(), 320, None, None)
        self.assertTrue(service._pcm.empty())
        service._speaking = False
        service._mute_until = time.monotonic() + 10
        service._capture(pcm(), 320, None, None)
        self.assertTrue(service._pcm.empty())
        service._mute_until = 0
        service._capture(pcm(), 320, None, None)
        self.assertEqual(service._pcm.qsize(), 1)
        service.close()

    def test_worker_delivers_completed_utterance_and_drops_stale_audio(self):
        received = []
        service = AudioService({"min_speech_ms": 100, "silence_ms": 100,
                                "max_utterance_ms": 500, "pre_roll_ms": 0}, received.append)
        # Old recordings must not become a new target command after a blocked worker.
        for _ in range(10):
            service._pcm.put_nowait((time.monotonic() - 2, pcm()))
        now = time.monotonic()
        for _ in range(5):
            service._pcm.put_nowait((now, pcm()))
        for _ in range(5):
            service._pcm.put_nowait((now, pcm(0)))
        worker = threading.Thread(target=service._listen_worker)
        service._threads.append(worker)
        with patch.object(service, "transcribe_wav", return_value="컵 찾아줘") as transcribe:
            worker.start()
            deadline = time.monotonic() + 2
            while not received and time.monotonic() < deadline:
                time.sleep(0.01)
            service.close()
        self.assertEqual(received, ["컵 찾아줘"])
        transcribe.assert_called_once()
        self.assertFalse(service.listening)
        self.assertFalse(worker.is_alive())

    def test_tts_failure_unmutes_and_reports_to_main_loop(self):
        service = AudioService({}, lambda text: None)
        service._started = True
        self.assertTrue(service.say("컵을 안내합니다"))
        with patch.object(service, "_play", side_effect=TimeoutError("speaker timeout")):
            service._speak_worker()
        self.assertFalse(service.speaking)
        self.assertTrue(service._stop.is_set())
        self.assertIn("speaker timeout", service.get_error())
        self.assertFalse(service.say("재시도"))
        service.close()

    def test_session_change_drops_inflight_stt_result(self):
        received = []
        started, release = threading.Event(), threading.Event()
        service = AudioService({"min_speech_ms": 100, "silence_ms": 100,
                                "max_utterance_ms": 500, "pre_roll_ms": 0}, received.append)
        for _ in range(5):
            service._pcm.put_nowait((time.monotonic(), pcm()))
        for _ in range(5):
            service._pcm.put_nowait((time.monotonic(), pcm(0)))
        def slow_transcribe(path):
            started.set()
            release.wait(2)
            return "컵 찾아줘"
        worker = threading.Thread(target=service._listen_worker)
        service._threads.append(worker)
        with patch.object(service, "transcribe_wav", side_effect=slow_transcribe):
            worker.start()
            self.assertTrue(started.wait(2))
            service.discard_input()
            release.set()
            deadline = time.monotonic() + 2
            while service.listening and time.monotonic() < deadline:
                time.sleep(0.01)
            service.close()
        self.assertEqual(received, [])
        self.assertFalse(worker.is_alive())

    def test_shutdown_closes_microphone_stream_even_after_abort_error(self):
        service = AudioService({}, lambda text: None)
        class BrokenStream:
            closed = False
            def abort(self):
                raise RuntimeError("device unplugged")
            def close(self):
                self.closed = True
        stream = BrokenStream()
        service._stream = stream
        service.close()
        self.assertTrue(stream.closed)
        self.assertIn("device unplugged", service.get_error())

    def test_unexpected_mic_stop_is_reported_but_normal_close_is_not(self):
        service = AudioService({}, lambda text: None)
        service._input_finished()
        self.assertIn("stopped unexpectedly", service.get_error())
        self.assertTrue(service._stop.is_set())
        service.close()
        service._input_finished()
        self.assertIsNone(service.get_error())

    def test_transcribe_validates_format_before_running(self):
        service = AudioService({}, lambda text: None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            write_wav(path, rate=44100)
            with patch.object(service, "_run") as run, self.assertRaises(ValueError):
                service.transcribe_wav(path)
            run.assert_not_called()

    def test_transcribe_reads_output_file_not_console_diagnostics(self):
        service = AudioService({}, lambda text: None)
        def fake_run(command, timeout):
            out = Path(command[command.index("-of") + 1]).with_suffix(".txt")
            out.write_text("\ufeff컵을\n찾아줘\n", encoding="utf-8")
            return "whisper timing diagnostics"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            write_wav(path)
            with patch.object(service, "_run", side_effect=fake_run):
                self.assertEqual(service.transcribe_wav(path), "컵을 찾아줘")

    def test_missing_transcript_is_an_explicit_failure(self):
        service = AudioService({}, lambda text: None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            write_wav(path)
            with patch.object(service, "_run"), self.assertRaisesRegex(RuntimeError, "transcript"):
                service.transcribe_wav(path)

    def test_tts_uses_stdin_then_reuses_cached_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            service = AudioService({"cache_dir": directory, "output_device": "plughw:CARD=USB,DEV=0"},
                                   lambda text: None)
            text = "--help; $(touch bad) 컵을 안내하겠습니다"
            def fake_run(command, timeout, input_text=None):
                if "-w" in command:
                    self.assertEqual(input_text, text)
                    self.assertNotIn(text, command)
                    self.assertIn("--stdin", command)
                    write_wav(Path(command[command.index("-w") + 1]))
                return ""
            with patch.object(service, "_run", side_effect=fake_run) as run:
                service._play(text)
                service._play(text)
            self.assertEqual(run.call_count, 3)  # One synth, two playback operations.
            self.assertEqual(run.call_args.args[0][2:4], ["-D", "plughw:CARD=USB,DEV=0"])

    def test_failed_synthesis_does_not_create_cache_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            service = AudioService({"cache_dir": directory}, lambda text: None)
            with patch.object(service, "_run", side_effect=TimeoutError("tts timeout")):
                with self.assertRaises(TimeoutError):
                    service._play("컵")
            self.assertFalse(tts_cache_path(service.config, "컵").exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_child_timeout_kills_and_reaps_process(self):
        service = AudioService({}, lambda text: None)
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            service._run([sys.executable, "-c", "import time; time.sleep(10)"], 0.15)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(service._processes, set())

    def test_process_error_includes_diagnostic(self):
        service = AudioService({}, lambda text: None)
        with self.assertRaisesRegex(RuntimeError, "microphone missing"):
            service._run([sys.executable, "-c", "import sys; sys.stderr.write('microphone missing'); sys.exit(3)"], 2)
        self.assertEqual(service._processes, set())

    def test_close_cancels_a_running_child(self):
        service = AudioService({}, lambda text: None)
        outcome = []
        def work():
            try:
                service._run([sys.executable, "-c", "import time; time.sleep(10)"], 20)
            except Exception as exc:
                outcome.append(type(exc).__name__)
        worker = threading.Thread(target=work)
        service._threads.append(worker)
        worker.start()
        deadline = time.monotonic() + 2
        while not service._processes and time.monotonic() < deadline:
            time.sleep(0.01)
        service.close()
        self.assertFalse(worker.is_alive())
        self.assertEqual(service._processes, set())
        self.assertTrue(outcome)

    def test_error_reporting_is_nonblocking_and_bounded(self):
        service = AudioService({}, lambda text: None)
        self.assertIsNone(service.get_error())
        for number in range(50):
            service._report(str(number))
        self.assertEqual(service._errors.qsize(), 8)
        messages = [service.get_error() for _ in range(8)]
        self.assertEqual(messages[-1], "49")
        self.assertIsNone(service.get_error())


if __name__ == "__main__":
    unittest.main()
