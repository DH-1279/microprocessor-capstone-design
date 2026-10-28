"""Exercise detector.py's real loop with synthetic capture/audio/inference."""

from contextlib import redirect_stderr, redirect_stdout
import asyncio
import importlib.util
import io
import math
from pathlib import Path
import socket
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ble_sender.integration import DetectorBLE, detector_arguments
from ble_sender.protocol import Channel, encode
from KHS_KMJ_vision import audio, hand_tracker, speech
from haptic_mvp import vision
from haptic_mvp.models import Detection
from test_sender import hand


class Frame:
    shape = (720, 1280, 3)

    def copy(self):
        return self


class FakeCV(ModuleType):
    COLOR_RGB2BGR = FONT_HERSHEY_SIMPLEX = LINE_AA = 0
    WINDOW_NORMAL = WND_PROP_FULLSCREEN = 0
    WINDOW_FULLSCREEN = 1

    def __init__(self, name):
        super().__init__(name)
        self.window_requests = []

    def namedWindow(self, name, flags=0):
        self.window_requests.append(("window", name, flags))

    def setWindowProperty(self, name, key, value):
        self.window_requests.append(("property", name, key, value))

    def cvtColor(self, frame, code):
        return frame

    def waitKey(self, delay):
        return 0

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def load_detector():
    path = Path(__file__).resolve().parents[2] / "KHS_KMJ_vision/detector.py"
    spec = importlib.util.spec_from_file_location("ble_test_detector", path)
    module = importlib.util.module_from_spec(spec)
    original_decode = vision.decode_hailo_nms
    with patch.dict(sys.modules, cv2=FakeCV("cv2"), hand_tracker=hand_tracker, audio=audio, speech=speech):
        try:
            spec.loader.exec_module(module)
        finally:
            vision.decode_hailo_nms = original_decode
    return module


def landmarks(flip=False):
    return [SimpleNamespace(x=x, y=y) for x, y in hand(flip=flip)]


def result(*names, score=0.95):
    return SimpleNamespace(
        hand_landmarks=[landmarks() for _ in names],
        handedness=[[SimpleNamespace(category_name=n, score=score)] for n in names],
    )


class ObservedHandTests(unittest.TestCase):
    def tracker(self):
        tracker = hand_tracker.HandTracker.__new__(hand_tracker.HandTracker)
        tracker.selected, tracker.worn_hand, tracker.threshold = "Any", "Left", 0.5
        tracker.stabilizer = hand_tracker.HandStabilizer(selected="Any")
        tracker.last_timestamp, tracker.last_frame_id = -1, None
        tracker.last_result = tracker.last_landmarks = tracker.last_observed_landmarks = None
        tracker.mp = SimpleNamespace(Image=lambda **kw: kw, ImageFormat=SimpleNamespace(SRGB=0))
        tracker.tracker = SimpleNamespace(detect_for_video=lambda *a: tracker.reading)
        return tracker

    def test_fresh_single_hand_is_exported_even_if_reported_right(self):
        tracker = self.tracker()
        tracker.reading = result("Right")
        self.assertIsNotNone(tracker.detect(object(), 1.0, 1))
        self.assertIs(tracker.last_observed_landmarks, tracker.reading.hand_landmarks[0])

    def test_held_hand_is_never_exported_as_a_new_observation(self):
        tracker = self.tracker()
        tracker.reading = result("Left")
        tracker.detect(object(), 1.0, 1)
        tracker.reading = result()
        self.assertIsNotNone(tracker.detect(object(), 1.1, 2))  # Existing display hold is preserved.
        self.assertIsNone(tracker.last_observed_landmarks)

    def test_ambiguous_low_score_and_invalid_fingertip_do_not_export(self):
        for reading in (result("Left", "Right"), result("Left", score=0.1), result("Left")):
            if len(reading.hand_landmarks) == 1 and reading.handedness[0][0].score > 0.5:
                reading.hand_landmarks[0][8].x = math.nan
            tracker = self.tracker()
            tracker.reading = reading
            tracker.detect(object(), 1.0, 1)
            self.assertIsNone(tracker.last_observed_landmarks)

    def test_reset_and_repeated_frame_clear_export(self):
        tracker = self.tracker()
        tracker.reading = result("Left")
        tracker.detect(object(), 1.0, 1)
        with self.assertRaises(ValueError):
            tracker.detect(object(), 1.0, 1)
        self.assertIsNone(tracker.last_observed_landmarks)
        tracker.reset()
        self.assertIsNone(tracker.last_observed_landmarks)


CUP = Detection("cup", (0.10, 0.50, 0.20, 0.60), 0.95)


class BridgeProbe:
    def __init__(self, args):
        self.sent, self.closed = [], False

    def start(self):
        pass

    def check(self):
        return True

    def stop(self):
        self.sent.append(None)

    def publish(self, *, active, **observation):
        self.sent.append(observation if active else None)

    def close(self):
        self.stop()
        self.closed = True


def replay(rows, *, real_runtime=False, fail_audio=False):
    module = load_detector()
    position = [-1]
    clock = [0.0]
    spoken, states = [], []
    state = SimpleNamespace(bridge=None, audio_closed=False)
    volume = Mock(return_value=SimpleNamespace(returncode=0))

    class Audio:
        def __init__(self, config, on_text):
            self.on_text = on_text
            state.audio = self

        def start(self):
            if fail_audio:
                raise RuntimeError("mock audio startup failed")

        def say(self, text):
            spoken.append(text)
            return True

        def close(self):
            state.audio_closed = True

    class Camera:
        def __init__(self, config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            position[0] += 1
            if position[0] == len(rows):
                return None
            row = rows[position[0]]
            if real_runtime:
                time.sleep(0.04)
                clock[0] = time.monotonic()
            else:
                clock[0] = 1.0 + position[0] / 15
            if row.get("text"):
                state.audio.on_text(row["text"])
            return Frame(), clock[0] - row.get("age", 0)

    class Detector(Camera):
        def detect(self, frame):
            if rows[position[0]].get("raise"):
                raise RuntimeError("mock inference failed")
            return rows[position[0]].get("objects", (CUP,))

    class Tracker(Camera):
        def detect(self, frame, captured):
            row = rows[position[0]]
            self.last_landmarks = row.get("points", landmarks())
            self.last_observed_landmarks = self.last_landmarks if row.get("valid", True) else None
            return object() if row.get("valid", True) or row.get("held") else None

    def bridge(args):
        state.bridge = DetectorBLE(args) if real_runtime else BridgeProbe(args)
        return state.bridge

    def overlay(frame, text, *args, **kw):
        if text.startswith("State: "):
            states.append(text.split(" | ")[0][7:])

    argv = ["--no-console", "--stable-frames", "1", "--mode", "3"]
    if real_runtime:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            argv += ["--port", str(probe.getsockname()[1])]
    updates = dict(
        AudioService=Audio,
        CameraSource=Camera,
        HailoDetector=Detector,
        HandTracker=Tracker,
        DetectorBLE=bridge,
        draw_hand=lambda *a: None,
        put_readable_text=overlay,
        subprocess=SimpleNamespace(run=volume, DEVNULL=-3),
    )
    if not real_runtime:
        updates["time"] = SimpleNamespace(monotonic=lambda: clock[0])
    output = io.StringIO()
    with patch.multiple(module, **updates), redirect_stdout(output), redirect_stderr(output):
        exit_code = module.main(argv)
    return SimpleNamespace(
        bridge=state.bridge,
        spoken=spoken,
        states=states,
        audio_closed=state.audio_closed,
        code=exit_code,
        log=output.getvalue(),
        volume_calls=volume.call_args_list,
        window_requests=module.cv2.window_requests,
    )


class DetectorLoopTests(unittest.TestCase):
    def test_wake_lock_cancel_and_relock_use_current_frame_and_new_lock_id(self):
        run = replay(
            [
                {},
                {"text": "캡스톤"},
                {"text": "컵"},
                {},
                {"text": "취소"},
                {"text": "컵"},
                {"text": "캡스톤"},
                {"text": "컵"},
                {},
            ]
        )
        active = [record for record in run.bridge.sent if record is not None]
        self.assertEqual([record["frame_id"] for record in active], [3, 4, 8, 9])
        self.assertEqual(active[0]["target_id"], active[1]["target_id"])
        self.assertNotEqual(active[0]["target_id"], active[2]["target_id"])
        self.assertEqual(active[0]["size"], (1280, 720))
        self.assertEqual(run.states[-1], "SEARCHING")
        self.assertTrue(run.bridge.closed and run.audio_closed)
        self.assertEqual(run.code, 0)

    def test_cancel_requires_wake_word_before_selecting_another_target(self):
        run = replay([{"text": "캡스톤"}, {"text": "컵"}, {"text": "취소"}, {"text": "컵"}, {}])
        active = [record for record in run.bridge.sent if record is not None]
        self.assertEqual([record["frame_id"] for record in active], [2])
        self.assertEqual(run.states[-1], "WAIT_WAKE")
        self.assertTrue(all(record is None for record in run.bridge.sent[2:]))
        self.assertTrue(any("다시 부르시려면 캡스톤" in text for text in run.spoken))

    def test_latest_fullscreen_and_volume_setup_are_preserved(self):
        run = replay([{}])
        self.assertIn(("property", "Capstone Vision Controller", 0, 1), run.window_requests)
        self.assertEqual(
            [call.args[0] for call in run.volume_calls],
            [["amixer", "-c", "3", "set", "PCM", "100%"], ["amixer", "set", "Master", "100%"]],
        )
        self.assertTrue(all(call.kwargs["timeout"] == 2 for call in run.volume_calls))

    def test_lost_held_stale_hand_or_missing_target_send_stop(self):
        for invalid in ({"valid": False}, {"valid": False, "held": True}, {"age": 1}, {"objects": []}):
            run = replay([{"text": "캡스톤"}, {"text": "컵"}, invalid])
            records = run.bridge.sent
            self.assertIsNotNone(records[1])
            self.assertIsNone(records[2])
            self.assertFalse(any("찾았습니다" in text for text in run.spoken))

    def test_reach_sends_stop_before_return_to_wake(self):
        near = landmarks()
        near[9] = SimpleNamespace(x=0.15, y=0.55)
        run = replay([{"text": "캡스톤"}, {"text": "컵"}] + [{"points": near} for _ in range(5)] + [{}])
        self.assertTrue(any("찾았습니다" in text for text in run.spoken))
        self.assertEqual(run.states[-1], "WAIT_WAKE")
        self.assertTrue(all(record is None for record in run.bridge.sent[-3:]))

    def test_invalid_hand_does_not_trigger_reach(self):
        near = landmarks()
        near[9] = SimpleNamespace(x=0.15, y=0.55)
        run = replay(
            [{"text": "캡스톤"}, {"text": "컵"}] + [{"points": near, "valid": False, "held": True} for _ in range(6)]
        )
        self.assertFalse(any("찾았습니다" in text for text in run.spoken))
        self.assertEqual(run.states[-1], "SEARCHING")

    def test_exception_and_audio_startup_failure_close_ble_and_audio(self):
        for rows, fail_audio in (([{"raise": True}], False), ([], True)):
            run = replay(rows, fail_audio=fail_audio)
            self.assertEqual(run.code, 1)
            self.assertTrue(run.bridge.closed and run.audio_closed)

    def test_real_background_loop_receives_detector_udp_and_writes_v2_packets(self):
        sent, packets = [], []

        class FakeLink:
            def __init__(self, address=None):
                pass

            async def __aenter__(self):
                self.sequence = 0
                await self.send(None)
                return self

            async def send(self, cue):
                sent.append(cue)
                packets.append(encode(self.sequence, cue))
                self.sequence += 1

            async def __aexit__(self, *args):
                await self.send(None)

        with patch("ble_sender.cli.Link", FakeLink):
            run = replay(
                [{"text": "캡스톤"}, {"text": "컵"}, {}, {}, {"valid": False}, {"text": "취소"}, {}], real_runtime=True
            )
        self.assertEqual(run.code, 0, run.log)
        self.assertTrue(any(cue is not None and cue.channel == Channel.TH for cue in sent))
        self.assertIsNone(sent[-1])
        self.assertTrue(all(len(packet) == 16 for packet in packets))
        self.assertFalse(run.bridge.thread.is_alive())


class LifecycleTests(unittest.TestCase):
    def test_no_ble_is_explicit_and_starts_no_thread(self):
        runtime = DetectorBLE(detector_arguments(["--no-ble"]))
        runtime.start()
        self.assertTrue(runtime.check())
        self.assertIsNone(runtime.thread)
        runtime.close()

    def test_connection_failure_fails_startup_without_publisher(self):
        async def failing(args, **kwargs):
            raise OSError("ESP missing")

        runtime = DetectorBLE(detector_arguments(["--no-console"]))
        with patch("ble_sender.integration.run", failing):
            with self.assertRaisesRegex(RuntimeError, "ESP missing"):
                runtime.start()
        runtime.close()
        self.assertIsNone(runtime.publisher)
        self.assertFalse(runtime.thread.is_alive())

    def test_owner_close_cancels_a_pending_connection(self):
        entered = threading.Event()

        class PendingLink:
            def __init__(self, address=None):
                pass

            async def __aenter__(self):
                entered.set()
                await asyncio.sleep(30)

            async def __aexit__(self, *args):
                pass

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        runtime = DetectorBLE(detector_arguments(["--no-console", "--port", str(port)]))
        with patch("ble_sender.cli.Link", PendingLink):
            runtime.thread = threading.Thread(target=runtime._worker, daemon=True)
            runtime.thread.start()
            try:
                self.assertTrue(entered.wait(2))
            finally:
                runtime.close()
        self.assertFalse(runtime.thread.is_alive())
        self.assertIsNone(runtime.error)


if __name__ == "__main__":
    unittest.main()
