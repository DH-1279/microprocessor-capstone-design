"""Pure geometry/selection tests plus adapter tests with SDK doubles.

No camera, Hailo or MediaPipe hardware is implied by passing these tests.
"""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from haptic_mvp.vision import (
    CameraSource, HailoDetector, LatestFrames, decode_hailo_nms, hand_from_result,
    letterbox_geometry, preprocess_rgb, read_labels, restore_box, run_vision, validate_rgb,
)


class GeometryTests(unittest.TestCase):
    def test_landscape_letterbox_restores_full_original_frame(self):
        geometry = letterbox_geometry(640, 480, 640, 640)
        self.assertEqual(geometry.top, 80)
        self.assertEqual(restore_box((0.125, 0, 0.875, 1), geometry), (0, 0, 1, 1))

    def test_portrait_letterbox_restores_full_original_frame(self):
        geometry = letterbox_geometry(480, 640, 640, 640)
        self.assertEqual(restore_box((0, 0.125, 1, 0.875), geometry), (0, 0, 1, 1))

    def test_rounding_inverse_uses_actual_resized_dimensions(self):
        g = letterbox_geometry(853, 479, 640, 640)
        original = (0.23, 0.17, 0.84, 0.91)
        x1, y1, x2, y2 = original
        padded = ((y1 * g.resized_height + g.top) / g.model_height,
                  (x1 * g.resized_width + g.left) / g.model_width,
                  (y2 * g.resized_height + g.top) / g.model_height,
                  (x2 * g.resized_width + g.left) / g.model_width)
        for actual, expected in zip(restore_box(padded, g), original):
            self.assertAlmostEqual(actual, expected)

    def test_padding_only_box_is_not_a_real_object(self):
        self.assertIsNone(restore_box((0, 0.2, 0.1, 0.5), letterbox_geometry(640, 480, 640, 640)))

    def test_padding_overlap_is_clipped_to_frame(self):
        box = restore_box((0, 0.2, 0.5, 0.8), letterbox_geometry(640, 480, 640, 640))
        self.assertEqual(box, (0.2, 0, 0.8, 0.5))

    def test_invalid_coordinates_are_rejected(self):
        g = letterbox_geometry(640, 480, 640, 640)
        for invalid in ((0, 0, float("nan"), 1), (0, 0, 2, 1), (0.8, 0, 0.2, 1)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                restore_box(invalid, g)

    def test_invalid_size_is_rejected(self):
        with self.assertRaises(ValueError):
            letterbox_geometry(0, 480, 640, 640)

    def test_nms_class_mapping_score_filter_and_sort(self):
        raw = [[[0.2, 0.1, 0.6, 0.8, 0.7], [0.1, 0.1, 0.3, 0.4, 0.1]],
               [[0.5, 0.3, 0.7, 0.6, 0.95]]]
        result = decode_hailo_nms(raw, ("cup", "bottle"), letterbox_geometry(640, 640, 640, 640))
        self.assertEqual([item.label for item in result], ["bottle", "cup"])
        self.assertEqual(result[1].box, (0.1, 0.2, 0.8, 0.6))

    def test_nms_limit(self):
        raw = [[[0, 0, 1, 1, 0.9], [0, 0, 0.5, 0.5, 0.8]]]
        self.assertEqual(len(decode_hailo_nms(raw, ("cup",), letterbox_geometry(1, 1, 1, 1),
                                            max_objects=1)), 1)

    def test_nms_rejects_missing_classes_extra_batch_raw_heads(self):
        for raw in ([], [[[[0, 0, 1, 1, 0.8]]]], [[[0, 0, 1, 1, 0.8, 7]]]):
            with self.subTest(raw=raw), self.assertRaises((ValueError, TypeError)):
                decode_hailo_nms(raw, ("cup",), letterbox_geometry(1, 1, 1, 1))

    def test_nms_rejects_nonfinite_confidence(self):
        with self.assertRaises(ValueError):
            decode_hailo_nms([[[0, 0, 1, 1, float("inf")]]], ("cup",),
                             letterbox_geometry(1, 1, 1, 1))

    def test_empty_class_is_valid(self):
        self.assertEqual(decode_hailo_nms([[]], ("cup",), letterbox_geometry(1, 1, 1, 1)), ())

    def test_label_order_retained_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.txt"
            path.write_text("cup\nbottle\n", encoding="utf-8")
            self.assertEqual(read_labels(str(path)), ("cup", "bottle"))
            path.write_text("cup\ncup\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_labels(str(path))


def hand_result(*names):
    return SimpleNamespace(
        hand_landmarks=[[SimpleNamespace(x=0.4, y=0.6) for _ in range(21)] for _ in names],
        handedness=[[SimpleNamespace(category_name=name, score=0.9)] for name in names])


class HandTests(unittest.TestCase):
    def test_select_requested_hand(self):
        hand = hand_from_result(hand_result("Left", "Right"), "Right", 0.6)
        self.assertEqual(hand.point, (0.4, 0.6))
        self.assertEqual(hand.handedness, "Right")

    def test_any_with_two_hands_is_ambiguous(self):
        self.assertIsNone(hand_from_result(hand_result("Left", "Right"), "Any", 0.6))

    def test_two_requested_hands_is_ambiguous(self):
        self.assertIsNone(hand_from_result(hand_result("Right", "Right"), "Right", 0.6))

    def test_wrong_hand_or_no_hand_stops(self):
        self.assertIsNone(hand_from_result(hand_result("Left"), "Right", 0.6))
        self.assertIsNone(hand_from_result(hand_result(), "Any", 0.6))

    def test_low_confidence_and_nan_landmark_do_not_guide(self):
        result = hand_result("Right")
        result.handedness[0][0].score = 0.4
        self.assertIsNone(hand_from_result(result, "Right", 0.6))
        result.handedness[0][0].score = 0.9
        result.hand_landmarks[0][5].x = float("nan")
        self.assertIsNone(hand_from_result(result, "Right", 0.6))


class PipelineTests(unittest.TestCase):
    def test_same_frame_and_capture_time_reach_both_models(self):
        rgb = SimpleNamespace(shape=(480, 640, 3))
        seen = []
        stop = threading.Event()
        detector = SimpleNamespace(detect=lambda image: seen.append(("object", image)) or ())
        tracker = SimpleNamespace(detect=lambda image, stamp: seen.append(("hand", image, stamp)))
        frames = SimpleNamespace(get=lambda _: (12, 42.0, rgb))

        def emit(observation):
            seen.append(observation)
            stop.set()

        with patch("haptic_mvp.vision.HailoDetector", return_value=nullcontext(detector)), \
             patch("haptic_mvp.vision.MediaPipeHandTracker", return_value=nullcontext(tracker)), \
             patch("haptic_mvp.vision.CameraSource", return_value=nullcontext(None)), \
             patch("haptic_mvp.vision.LatestFrames", return_value=nullcontext(frames)):
            run_vision({}, emit, stop)
        self.assertIs(seen[0][1], rgb)
        self.assertIs(seen[1][1], rgb)
        self.assertEqual(seen[1][2], 42.0)
        self.assertEqual(seen[2].captured_at, 42.0)
        self.assertEqual(seen[2].frame_id, 12)

    def test_newest_frame_replaces_old_frames(self):
        counter = [0]

        def read():
            counter[0] += 1
            return (f"frame-{counter[0]}", float(counter[0])) if counter[0] <= 5 else None

        source = SimpleNamespace(read=read, replay=False)
        with LatestFrames(source) as frames:
            self.assertTrue(frames.finished.wait(2))
            self.assertEqual(frames.get(threading.Event()), (5, 5.0, "frame-5"))
            self.assertIsNone(frames.get(threading.Event()))

    def test_wrong_aspect_does_not_emit_guidance_input(self):
        detector = SimpleNamespace(detect=lambda _: self.fail("must reject before inference"))
        frames = SimpleNamespace(get=lambda _: (1, 42.0, SimpleNamespace(shape=(480, 1280, 3))))
        with patch("haptic_mvp.vision.HailoDetector", return_value=nullcontext(detector)), \
             patch("haptic_mvp.vision.MediaPipeHandTracker", return_value=nullcontext(None)), \
             patch("haptic_mvp.vision.CameraSource", return_value=nullcontext(None)), \
             patch("haptic_mvp.vision.LatestFrames", return_value=nullcontext(frames)), \
             self.assertRaisesRegex(ValueError, "aspect"):
            run_vision({}, lambda _: self.fail("must not emit"), threading.Event())

    def test_camera_failure_propagates(self):
        def fail():
            raise OSError("unplugged")

        with LatestFrames(SimpleNamespace(read=fail)) as frames:
            self.assertTrue(frames.finished.wait(2))
            with self.assertRaisesRegex(RuntimeError, "unplugged"):
                frames.get(threading.Event())

    def test_invalid_config_fails_before_hardware_import(self):
        for config in ({"detector": "magic"}, {"hand": "Unknown"}, {"fps": 0},
                       {"score_threshold": float("nan")}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                run_vision(config, lambda _: None, threading.Event())


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy adapter checks are optional on stdlib hosts")
class ArrayAdapterTests(unittest.TestCase):
    def test_preprocess_keeps_rgb_and_pads_114(self):
        import numpy as np

        frame = np.array([[[255, 10, 30], [20, 40, 60]]], dtype=np.uint8)
        cv2 = SimpleNamespace(resize=lambda image, size: image)
        with patch.dict("sys.modules", {"cv2": cv2}):
            result, geometry = preprocess_rgb(frame, 2, 3)
        self.assertTrue(np.array_equal(result[1], frame[0]))
        self.assertTrue(np.all(result[0] == 114))
        self.assertTrue(np.all(result[2] == 114))
        self.assertEqual(geometry.top, 1)
        self.assertTrue(result.flags.c_contiguous)

    def test_camera_mirror_is_applied_to_rgb_before_both_models(self):
        import numpy as np

        rgb = np.array([[[255, 10, 30], [20, 40, 60]]], dtype=np.uint8)
        source = CameraSource({"mirror": True})
        source.camera = SimpleNamespace(capture_array=lambda _: rgb)
        result, captured_at = source.read()
        self.assertTrue(np.array_equal(result, rgb[:, ::-1]))
        self.assertGreater(captured_at, 0)

    def test_invalid_frame_formats_are_rejected(self):
        import numpy as np

        for frame in (np.zeros((2, 2)), np.zeros((2, 2, 4), dtype=np.uint8),
                      np.zeros((2, 2, 3)), np.zeros((0, 2, 3), dtype=np.uint8)):
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                validate_rgb(frame)

    def test_hailo_sdk_contract_runs_float32_nms_and_rejects_raw_outputs(self):
        import numpy as np

        entered = []
        raw = [np.array([[0.125, 0, 0.875, 1, 0.9]], dtype=np.float32)]
        output = SimpleNamespace(name="nms", shape=(1, 5, 10), is_nms=True,
                                 format=SimpleNamespace(order="NMS"),
                                 set_format_type=lambda value: entered.append(("out", value)))
        input_stream = SimpleNamespace(shape=(640, 640, 3), format=SimpleNamespace(order="NHWC"),
                                       set_format_type=lambda value: entered.append(("in", value)))
        bindings = SimpleNamespace(input=lambda: SimpleNamespace(set_buffer=lambda frame: entered.append(frame.shape)),
                                   output=lambda: SimpleNamespace(get_buffer=lambda tf_format: raw))
        configured = SimpleNamespace(create_bindings=lambda output_buffers: bindings,
                                      run=lambda items, timeout: entered.append(("run", timeout)))
        model = SimpleNamespace(inputs=[input_stream], outputs=[output], input=lambda: input_stream,
                                 output=lambda: output, set_batch_size=lambda size: None,
                                 configure=lambda: nullcontext(configured))
        physical = SimpleNamespace(identify=lambda: SimpleNamespace(device_architecture="HAILO10H"))
        device = SimpleNamespace(get_physical_devices=lambda: [nullcontext(physical)],
                                  create_infer_model=lambda path: model)
        vdevice_type = type("VDeviceDouble", (), {"create_params": staticmethod(SimpleNamespace),
                                                "__new__": lambda cls, params: nullcontext(device)})
        module = SimpleNamespace(VDevice=vdevice_type, FormatType=SimpleNamespace(UINT8="UINT8", FLOAT32="FLOAT32"),
                                  HailoSchedulingAlgorithm=SimpleNamespace(ROUND_ROBIN="ROUND_ROBIN"))
        formats = SimpleNamespace(FormatOrder=SimpleNamespace(NHWC="NHWC", HAILO_NMS_BY_CLASS="NMS"))
        with tempfile.TemporaryDirectory() as folder:
            model_file = Path(folder) / "model.hef"
            model_file.write_bytes(b"fake")
            labels_file = Path(folder) / "labels.txt"
            labels_file.write_text("cup\n", encoding="utf-8")
            config = {"object_model": str(model_file), "labels_file": str(labels_file)}
            with patch.dict("sys.modules", {"hailo_platform": module,
                                             "hailo_platform.pyhailort.pyhailort": formats}), \
                 patch("haptic_mvp.vision.preprocess_rgb", return_value=(
                     np.zeros((640, 640, 3), dtype=np.uint8), letterbox_geometry(640, 480, 640, 640))):
                with HailoDetector(config) as detector:
                    detected = detector.detect(None)
                self.assertEqual(detected[0].box, (0, 0, 1, 1))
                self.assertIn(("out", "FLOAT32"), entered)
                self.assertIn(("run", 1500), entered)
                output.is_nms = False
                with self.assertRaisesRegex(ValueError, "NMS"):
                    with HailoDetector(config):
                        pass


if __name__ == "__main__":
    unittest.main()
