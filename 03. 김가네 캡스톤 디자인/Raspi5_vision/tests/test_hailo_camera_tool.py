"""Preview control/cleanup and source-image integrity without GUI or devices."""

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from tools import check_hailo_camera as tool
from haptic_mvp.vision import decode_hailo_nms, letterbox_geometry


class CameraToolTests(unittest.TestCase):
    def run_tool(self, args, quit_frames=None, detection_outcomes=None):
        with TemporaryDirectory() as tmp:
            model, labels = Path(tmp) / "model.hef", Path(tmp) / "labels.txt"
            model.write_bytes(b"test double, not a real HEF")
            labels.write_text("cup\n")
            detector, camera = MagicMock(), MagicMock()
            detector.__enter__.return_value = detector
            camera.__enter__.return_value = camera
            detector.model_width = detector.model_height = 640
            detector.labels = ["cup"]
            detector.detect.return_value = []
            if detection_outcomes is not None:
                detector.detect.side_effect = detection_outcomes
            camera.read.return_value = (object(), 1.0)
            preview = None if quit_frames is None else object()
            with patch.object(tool, "load_config", return_value={"vision": {}}), \
                    patch.object(tool, "HailoDetector", return_value=detector), \
                    patch.object(tool, "CameraSource", return_value=camera), \
                    patch.object(tool, "preview_window", return_value=nullcontext(preview)), \
                    patch.object(tool, "show_preview", side_effect=quit_frames) as show, \
                    patch.object(tool.time, "sleep") as sleep, patch("builtins.print") as output:
                try:
                    result = tool.main(["--model", str(model), "--labels", str(labels), *args])
                finally:
                    self.assertTrue(detector.__exit__.called)
                    self.assertTrue(camera.__exit__.called)
                    self.last_preview_calls = show.call_args_list
                    self.last_output = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
            return result, camera.read.call_count, sleep.call_count, show.call_count

    def test_headless_default_still_stops_at_ten_frames(self):
        self.assertEqual(self.run_tool([]), (0, 10, 9, 0))

    def test_preview_runs_past_ten_frames_until_quit_and_has_no_text_delay(self):
        self.assertEqual(self.run_tool(["--preview"], [False] * 12 + [True]), (0, 13, 0, 13))

    def test_preview_honors_explicit_frame_limit(self):
        self.assertEqual(self.run_tool(["--preview", "--frames", "3"], [False] * 3), (0, 3, 0, 3))

    def test_ctrl_c_releases_devices(self):
        self.assertEqual(self.run_tool(["--preview"], [KeyboardInterrupt]), (130, 1, 0, 1))

    def test_preview_failure_releases_devices(self):
        with self.assertRaisesRegex(RuntimeError, "window failed"):
            self.run_tool(["--preview"], [RuntimeError("window failed")])

    def test_out_of_range_score_is_discarded_and_preview_recovers(self):
        # Exact float32 row from the 2026-10-17 Pi log, not a fabricated tolerance.
        raw = [np.array([[1.3029966e-1, 3.0338764e-4, 8.7358201e-1, 9.9982202e-1, 1.0005449]],
                        dtype=np.float32)]
        with self.assertRaisesRegex(ValueError, "confidence") as caught:
            decode_hailo_nms(raw, ("Earbud",), letterbox_geometry(640, 480, 640, 640))
        valid = [SimpleNamespace(label="cup", score=0.8, box=(0.1, 0.1, 0.4, 0.4))]
        self.assertEqual(self.run_tool(["--preview"], [False, False, True],
                                      [valid, caught.exception, valid]), (0, 3, 0, 3))
        self.assertEqual([call.kwargs["invalid"] for call in self.last_preview_calls], [False, True, False])
        self.assertEqual(self.last_preview_calls[1].args[2], ())
        self.assertEqual(self.last_preview_calls[2].args[2], valid)
        self.assertIn("검증 실패 1프레임", self.last_output)

    def test_headless_inspection_still_rejects_invalid_output(self):
        with self.assertRaisesRegex(ValueError, "confidence"):
            self.run_tool([], detection_outcomes=[ValueError("Invalid confidence")])

    def test_device_failure_is_not_swallowed_by_preview(self):
        with self.assertRaisesRegex(RuntimeError, "device failure"):
            self.run_tool(["--preview"], [False], [RuntimeError("device failure")])

    def test_repeated_invalid_frames_are_counted_but_diagnostics_are_rate_limited(self):
        with patch.object(tool.time, "monotonic", return_value=10):
            self.assertEqual(self.run_tool(["--preview"], [False, False, True],
                                          [ValueError("Invalid confidence")] * 3), (0, 3, 0, 3))
        self.assertEqual(self.last_output.count("탐지 결과 검증 실패 ("), 1)
        self.assertIn("검증 실패 3프레임", self.last_output)

    def test_missing_file_fails_before_gui_or_hardware(self):
        with TemporaryDirectory() as tmp, patch.object(tool, "preview_window") as window, \
                patch.object(tool, "HailoDetector") as detector, patch("sys.stderr"):
            with self.assertRaises(SystemExit) as error:
                tool.main(["--preview", "--model", str(Path(tmp) / "absent.hef")])
            self.assertEqual(error.exception.code, 2)
            window.assert_not_called()
            detector.assert_not_called()


class PreviewTests(unittest.TestCase):
    def test_invalid_frame_shows_a_warning_instead_of_normal_zero_detections(self):
        cv2 = MagicMock()
        cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.waitKey.return_value = ord("q")
        self.assertTrue(tool.show_preview(cv2, object(), (), 13.6, invalid=True))
        self.assertIn("INVALID OUTPUT", cv2.putText.call_args.args[1])
        cv2.rectangle.assert_not_called()

    def test_ssh_without_desktop_is_rejected_without_opening_gui(self):
        with patch.object(tool.sys, "platform", "linux"), patch.dict(tool.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PuTTY"):
                with tool.preview_window(True):
                    self.fail("Must reject missing display")

    def test_gui_backend_missing_is_reported(self):
        cv2 = MagicMock()
        cv2.currentUIFramework.return_value = ""
        with patch.dict(tool.os.environ, {"DISPLAY": ":0"}), patch.dict("sys.modules", {"cv2": cv2}):
            with self.assertRaisesRegex(RuntimeError, "GUI"):
                with tool.preview_window(True):
                    self.fail("Must reject headless OpenCV")
        cv2.namedWindow.assert_not_called()

    def test_gui_window_is_closed_even_on_exception(self):
        cv2 = MagicMock()
        cv2.currentUIFramework.return_value = "GTK3"
        with patch.dict(tool.os.environ, {"DISPLAY": ":0"}), patch.dict("sys.modules", {"cv2": cv2}):
            with self.assertRaises(RuntimeError):
                with tool.preview_window(True):
                    raise RuntimeError("inference failed")
        cv2.destroyAllWindows.assert_called_once()

    def test_preview_uses_bgr_copy_and_clips_pixel_edges(self):
        for key, should_quit in ((ord("q"), True), (27, True), (-1, False)):
            with self.subTest(key=key):
                cv2 = MagicMock()
                cv2.cvtColor.side_effect = lambda rgb, mode: rgb[:, :, ::-1].copy()
                cv2.waitKey.return_value = key
                rgb = np.zeros((48, 64, 3), dtype=np.uint8)
                rgb[:, :, 0] = 255
                original = rgb.copy()
                objects = [SimpleNamespace(label="cup", score=0.75, box=(0.0, 0.0, 1.0, 1.0))]
                self.assertEqual(tool.show_preview(cv2, rgb, objects, 13.6), should_quit)
                self.assertTrue(np.array_equal(rgb, original))
                shown = cv2.imshow.call_args.args[1]
                self.assertEqual(tuple(shown[0, 0]), (0, 0, 255))
                self.assertEqual(cv2.rectangle.call_args.args[1:3], ((0, 0), (63, 47)))


if __name__ == "__main__":
    unittest.main()
