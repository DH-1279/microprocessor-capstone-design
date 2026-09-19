"""Preflight tests; subprocess probes do not instantiate devices or models."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from haptic_mvp.config import load_config
from haptic_mvp.doctor import _asset_check, _deep_import, _selected_modules, checks, doctor


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_default_backend_selection_excludes_unselected_sdk(self):
        names = _selected_modules(self.config)
        self.assertIn("picamera2", names)
        self.assertIn("hailo_platform", names)
        self.assertIn("sounddevice", names)
        self.assertNotIn("ultralytics", names)
        self.assertNotIn("bleak", names)

    def test_cpu_usb_console_ble_selection(self):
        self.config["vision"].update(source="opencv", detector="ultralytics")
        self.config["audio"]["enabled"] = False
        self.config["ble"]["dry_run"] = False
        names = _selected_modules(self.config)
        self.assertIn("ultralytics", names)
        self.assertIn("bleak", names)
        self.assertNotIn("picamera2", names)
        self.assertNotIn("hailo_platform", names)
        self.assertNotIn("sounddevice", names)

    def test_missing_dependencies_are_not_imported(self):
        with patch("haptic_mvp.doctor.importlib.util.find_spec", return_value=None), \
             patch("haptic_mvp.doctor._deep_import") as probe:
            results = checks(self.config, deep=True)
        probe.assert_not_called()
        self.assertTrue(any(name == "mediapipe" and not ok for name, ok, _ in results))

    def test_available_hailo_also_checks_modern_infer_api(self):
        with patch("haptic_mvp.doctor.importlib.util.find_spec", return_value=object()), \
             patch("haptic_mvp.doctor._deep_import", return_value=(True, "API OK")) as probe:
            checks(self.config, deep=True)
        self.assertIn("hailo_platform.pyhailort.pyhailort", [call.args[0] for call in probe.call_args_list])

    def test_file_must_exist_be_nonempty_and_have_expected_extension(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "model.task"
            self.assertFalse(_asset_check(str(file), ".task")[0])
            file.touch()
            self.assertFalse(_asset_check(str(file), ".task")[0])
            file.write_bytes(b"placeholder")
            self.assertTrue(_asset_check(str(file), ".task")[0])
            self.assertFalse(_asset_check(str(file), ".hef")[0])
            self.assertFalse(_asset_check(folder)[0])

    def test_labels_missing_requested_target_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            labels = Path(folder) / "labels.txt"
            labels.write_text("cup\nbottle\n", encoding="utf-8")
            self.config["vision"]["labels_file"] = str(labels)
            results = checks(self.config)
        item = next(item for item in results if item[0] == "target labels")
        self.assertFalse(item[1])
        self.assertIn("cell phone", item[2])

    def test_duplicate_labels_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            labels = Path(folder) / "labels.txt"
            labels.write_text("cup\ncup\n", encoding="utf-8")
            self.config["vision"]["labels_file"] = str(labels)
            item = next(item for item in checks(self.config) if item[0] == "target labels")
        self.assertFalse(item[1])
        self.assertIn("duplicate", item[2])

    def test_readable_video_file_required_for_replay(self):
        self.config["vision"].update(source="opencv", device="does-not-exist.mp4")
        item = next(item for item in checks(self.config) if item[0] == "video file")
        self.assertFalse(item[1])

    def test_executable_home_path_is_expanded_like_audio_runtime(self):
        self.config["audio"]["whisper_executable"] = "~/Desktop/kimGA_raspi/whisper.cpp/build/bin/whisper-cli"
        with patch("haptic_mvp.doctor.shutil.which", return_value="/resolved/whisper-cli") as find:
            checks(self.config)
        self.assertEqual(find.call_args_list[0].args[0],
                         str(Path(self.config["audio"]["whisper_executable"]).expanduser()))
        self.assertNotIn("~", find.call_args_list[0].args[0])

    def test_doctor_states_limits_and_returns_failure(self):
        with patch("haptic_mvp.doctor.checks", return_value=[("SDK", False, "missing")]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(doctor(self.config, deep=True), 1)
        self.assertIn("no devices/models were opened", output.getvalue())
        self.assertIn("still need hardware tests", output.getvalue())

    def test_doctor_cli_deep_option_reaches_preflight(self):
        from haptic_mvp.__main__ import main

        with patch("haptic_mvp.doctor.doctor", return_value=0) as run:
            self.assertEqual(main(["doctor", "--deep", "--console", "--ble"]), 0)
        self.assertTrue(run.call_args.kwargs["deep"])
        selected = run.call_args.args[0]
        self.assertFalse(selected["audio"]["enabled"])
        self.assertFalse(selected["ble"]["dry_run"])


class ImportProbeTests(unittest.TestCase):
    def test_real_child_import_and_attribute_check(self):
        with patch.dict("haptic_mvp.doctor._CONTRACTS", {"json": ("loads", "dumps")}):
            ok, detail = _deep_import("json")
        self.assertTrue(ok, detail)
        self.assertIn("import/API OK", detail)

    def test_real_child_reports_removed_api(self):
        with patch.dict("haptic_mvp.doctor._CONTRACTS", {"json": ("missing_api",)}):
            ok, detail = _deep_import("json")
        self.assertFalse(ok)
        self.assertIn("AttributeError", detail)

    def test_child_does_not_construct_device(self):
        with tempfile.TemporaryDirectory() as folder:
            module = Path(folder) / "doctor_device_double.py"
            module.write_text("class Camera:\n    def __init__(self):\n        raise RuntimeError('device opened')\n"
                              "    def capture_array(self):\n        pass\n", encoding="utf-8")
            with patch.dict(os.environ, {"PYTHONPATH": folder}), \
                 patch.dict("haptic_mvp.doctor._CONTRACTS", {"doctor_device_double": ("Camera.capture_array",)}):
                ok, detail = _deep_import("doctor_device_double")
        self.assertTrue(ok, detail)

    def test_real_import_failure_reports_native_library_error(self):
        with tempfile.TemporaryDirectory() as folder:
            module = Path(folder) / "doctor_broken_double.py"
            module.write_text("raise ImportError('native ABI mismatch')\n", encoding="utf-8")
            with patch.dict(os.environ, {"PYTHONPATH": folder}), \
                 patch.dict("haptic_mvp.doctor._CONTRACTS", {"doctor_broken_double": ()}):
                ok, detail = _deep_import("doctor_broken_double")
        self.assertFalse(ok)
        self.assertIn("native ABI mismatch", detail)

    def test_hanging_import_is_terminated(self):
        with tempfile.TemporaryDirectory() as folder:
            module = Path(folder) / "doctor_hanging_double.py"
            module.write_text("import time\ntime.sleep(10)\n", encoding="utf-8")
            with patch.dict(os.environ, {"PYTHONPATH": folder}), \
                 patch.dict("haptic_mvp.doctor._CONTRACTS", {"doctor_hanging_double": ()}):
                ok, detail = _deep_import("doctor_hanging_double", timeout_s=0.2)
        self.assertFalse(ok)
        self.assertIn("timed out", detail)

    def test_native_crash_is_contained(self):
        crashed = SimpleNamespace(returncode=-11, stdout="", stderr="Segmentation fault")
        with patch("haptic_mvp.doctor.subprocess.run", return_value=crashed):
            ok, detail = _deep_import("mediapipe")
        self.assertFalse(ok)
        self.assertIn("-11", detail)
        self.assertIn("Segmentation fault", detail)

    def test_nonzero_exit_cannot_be_overridden_by_success_payload(self):
        process = SimpleNamespace(returncode=4, stdout='KIMGANE_IMPORT_RESULT={"ok": true, "detail": "loaded"}\n', stderr="")
        with patch("haptic_mvp.doctor.subprocess.run", return_value=process):
            self.assertFalse(_deep_import("mediapipe")[0])

    def test_launch_is_bounded_and_no_shell(self):
        with patch("haptic_mvp.doctor.subprocess.run", side_effect=subprocess.TimeoutExpired("probe", 15)) as run:
            ok, detail = _deep_import("numpy")
        self.assertFalse(ok)
        self.assertEqual(run.call_args.kwargs["timeout"], 15)
        self.assertNotIn("shell", run.call_args.kwargs)
        arguments = run.call_args.args[0]
        self.assertEqual(arguments[1:3], ["-B", "-c"])
        self.assertEqual(json.loads(arguments[-1])[0], "empty")


if __name__ == "__main__":
    unittest.main()
