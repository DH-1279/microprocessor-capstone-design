"""Validate unavailable/invalid readings and native-reader error handling."""

import io
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.thermal_monitor import (
    FanReading, TerminalScreen, fan_status, fit_line, main, read_fan, read_hailo, render, temperature,
)


class ThermalMonitorTests(unittest.TestCase):
    def test_unavailable_and_nonfinite_readings_are_rejected(self):
        for value in ("-1", "nan", "inf", "-inf", "151", "not a temperature"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                temperature(value)

    def test_native_reading_is_parsed(self):
        with patch("tools.thermal_monitor.subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout="54.25\n", stderr="")) as run:
            self.assertEqual(read_hailo("helper"), 54.25)
            self.assertEqual(run.call_args.kwargs["timeout"], 4)

    def test_native_failure_does_not_become_zero_degrees(self):
        with patch("tools.thermal_monitor.subprocess.run", return_value=SimpleNamespace(
            returncode=2, stdout="", stderr="Health query failed: status=6")):
            with self.assertRaisesRegex(RuntimeError, "status=6"):
                read_hailo("helper")

    def test_native_timeout_is_not_a_reading(self):
        with patch("tools.thermal_monitor.subprocess.run", side_effect=subprocess.TimeoutExpired("helper", 4)):
            with self.assertRaises(subprocess.TimeoutExpired):
                read_hailo("helper")

    def test_failed_sensor_shows_unavailable_even_when_previous_peak_exists(self):
        output = render([42, None], {"CPU": 45, "HAILO": 55}, ["HAILO: failed"], color=False)
        self.assertIn("N/A", output)
        self.assertNotIn("55.0", output)
        self.assertIn("LIVE", output)
        self.assertNotIn("\033", output)

    def test_subprocess_errors_cannot_inject_terminal_escape(self):
        output = render([None, None], {}, ["error\033[2J\nnext"], color=False)
        self.assertNotIn("\033", output)

    def test_demo_does_not_build_or_access_sensors(self):
        with patch("tools.thermal_monitor.build_helper") as build, \
                patch("tools.thermal_monitor.read_cpu") as cpu, \
                patch("tools.thermal_monitor.read_hailo") as hailo, \
                patch("tools.thermal_monitor.read_fan") as fan, \
                patch("builtins.print") as output:
            self.assertEqual(main(["--demo", "--once"]), 0)
            build.assert_not_called()
            cpu.assert_not_called()
            hailo.assert_not_called()
            fan.assert_not_called()
            self.assertIn("SIMULATED", output.call_args.args[0])
            self.assertIn("FAN RUNNING 2726 RPM", output.call_args.args[0])

    def test_invalid_refresh_interval_is_rejected(self):
        for value in ("0", "nan", "inf"):
            with self.subTest(value=value), patch("sys.stderr"), self.assertRaises(SystemExit):
                main(["--interval", value, "--once"])


class FanMonitorTests(unittest.TestCase):
    def make_device(self, root, name="pwmfan", rpm="2726", pwm="75", index=8):
        device = root / f"hwmon{index}"
        device.mkdir()
        (device / "name").write_text(name)
        if rpm is not None:
            (device / "fan1_input").write_text(rpm)
        if pwm is not None:
            (device / "pwm1").write_text(pwm)
        return device

    def test_finds_named_fan_even_when_hwmon_numbers_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_device(root, name="other-fan", rpm="9999", index=2)
            self.make_device(root)
            self.assertEqual(read_fan(root), FanReading(2726, 75))

    def test_optional_pwm_failure_does_not_hide_valid_rpm(self):
        for pwm in (None, "broken", "256", "-1"):
            with self.subTest(pwm=pwm), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.make_device(root, pwm=pwm)
                self.assertEqual(read_fan(root), FanReading(2726))

    def test_missing_or_invalid_rpm_cannot_claim_rotation(self):
        for rpm in (None, "-1", "nan", "broken"):
            with self.subTest(rpm=rpm), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.make_device(root, rpm=rpm)
                with self.assertRaises((OSError, ValueError)):
                    read_fan(root)

    def test_absent_or_ambiguous_fan_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "not found"):
                read_fan(root)
            self.make_device(root)
            self.make_device(root, index=9)
            with self.assertRaisesRegex(RuntimeError, "Multiple"):
                read_fan(root)

    def test_status_distinguishes_stop_missing_and_no_rotation_with_pwm(self):
        for reading, expected in ((FanReading(2623), "RUNNING 2623 RPM"),
                                  (FanReading(0, 0), "STOPPED 0 RPM"),
                                  (FanReading(0, 255), "CHECK 0 RPM"), (None, "N/A")):
            with self.subTest(reading=reading):
                self.assertIn(expected, fan_status(reading)[0])

    def test_fan_is_beside_cpu_in_wide_screen_and_separate_in_narrow_screen(self):
        for columns in (80, 40):
            with self.subTest(columns=columns):
                lines = render([42, 50], {"CPU": 42, "HAILO": 50}, [], color=False,
                               fan=FanReading(9077, 255), columns=columns).splitlines()
                cpu_row = next(i for i, line in enumerate(lines) if line.startswith("  CPU"))
                fan_row = next(i for i, line in enumerate(lines) if "RUNNING" in line)
                self.assertEqual(cpu_row == fan_row, columns >= 76)

    def test_fan_failure_keeps_cpu_and_hailo_available(self):
        with patch("tools.thermal_monitor.build_helper", return_value=Path("helper")), \
                patch("tools.thermal_monitor.read_cpu", return_value=42), \
                patch("tools.thermal_monitor.read_hailo", return_value=50), \
                patch("tools.thermal_monitor.read_fan", side_effect=OSError("sensor missing")), \
                patch("builtins.print") as output:
            self.assertEqual(main(["--once"]), 1)
            frame = output.call_args.args[0]
            self.assertIn("42.0", frame)
            self.assertIn("50.0", frame)
            self.assertIn("FAN N/A", frame)
            self.assertNotIn("STOPPED", frame)


class TerminalRefreshTests(unittest.TestCase):
    def test_width_clip_counts_wide_characters_and_preserves_sgr(self):
        self.assertEqual(fit_line("가나다", 5), "가나")
        self.assertEqual(fit_line("\033[32m123456\033[0m", 4), "\033[32m1234\033[0m")

    def test_repeated_frames_fit_terminal_without_newlines_and_restore_screen(self):
        output = io.StringIO()
        sizes = (os.terminal_size((80, 24)), os.terminal_size((20, 6)))
        with TerminalScreen(output) as screen:
            for size in sizes:
                before = len(output.getvalue())
                with patch("tools.thermal_monitor.shutil.get_terminal_size", return_value=size):
                    screen.draw("\n".join(["가" * 90] * 30))
                frame = output.getvalue()[before:]
                self.assertNotIn("\n", frame)
                rows = re.findall(r"\033\[(\d+);1H\033\[2K([^\033]*)", frame)
                self.assertEqual(len(rows), size.lines - 1)
                for row, text in rows:
                    self.assertLess(int(row), size.lines)
                    self.assertLessEqual(len(text) * 2, size.columns - 1)
        self.assertTrue(output.getvalue().startswith("\033[?1049h"))
        self.assertTrue(output.getvalue().endswith("\033[0m\033[?25h\033[?1049l"))

    def test_screen_restores_cursor_on_failure(self):
        output = io.StringIO()
        with self.assertRaises(RuntimeError):
            with TerminalScreen(output):
                raise RuntimeError("failed")
        self.assertIn("\033[?25h\033[?1049l", output.getvalue())

    def test_forced_screen_handles_ctrl_c_and_never_appends_frames(self):
        output = io.StringIO()
        with patch("sys.stdout", output), \
                patch("tools.thermal_monitor.time.sleep", side_effect=KeyboardInterrupt):
            self.assertEqual(main(["--demo", "--screen"]), 0)
        self.assertIn("FAN RUNNING 2726 RPM", output.getvalue())
        self.assertNotIn("\n", output.getvalue())
        self.assertTrue(output.getvalue().endswith("\033[?25h\033[?1049l"))

    def test_redirected_output_is_one_snapshot_by_default(self):
        output = io.StringIO()
        with patch("sys.stdout", output), patch("tools.thermal_monitor.time.sleep") as sleep:
            self.assertEqual(main(["--demo"]), 0)
        sleep.assert_not_called()
        self.assertEqual(output.getvalue().count("THERMAL MONITOR"), 1)
        self.assertNotIn("\033", output.getvalue())

    def test_explicit_log_mode_appends_plain_samples(self):
        output = io.StringIO()
        with patch("sys.stdout", output), \
                patch("tools.thermal_monitor.time.sleep", side_effect=[None, KeyboardInterrupt]):
            self.assertEqual(main(["--demo", "--log"]), 0)
        self.assertEqual(output.getvalue().count("THERMAL MONITOR"), 2)
        self.assertNotIn("\033", output.getvalue())


if __name__ == "__main__":
    unittest.main()
