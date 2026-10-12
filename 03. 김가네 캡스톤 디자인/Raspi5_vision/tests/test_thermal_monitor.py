"""Validate unavailable/invalid readings and native-reader error handling."""

import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.thermal_monitor import main, read_hailo, render, temperature


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
                patch("builtins.print") as output:
            self.assertEqual(main(["--demo", "--once"]), 0)
            build.assert_not_called()
            cpu.assert_not_called()
            hailo.assert_not_called()
            self.assertIn("SIMULATED", output.call_args.args[0])

    def test_invalid_refresh_interval_is_rejected(self):
        for value in ("0", "nan", "inf"):
            with self.subTest(value=value), patch("sys.stderr"), self.assertRaises(SystemExit):
                main(["--interval", value, "--once"])


if __name__ == "__main__":
    unittest.main()
