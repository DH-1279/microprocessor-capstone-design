"""Sensor coverage and aspect selection using the reported IMX708 mode table."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from haptic_mvp.vision import CameraSource, full_fov_sensor_mode


def modes():
    return [
        {"size": (1536, 864), "bit_depth": 10, "fps": 120.13, "crop_limits": (768, 432, 3072, 1728)},
        {"size": (2304, 1296), "bit_depth": 10, "fps": 56.03, "crop_limits": (0, 0, 4608, 2592)},
        {"size": (4608, 2592), "bit_depth": 10, "fps": 14.35, "crop_limits": (0, 0, 4608, 2592)},
    ]


class SensorModeTests(unittest.TestCase):
    def test_full_coverage_wins_over_fast_small_central_crop(self):
        self.assertEqual(full_fov_sensor_mode(modes(), 15)["size"], (2304, 1296))

    def test_full_resolution_is_not_required_for_full_coverage(self):
        self.assertEqual(full_fov_sensor_mode(modes(), 10)["size"], (2304, 1296))

    def test_high_fps_does_not_silently_choose_a_cropped_mode(self):
        with self.assertRaisesRegex(ValueError, "56.03"):
            full_fov_sensor_mode(modes(), 60)

    def test_missing_sensor_modes_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no sensor modes"):
            full_fov_sensor_mode([], 15)


class CameraConfigurationTests(unittest.TestCase):
    def camera(self, actual_size=(2304, 1296)):
        camera = MagicMock()
        camera.sensor_modes = modes()
        camera.camera_properties = {"ScalerCropMaximum": (0, 0, 4608, 2592)}
        camera.create_video_configuration.side_effect = lambda **kwargs: kwargs
        camera.camera_configuration.return_value = {"sensor": {"output_size": actual_size}}
        return camera

    def test_full_fov_uses_native_aspect_and_maximum_crop(self):
        camera = self.camera()
        config = {"width": 640, "height": 480, "fps": 15}
        module = SimpleNamespace(Picamera2=MagicMock(return_value=camera))
        with patch.dict(sys.modules, {"picamera2": module}):
            with CameraSource(config, full_fov=True) as source:
                self.assertEqual(source.full_fov_info["output"], (640, 360))
                self.assertEqual(source.full_fov_info["crop"], (0, 0, 4608, 2592))
        settings = camera.create_video_configuration.call_args.kwargs
        self.assertEqual(settings["main"]["size"], (640, 360))
        self.assertEqual(settings["sensor"], {"output_size": (2304, 1296), "bit_depth": 10})
        self.assertEqual(settings["main"]["format"], "BGR888")
        camera.set_controls.assert_called_once_with({"ScalerCrop": (0, 0, 4608, 2592)})
        camera.stop.assert_called_once()
        camera.close.assert_called_once()
        self.assertEqual(config["height"], 480)

    def test_default_capture_size_and_auto_sensor_selection_are_preserved(self):
        camera = self.camera()
        with patch.dict(sys.modules, {"picamera2": SimpleNamespace(Picamera2=lambda **kwargs: camera)}):
            with CameraSource({"width": 640, "height": 480, "fps": 15}) as source:
                self.assertIsNone(source.full_fov_info)
        settings = camera.create_video_configuration.call_args.kwargs
        self.assertEqual(settings["main"]["size"], (640, 480))
        self.assertEqual(settings["sensor"], {})
        camera.set_controls.assert_not_called()

    def test_incorrectly_applied_sensor_mode_closes_camera(self):
        camera = self.camera(actual_size=(1536, 864))
        with patch.dict(sys.modules, {"picamera2": SimpleNamespace(Picamera2=lambda **kwargs: camera)}):
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                with CameraSource({"fps": 15}, full_fov=True):
                    self.fail("Must not silently claim full coverage")
        camera.start.assert_not_called()
        camera.close.assert_called_once()

    def test_full_fov_rejects_non_csi_sources_before_opening(self):
        with self.assertRaisesRegex(ValueError, "Picamera2"):
            with CameraSource({"source": "opencv"}, full_fov=True):
                self.fail("USB cameras have different mode controls")


if __name__ == "__main__":
    unittest.main()
