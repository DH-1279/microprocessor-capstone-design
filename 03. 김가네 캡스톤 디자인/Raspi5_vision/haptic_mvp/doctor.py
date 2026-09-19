"""Availability and optional isolated import checks; never open physical devices."""

import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


# Only inspect attributes used by our adapters. No device/model constructor runs.
_CONTRACTS = {
    "numpy": ("empty", "full", "ascontiguousarray", "uint8", "float32"),
    "cv2": ("VideoCapture", "resize", "cvtColor", "COLOR_RGB2BGR"),
    "mediapipe": ("Image", "ImageFormat.SRGB", "tasks.BaseOptions",
                  "tasks.vision.RunningMode.VIDEO", "tasks.vision.HandLandmarkerOptions",
                  "tasks.vision.HandLandmarker.create_from_options",
                  "tasks.vision.HandLandmarker.detect_for_video"),
    "picamera2": ("Picamera2.capture_array", "Picamera2.create_video_configuration",
                  "Picamera2.configure", "Picamera2.start", "Picamera2.stop", "Picamera2.close"),
    "hailo_platform": ("VDevice.create_params", "VDevice.create_infer_model",
                       "VDevice.get_physical_devices", "FormatType.UINT8", "FormatType.FLOAT32",
                       "HailoSchedulingAlgorithm.ROUND_ROBIN"),
    "hailo_platform.pyhailort.pyhailort": ("FormatOrder.NHWC", "FormatOrder.HAILO_NMS_BY_CLASS",
                                           "ConfiguredInferModel.create_bindings", "ConfiguredInferModel.run",
                                           "InferModel.configure", "InferModel.set_batch_size"),
    "ultralytics": ("YOLO",),
    "bleak": ("BleakScanner.discover", "BleakClient.write_gatt_char", "BleakClient.read_gatt_char"),
    "sounddevice": ("InputStream", "query_devices"),
}

_RESULT_PREFIX = "KIMGANE_IMPORT_RESULT="
_IMPORT_PROBE = r'''
import importlib
import importlib.metadata
import json
import sys

name = sys.argv[1]
try:
    module = importlib.import_module(name)
    for path in json.loads(sys.argv[2]):
        value = module
        for part in path.split('.'):
            value = getattr(value, part)
    distribution = {"cv2": "opencv-contrib-python", "hailo_platform": "hailort"}.get(name, name)
    version = str(getattr(module, '__version__', ''))
    if not version:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = 'version unavailable'
    result = {'ok': True, 'detail': 'import/API OK; ' + version}
except Exception as error:
    result = {'ok': False, 'detail': type(error).__name__ + ': ' + str(error)}
print('KIMGANE_IMPORT_RESULT=' + json.dumps(result, ensure_ascii=True), flush=True)
raise SystemExit(0 if result['ok'] else 2)
'''


def _selected_modules(config: dict) -> list[str]:
    v, a = config["vision"], config["audio"]
    modules = ["numpy", "cv2", "mediapipe"]
    if v["source"] == "picamera2":
        modules.append("picamera2")
    modules.append("hailo_platform" if v["detector"] == "hailo" else "ultralytics")
    if not config["ble"]["dry_run"]:
        modules.append("bleak")
    if a["enabled"]:
        modules.append("sounddevice")
    return modules


def _deep_import(name: str, timeout_s: float = 15) -> tuple[bool, str]:
    """Contain native SDK crashes and hangs in a child using this same Python."""
    with tempfile.TemporaryDirectory(prefix="kimgane-import-check-") as cache:
        environment = os.environ.copy()
        # Some SDKs create settings/plot caches at import; keep these temporary.
        environment.update({"PYTHONDONTWRITEBYTECODE": "1", "YOLO_CONFIG_DIR": cache,
                            "MPLCONFIGDIR": cache, "XDG_CACHE_HOME": cache})
        try:
            result = subprocess.run(
                [sys.executable, "-B", "-c", _IMPORT_PROBE, name, json.dumps(_CONTRACTS[name])],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout_s, check=False, env=environment,
            )
        except subprocess.TimeoutExpired:
            return False, f"import timed out after {timeout_s:g}s (child terminated)"
        except OSError as error:
            return False, f"could not start import check: {error}"
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            try:
                report = json.loads(line[len(_RESULT_PREFIX):])
                detail = report["detail"]
                if type(report["ok"]) is not bool or not isinstance(detail, str):
                    raise ValueError("unexpected probe fields")
            except (KeyError, TypeError, ValueError):
                return False, "invalid import-check result"
            return report["ok"] and result.returncode == 0, detail[-1500:]
    detail = (result.stderr or result.stdout).strip()[-1500:]
    return False, f"import process exited {result.returncode} without result; {detail or 'possible native SDK crash'}"


def _asset_check(value: str, suffix: str | None = None) -> tuple[bool, str]:
    path = Path(value)
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False, f"missing or empty: {path}"
        if suffix is not None and path.suffix.lower() != suffix:
            return False, f"expected {suffix}: {path}"
    except OSError as error:
        return False, f"cannot read {path}: {error}"
    return True, str(path)


def checks(config: dict, deep: bool = False) -> list[tuple[str, bool, str]]:
    results = [("Python", sys.version_info >= (3, 10), platform.python_version()),
               ("Linux", sys.platform == "linux", sys.platform),
               ("Architecture", platform.machine().lower() in ("aarch64", "arm64"), platform.machine())]
    v, a = config["vision"], config["audio"]
    modules = _selected_modules(config)
    for name in modules:
        try:
            ok = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            ok = False
        results.append((name, ok, "found" if ok else "not installed in this Python environment"))
        if deep and ok:
            imported, detail = _deep_import(name)
            results.append((f"{name} import/API", imported, detail))
            if name == "hailo_platform" and imported:
                imported, detail = _deep_import("hailo_platform.pyhailort.pyhailort")
                results.append(("HailoRT InferModel API", imported, detail))
    files: list[tuple[str, str, str | None]] = [
        ("object_model", v["object_model"], ".hef" if v["detector"] == "hailo" else ".pt"),
        ("hand_model", v["hand_model"], ".task"),
    ]
    if v["detector"] == "hailo":
        files.append(("labels_file", v["labels_file"], None))
    if a["enabled"]:
        files.append(("whisper_model", a["whisper_model"], ".bin"))
        for key in ("whisper_executable", "tts_executable", "playback_executable"):
            binary = str(Path(a[key]).expanduser())
            found = shutil.which(binary)
            results.append((key, found is not None, found or binary))
    for name, value, suffix in files:
        ok, detail = _asset_check(value, suffix)
        results.append((name, ok, detail))
        if name == "labels_file" and ok:
            try:
                from .vision import read_labels
                labels = read_labels(value)
                missing_targets = set(config["controller"]["targets"]) - set(labels)
                results.append(("target labels", not missing_targets,
                                f"missing from labels: {sorted(missing_targets)}" if missing_targets
                                else f"{len(labels)} labels; all configured targets present"))
            except (OSError, UnicodeError, ValueError) as error:
                results.append(("target labels", False, str(error)))
    if v["source"] == "opencv" and isinstance(v["device"], str):
        ok, detail = _asset_check(v["device"])
        results.append(("video file", ok, detail))
    return results


def doctor(config: dict, deep: bool = False) -> int:
    results = checks(config, deep=deep)
    for name, ok, detail in results:
        print(f"{'OK  ' if ok else 'MISS'} {name}: {detail}")
    if deep:
        print("Deep checks import dependencies and inspect APIs in bounded subprocesses; no devices/models were opened.")
    else:
        print("Doctor checks availability only. Add --deep to check imports, native-library loading and required APIs.")
    print("Camera/audio quality, model contents, Hailo HEF compatibility and BLE connections still need hardware tests.")
    return 0 if all(ok for _, ok, _ in results) else 1
