"""Configuration loading with fail-fast checks before hardware starts."""

import copy
import json
from pathlib import Path

from .models import finite_number

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "mvp.json"


def _merge(base: dict, changes: dict, prefix: str = "") -> dict:
    for key, value in changes.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected object: {prefix}{key}")
            _merge(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value
    return base


def _number(mapping: dict, key: str, minimum: float, maximum: float, integer: bool = False) -> None:
    value = mapping[key]
    if not finite_number(value) or not minimum <= value <= maximum or integer and type(value) is not int:
        raise ValueError(f"{key} must be {'integer ' if integer else ''}{minimum}..{maximum}")


def validate(config: dict) -> None:
    c, b, v, a = (config[key] for key in ("controller", "ble", "vision", "audio"))
    for key in ("object_score", "hand_score", "tracking_iou", "ambiguity_margin", "side_separation"):
        _number(c, key, 0.001, 1)
    _number(c, "max_frame_age_s", 0.05, 2)
    _number(c, "announce_interval_s", 1, 120)
    _number(c, "stable_frames", 1, 30, True)
    _number(c, "image_aspect", 0.25, 4)
    _number(c, "near_distance", 0, 1)
    _number(c, "middle_distance", 0, 2)
    if c["near_distance"] >= c["middle_distance"]:
        raise ValueError("near_distance must be smaller than middle_distance")
    if type(c["rotation_deg"]) is not int or c["rotation_deg"] not in (0, 90, 180, 270):
        raise ValueError("rotation_deg must be 0, 90, 180, or 270")
    if type(c["flip_x"]) is not bool or type(b["dry_run"]) is not bool or type(a["enabled"]) is not bool:
        raise ValueError("flip_x, dry_run, enabled must be booleans")
    if set(c["channel_map"]) != {"left", "right", "forward", "back", "near"}:
        raise ValueError("channel_map must specify five directions")
    channels = list(c["channel_map"].values())
    if any(type(x) is not int for x in channels) or sorted(channels) != list(range(5)):
        raise ValueError("channel_map must use each index 0..4 exactly once")
    for key in ("near", "middle", "far"):
        _number(c["strengths"], key, 1, 255, True)
    for item in c["targets"].values():
        if (not isinstance(item["name"], str) or not item["name"].strip()
                or not isinstance(item["aliases"], list) or not item["aliases"]
                or any(not isinstance(x, str) or not x.strip() for x in item["aliases"])):
            raise ValueError("targets require a name and nonempty aliases")
    _number(b, "ttl_ms", 100, 1000, True)
    _number(b, "send_interval_s", 0.02, 1)
    if b["send_interval_s"] * 2000 >= b["ttl_ms"]:
        raise ValueError("ttl_ms must exceed twice the send interval")
    for key in ("scan_timeout_s", "operation_timeout_s", "reconnect_delay_s"):
        _number(b, key, 0.1, 60)
    if b["address"] is not None and not isinstance(b["address"], str):
        raise ValueError("BLE address must be string or null")
    for key in ("width", "height"):
        _number(v, key, 64, 4096, True)
    _number(v, "fps", 1, 60)
    if abs(c["image_aspect"] - v["width"] / v["height"]) > 0.01:
        raise ValueError("image_aspect must match configured width / height")
    if v["source"] not in ("picamera2", "opencv") or v["detector"] not in ("hailo", "ultralytics"):
        raise ValueError("unsupported vision source or detector")
    if v["hand"] not in ("Right", "Left", "Any"):
        raise ValueError("hand must be Right, Left, or Any")
    for key in ("mirror", "show_preview"):
        if type(v[key]) is not bool:
            raise ValueError(f"vision.{key} must be boolean")
    for key in ("score_threshold", "hand_score_threshold"):
        _number(v, key, 0.001, 1)
    _number(v, "max_objects", 1, 100, True)
    _number(v, "infer_timeout_ms", 100, 10000, True)
    if not (type(v["device"]) is int and v["device"] >= 0
            or isinstance(v["device"], str) and v["device"].strip()):
        raise ValueError("vision.device must be camera number or local device/file path")
    for section, keys in ((v, ("object_model", "labels_file", "hand_model")),
                          (a, ("whisper_model", "whisper_executable", "cache_dir"))):
        if any(not isinstance(section[key], str) or not section[key].strip() for key in keys):
            raise ValueError("model, executable and cache paths must be nonempty strings")


def load_config(path: str | Path | None = None, resolve_paths: bool = True) -> dict:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        changes = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(changes, dict):
            raise ValueError("configuration must be a JSON object")
        config = _merge(copy.deepcopy(config), changes)
    validate(config)
    if resolve_paths:
        for section, fields in (("vision", ("object_model", "labels_file", "hand_model")),
                                ("audio", ("whisper_model", "cache_dir"))):
            for key in fields:
                item = Path(config[section][key]).expanduser()
                config[section][key] = str(item if item.is_absolute() else ROOT / item)
        device = config["vision"]["device"]
        if isinstance(device, str) and not device.startswith("/dev/"):
            item = Path(device).expanduser()
            config["vision"]["device"] = str(item if item.is_absolute() else ROOT / item)
    return config
