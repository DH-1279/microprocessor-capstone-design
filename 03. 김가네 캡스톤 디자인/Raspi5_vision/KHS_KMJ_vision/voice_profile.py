"""Shared voice settings; importable without camera, Hailo or audio hardware."""

from pathlib import Path


TARGET_MAPPING = {
    "Earbud": {"name": "이어폰", "aliases": ["이어폰", "무선이어폰", "이어버드"]},
    "bottle": {"name": "물병", "aliases": ["물병", "병", "페트병"]},
    "cell phone": {"name": "휴대폰", "aliases": ["휴대폰", "스마트폰", "핸드폰", "폰"]},
    "cup": {"name": "컵", "aliases": ["컵", "머그컵", "종이컵"]},
    "remote": {"name": "리모컨", "aliases": ["리모컨", "리모콘"]},
    "Glasses": {"name": "안경", "aliases": ["안경", "돋보기"]},
    "Medicine": {"name": "약통", "aliases": ["약통", "약"]},
    "key": {"name": "열쇠", "aliases": ["열쇠", "키", "차키"]},
    "Pen": {"name": "볼펜", "aliases": ["볼펜", "펜", "연필"]},
    "Tumbler": {"name": "텀블러", "aliases": ["텀블러", "보온병"]},
    "Wallet": {"name": "지갑", "aliases": ["지갑", "카드지갑"]}
}


def detector_audio_config(project_root):
    return {
        "enabled": True,
        "whisper_executable": "/home/dh1279/Desktop/kimGA_raspi/whisper.cpp/build/bin/whisper-cli",
        "whisper_model": str(Path(project_root) / "models" / "ggml-tiny.bin"),
        "tts_executable": "/usr/bin/espeak-ng",
        "playback_executable": "/usr/bin/aplay",
        "tts_voice": "M1",  # Supertonic 보이스 스타일 (M1, F1 등)
        "cache_dir": "~/.cache/kimgane-haptic/tts",
        "ready_tone_ms": 250,
        "ready_tone_amplitude": 12000,
        "output_device": "plughw:3,0",
        "input_device": 1,
    }
