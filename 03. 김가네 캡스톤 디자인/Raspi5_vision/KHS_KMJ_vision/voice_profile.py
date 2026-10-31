"""Shared voice settings; importable without camera, Hailo or audio hardware."""

from pathlib import Path


TARGET_MAPPING = {
    "bottle": {"name": "물병", "aliases": ["물병", "병", "페트병"]},
    "cup": {"name": "컵", "aliases": ["컵", "머그컵", "종이컵"]},
    "mouse": {"name": "마우스", "aliases": ["마우스", "무선마우스"]},
    "remote": {"name": "리모컨", "aliases": ["리모컨", "리모콘"]},
    "Glasses": {"name": "안경", "aliases": ["안경", "돋보기"]},
    "Pen": {"name": "볼펜", "aliases": ["볼펜", "펜", "연필"]},
    "Tumbler": {"name": "텀블러", "aliases": ["텀블러", "보온병"]},
    "Wallet": {"name": "지갑", "aliases": ["지갑", "카드지갑"]}
}


def detector_audio_config(project_root):
    speaker_card = "UACDemoV10"
    return {
        "enabled": True,
        "whisper_executable": "/home/dh1279/Desktop/kimGA_raspi/whisper.cpp/build/bin/whisper-cli",
        "whisper_model": str(Path(project_root) / "models" / "ggml-tiny.bin"),
        "tts_executable": "/usr/bin/espeak-ng",
        "playback_executable": "/usr/bin/aplay",
        "tts_voice": "M1",  # Supertonic 보이스 스타일 (M1, F1 등)
        "cache_dir": "~/.cache/kimgane-haptic/tts",
        "ready_tone_ms": 250,
        "ready_tone_amplitude": 6000,
        "output_device": f"plughw:CARD={speaker_card},DEV=0",
        "output_mixer_card": speaker_card,
        "input_backend": "alsa",
        "input_alsa_device": "plughw:CARD=GM20U,DEV=0",
        "capture_executable": "/usr/bin/arecord",
        "mic_retry_s": 2.0,
        "mic_stall_s": 2.0,
    }
