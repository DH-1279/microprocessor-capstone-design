"""Fill the runtime TTS cache ahead of time, so the Pi never synthesises live.

audio.py already caches each announcement at

    cache_dir / sha256("espeak-ng\\0voice\\0rate\\0text").hexdigest() + ".wav"

and reuses the file when it exists. Writing the same files beforehand means the
Pi only plays them: no espeak-ng process, no first-utterance delay, and the
audio code stays unchanged.

The sentences are generated from controller.targets, so object names and the
particles that follow them come from the config rather than a second list that
can drift away from it.

Run on the Pi, or on any machine with the same espeak-ng voice and rate, then
copy the folder to the Pi.

    python -m KMJ_hand_audio_ui.tools.prebuild_tts
    python -m KMJ_hand_audio_ui.tools.prebuild_tts --config config/cpu.json
    python -m KMJ_hand_audio_ui.tools.prebuild_tts --list
    python -m KMJ_hand_audio_ui.tools.prebuild_tts --play ready
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

# Fixed announcements. Wording follows the accessibility UX spec: visual
# proximity is reported as such, never as having arrived or being able to grasp.
FIXED = {
    "ready": "준비됐어요. 찾을 물건을 말하거나 목록을 선택해 주세요.",
    "stopped": "안내를 멈췄어요.",
    "hand_lost": "손이 보이지 않아 안내를 잠시 멈춥니다.",
    "hand_recovered": "손을 다시 확인했어요. 계속할까요?",
    "target_lost": "목표를 놓쳤습니다. 물체를 다시 선택해 주세요.",
    "stale_frame": "영상이 갱신되지 않아 안내를 멈춥니다.",
    "ble_lost": "장갑 연결 상태가 바뀌었어요. 다시 선택해 주세요.",
    "near": "영상에서 목표와 가까워졌어요. 손으로 확인해 주세요.",
    "nothing_around": "주변에 아는 물건이 없습니다.",
    "unknown": "알아듣지 못했어요. 다시 말씀해 주세요.",
    "one_target": "찾을 물체 하나를 다시 말씀해 주세요.",
}

# Sentences the controller builds around an object name.
# {name} is the configured display name; {subject} and {object} pick the
# particle that matches its final consonant.
TEMPLATES = {
    "selected": "{name}{object} 찾아 안내하겠습니다.",
    "not_stable": "현재 {name}{subject} 안정적으로 보이지 않습니다.",
    "several": "{name}{subject} 여러 개 보입니다. 왼쪽 또는 오른쪽을 말씀해 주세요.",
}


def has_final_consonant(word: str) -> bool:
    if not word:
        return False
    code = ord(word[-1])
    return 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0


def with_particles(template: str, name: str) -> str:
    final = has_final_consonant(name)
    return template.format(name=name,
                           subject="이" if final else "가",
                           object="을" if final else "를")


def cache_name(voice: str, rate: int, text: str) -> str:
    """Must match audio.tts_cache_path, or the runtime will synthesise again."""
    key = f"espeak-ng\0{voice}\0{rate}\0{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest() + ".wav"


def sentences(targets: dict) -> dict[str, str]:
    items = dict(FIXED)
    for label, item in targets.items():
        name = str(item.get("name", label))
        key = label.replace(" ", "_")
        for template_key, template in TEMPLATES.items():
            items[f"{template_key}_{key}"] = with_particles(template, name)
    return items


def find_espeak(configured: str) -> str:
    import shutil

    for candidate in (configured, "espeak-ng", "espeak"):
        found = shutil.which(str(Path(candidate).expanduser()))
        if found:
            return found
    for guess in (r"C:\Program Files\eSpeak NG\espeak-ng.exe",
                  r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe"):
        if Path(guess).exists():
            return guess
    raise RuntimeError("espeak-ng not found; install it or pass --espeak")


def synthesise(espeak: str, text: str, target: Path, voice: str,
               rate: int) -> int:
    """Write via a temporary file so a failure cannot leave a broken entry."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tts-", dir=target.parent) as folder:
        temporary = Path(folder) / "speech.wav"
        result = subprocess.run(
            [espeak, "-b", "1", "-v", voice, "-s", str(rate),
             "-w", str(temporary), "--stdin"],
            input=text.encode("utf-8"), capture_output=True)
        if result.returncode:
            detail = result.stderr.decode("utf-8", "replace").strip()[-400:]
            raise RuntimeError(f"espeak-ng exited {result.returncode}: {detail}")
        with wave.open(str(temporary), "rb") as wav:
            if wav.getnframes() == 0:
                raise RuntimeError("espeak-ng produced an empty WAV")
        size = temporary.stat().st_size
        temporary.replace(target)
    return size


def play(path: Path) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"(New-Object Media.SoundPlayer '{path}').PlaySync()"],
                           check=False)
        else:
            subprocess.run(["aplay", "-q", str(path)], check=False)
    except OSError as error:
        print(f"  재생 실패: {error}")


def load(config_path: str | None):
    """Use the project's own loader so paths and defaults stay in one place."""
    try:
        from haptic_mvp.config import load_config
    except ImportError:
        raise SystemExit("Run from Raspi5_vision so haptic_mvp is importable.")
    return load_config(config_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--espeak", default=None,
                        help="override audio.tts_executable")
    parser.add_argument("--rate", type=int, default=None,
                        help="override audio.tts_rate; must match the runtime")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--play", metavar="KEY", default=None)
    parser.add_argument("--force", action="store_true",
                        help="rebuild entries that already exist")
    args = parser.parse_args()

    config = load(args.config)
    audio, controller = config["audio"], config["controller"]
    voice = audio["tts_voice"]
    rate = args.rate if args.rate is not None else int(audio["tts_rate"])
    cache = Path(audio["cache_dir"]).expanduser()
    items = sentences(controller["targets"])

    if args.list:
        print(f"\n{len(items)} sentences -> {cache}\n")
        for key, text in items.items():
            print(f"  {key:22s}  {text}")
        print()
        return 0

    if args.play:
        text = items.get(args.play)
        if text is None:
            print(f"Unknown key: {args.play}. Use --list.")
            return 1
        path = cache / cache_name(voice, rate, text)
        if not path.is_file():
            print(f"Not built yet: {path}")
            return 1
        print(f"{args.play}: {text}")
        play(path)
        return 0

    espeak = find_espeak(args.espeak or audio["tts_executable"])
    print(f"engine {espeak}")
    print(f"voice {voice}   rate {rate}")
    print(f"cache {cache}\n")

    built = skipped = 0
    total_bytes = 0
    failures = []
    for key, text in items.items():
        path = cache / cache_name(voice, rate, text)
        if path.is_file() and not args.force:
            skipped += 1
            total_bytes += path.stat().st_size
            print(f"  = {key:22s} {text}")
            continue
        try:
            total_bytes += synthesise(espeak, text, path, voice, rate)
            built += 1
            print(f"  + {key:22s} {text}")
        except Exception as error:
            failures.append((key, str(error)))
            print(f"  ! {key:22s} {error}")

    print(f"\nbuilt {built}, already present {skipped}, failed {len(failures)}")
    print(f"{total_bytes / 1024:.0f} KB in {cache}")
    if failures:
        return 1

    print("\nThe runtime reuses these files; audio.py is unchanged.")
    print("Changing audio.tts_voice or tts_rate changes the cache key, so")
    print("rebuild after editing either value.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
