"""Prepare reusable Korean prompts, without opening the microphone/camera/BLE.

Run from Raspi5_vision: python -m KHS_KMJ_vision.prepare_audio
"""

import argparse
from pathlib import Path

from .audio import AudioService, tts_cache_path, valid_wav
from .dialogue import HAPTIC_HELP, SHORT_HAPTIC_HELP, STARTUP_HELP, target_announcement_parts
from .speech import REFUSAL_SPEECH
from .voice_profile import TARGET_MAPPING, detector_audio_config
from ble_sender.auto_link import MESSAGES


def prompt_catalog():
    """Finite prompts only: arbitrary combinations of detected items stay dynamic."""
    texts = [STARTUP_HELP, HAPTIC_HELP, SHORT_HAPTIC_HELP,
             "손이 인식되지 않습니다.",
             "물건을 찾을 수 없습니다. 다시 말씀해 주세요.",
             "취소되었습니다. 다시 부르시려면 캡스톤이라고 말씀해주세요.",
             "안내를 중지합니다. 다시 부르시려면 캡스톤이라고 말씀해주세요.",
             "진동 안내가 중단되었습니다. 연결 상태를 확인해주세요.",
             "현재 책상위엔 인식된 물건이 없습니다.",
             "찾을 물건을 말해주세요. 현재 책상위엔 인식된 물건이 없습니다."]
    texts.extend(MESSAGES.values())
    texts.extend(REFUSAL_SPEECH.values())
    for target in TARGET_MAPPING.values():
        name = target["name"]
        texts.extend(target_announcement_parts(name, True))
        texts.extend((f"현재 화면에 {name}이 없습니다. 다시 말씀해 주세요.",
                      f"현재 {name}을 찾고 있습니다.",
                      f"{name}을 획득했습니다. 다른 물건을 찾으려면 캡스톤을 말씀해주세요."))
    return tuple(dict.fromkeys(text for text in texts if text))


def main(argv=None):
    parser = argparse.ArgumentParser(description="고정 안내·물체별 음성을 한 번 합성해 저장합니다. 재생하지 않습니다.")
    parser.add_argument("--check", action="store_true", help="합성하지 않고 누락·손상된 WAV만 검사")
    parser.add_argument("--list", action="store_true", help="사전 생성할 문장 목록만 출력")
    args = parser.parse_args(argv)
    texts = prompt_catalog()
    if args.list:
        for text in texts:
            print(text)
        return 0
    config = detector_audio_config(Path(__file__).resolve().parents[1])
    service = AudioService(config, on_text=lambda _: None)
    paths = [(text, tts_cache_path(service.config, text)) for text in texts]
    paths.append(("입력 준비음", service.ready_tone_path()))
    if args.check:
        missing = [(text, path) for text, path in paths if not valid_wav(path)]
        for text, path in missing:
            print(f"준비 필요 | {text} | {path}")
        print(f"음성 파일 검사 | 정상={len(paths) - len(missing)} 준비필요={len(missing)}")
        return int(bool(missing))
    try:
        service.prepare_ready_tone()
        for index, text in enumerate(texts, 1):
            print(f"음성 준비 {index}/{len(texts)}", flush=True)
            service.prepare_speech(text)
        print(f"음성 준비 완료 | {len(texts)}개 문장 + 준비음 | {config['cache_dir']}")
        return 0
    except Exception as exc:
        print(f"음성 준비 실패 | {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
