"""Check the command path from the keyboard, before the microphone arrives.

`run --console` already accepts typed commands, so the same sentences a user
would speak can be checked today. This tool shows what speech.py decides for
each one, and why a refusal happened.

Two modes:

    replay    run a fixed list and report how many were handled as expected.
              The list includes the whisper-cli output recorded on 2026-10-12,
              so a change in the parser shows up as a failure here.

    live      type sentences and see the decision, with the state you choose.

Neither mode opens a microphone, a camera or a BLE link.

    python -m KMJ_hand_audio_ui.tools.console_check
    python -m KMJ_hand_audio_ui.tools.console_check --live
    python -m KMJ_hand_audio_ui.tools.console_check --state guiding
"""

from __future__ import annotations

import argparse
import sys

from .. import speech

# Fallback used when the project config cannot be loaded, so the tool still
# runs on a machine that only has this folder.
FALLBACK_TARGETS = {
    "cup": {"name": "컵", "aliases": ["컵", "머그", "cup"]},
    "bottle": {"name": "물병", "aliases": ["물병", "생수", "보틀", "bottle"]},
    "cell phone": {"name": "휴대전화",
                   "aliases": ["휴대폰", "휴대전화", "스마트폰", "핸드폰", "phone"]},
    "book": {"name": "책", "aliases": ["책", "book"]},
    "mouse": {"name": "마우스", "aliases": ["마우스", "mouse"]},
}

# (상태, 입력, 기대 결과). None means the utterance must not start anything.
CASES = [
    # 사용자가 실제로 말할 문장
    ("scanning", "컵 찾아줘", "cup"),
    ("scanning", "물병 찾아줘", "bottle"),
    ("scanning", "휴대폰 찾아줘", "cell phone"),
    ("scanning", "책 찾아줘", "book"),
    ("scanning", "마우스 찾아줘", "mouse"),
    ("scanning", "머그 찾아줘", "cup"),
    ("scanning", "생수 찾아줘", "bottle"),
    ("scanning", "리스트", "list"),
    ("scanning", "목록", "list"),

    # 2026-10-12 마이크 측정에서 whisper 가 실제로 내보낸 문장
    ("scanning", "컴 찾아줘", "cup"),
    ("scanning", "눈병 찾아줘", "bottle"),
    ("scanning", "물농 찾아줘", "bottle"),
    ("scanning", "마우스 타자줘", "mouse"),
    ("scanning", "휴데폰", "cell phone"),
    ("scanning", "컵찾아줘", "cup"),
    ("selecting", "모른 쪽", "right"),
    ("guiding", "전지", "stop"),

    # 안내 중에 들어오는 명령
    ("guiding", "정지", "stop"),
    ("guiding", "멈춰", "stop"),
    ("guiding", "목록", "list"),
    ("guiding", "컵 찾아줘", None),
    ("selecting", "왼쪽", "left"),
    ("selecting", "오른쪽", "right"),
    ("waiting", "계속", "resume"),
    ("waiting", "시작", None),
    ("near", "찾았어", "done"),
    ("stopped", "컵 찾아줘", "cup"),

    # 안내를 시작하면 안 되는 발화
    ("scanning", "책상 정리했어", None),
    ("scanning", "책상 위에 뭐 있어", None),
    ("scanning", "컵 말고 책", None),
    ("scanning", "컵이랑 책 찾아줘", None),
    ("scanning", "마우스패드", None),
    ("scanning", "물컵 어디", None),
    ("scanning", "전화 왔어", None),
    ("scanning", "오늘 날씨 어때", None),
    ("scanning", "컵 정지", None),
    ("scanning", "", None),
]


def load_targets(config_path: str | None) -> tuple[dict, str]:
    """Prefer the project config so aliases match what the runtime uses."""
    try:
        from haptic_mvp.config import load_config
    except ImportError:
        return FALLBACK_TARGETS, "fallback (haptic_mvp not importable)"
    try:
        return load_config(config_path)["controller"]["targets"], "config"
    except Exception as error:
        print(f"설정을 읽지 못해 기본 목록을 씁니다: {error}\n")
        return FALLBACK_TARGETS, "fallback"


def describe(intent) -> str:
    if intent:
        return f"{intent.kind}/{intent.name}  (매칭 {intent.detail})"
    detail = f", {intent.detail}" if intent.detail else ""
    reply = speech.refusal_speech(intent)
    return f"받지 않음  ({intent.reason}{detail})" + (f'  -> "{reply}"' if reply else "")


def run_replay(targets: dict) -> int:
    print(f"\n{len(CASES)}개 문장을 상태별로 넣어 봅니다.\n")
    passed = 0
    failed = []
    for state, text, expected in CASES:
        intent = speech.parse_input(text, targets, state)
        ok = intent.name == expected
        passed += ok
        mark = "O" if ok else "X"
        shown = text if text else "(빈 입력)"
        print(f"  {mark}  [{state:9s}] {shown:18s} -> {describe(intent)}")
        if not ok:
            failed.append((state, text, expected, intent.name))

    print(f"\n  {passed}/{len(CASES)}")
    if failed:
        print("\n기대와 다른 항목")
        for state, text, expected, got in failed:
            print(f"  [{state}] '{text}'  기대 {expected}  실제 {got}")
        return 1

    print("\n마이크가 오기 전까지는 이 경로로 명령 처리를 확인합니다.")
    print("실제 녹음과 whisper-cli 호출은 audio.py가 담당하며, 아직 확인하지")
    print("못했습니다.")
    return 0


def run_live(targets: dict, state: str) -> int:
    states = ", ".join(speech.STATE_COMMANDS)
    print(f"\n문장을 입력하세요. 빈 줄이면 끝납니다.  (상태: {state})")
    print(f"상태를 바꾸려면 :이름 을 입력하세요.  {states}\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        if text.startswith(":"):
            name = text[1:]
            if name in speech.STATE_COMMANDS:
                state = name
                print(f"  상태를 '{state}'(으)로 바꿨습니다.\n")
            else:
                print(f"  알 수 없는 상태입니다. {states}\n")
            continue
        print(f"  {describe(speech.parse_input(text, targets, state))}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--live", action="store_true",
                        help="문장을 직접 입력해 본다")
    parser.add_argument("--state", default="scanning",
                        choices=list(speech.STATE_COMMANDS))
    args = parser.parse_args()

    targets, source = load_targets(args.config)
    print(f"대상 목록: {', '.join(targets)}  ({source})")

    if args.live:
        return run_live(targets, args.state)
    return run_replay(targets)


if __name__ == "__main__":
    sys.exit(main())
