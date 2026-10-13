"""
STT 인식률 측정 도구

마이크로 명령을 녹음해 Whisper로 인식하고, 어휘 매핑까지 적용한 뒤
최종 명령이 올바른지 측정한다.

문서에 정한 전환 기준을 판단하기 위한 것이다.
  - 명령 인식 성공률 90% 미만이면 base로 상향
  - 매핑 계층까지 적용한 뒤 측정한다

준비
    py -m pip install faster-whisper sounddevice
    command_mapper.py 를 같은 폴더에 둔다

실행
    py stt_test.py --list              입력 장치 목록 보기
    py stt_test.py --name K66          이름으로 마이크 선택 (권장)
    py stt_test.py --device 11         번호로 마이크 선택
    py stt_test.py --name K66 --model base
    py stt_test.py --name K66 --free   자유 입력 모드

장치 번호는 바뀔 수 있으므로 --name 으로 고르는 편이 안전하다.
첫 실행 시 모델을 자동으로 받는다. tiny는 약 75MB, base는 약 145MB.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import sounddevice as sd

from command_mapper import parse, STATES

TARGET_RATE = 16000          # Whisper가 쓰는 샘플레이트
RESULT_CSV = "stt_results.csv"

TEST_PROMPTS = [
    ("cup", "컵 찾아줘"),
    ("bottle", "물병 찾아줘"),
    ("phone", "휴대폰 찾아줘"),
    ("book", "책 찾아줘"),
    ("mouse", "마우스 찾아줘"),
    ("list", "목록"),
    ("stop", "정지"),
    ("left", "왼쪽"),
    ("right", "오른쪽"),
]


# --- 장치 선택 --------------------------------------------------------------

def list_input_devices():
    """입력이 가능한 장치만 추려서 보여준다."""
    print("\n사용 가능한 입력 장치\n")
    found = False
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            api = sd.query_hostapis(d["hostapi"])["name"]
            print(f"  {i:3d}  {d['name']}")
            print(f"       {api} / 채널 {d['max_input_channels']} / "
                  f"기본 {int(d['default_samplerate'])}Hz")
            found = True
    if not found:
        print("  입력 장치를 찾지 못했습니다.")
    print()


def find_device(name):
    """이름 일부로 입력 장치를 찾는다."""
    target = name.lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and target in d["name"].lower():
            return i
    return None


def negotiate(device):
    """
    장치가 실제로 받아주는 (샘플레이트, 채널) 조합을 찾는다.

    WDM-KS 같은 일부 드라이버는 16kHz나 모노를 거부한다.
    그런 경우 장치가 되는 설정으로 받은 뒤 나중에 변환한다.
    """
    info = sd.query_devices(device, "input")
    max_ch = int(info["max_input_channels"])
    default_sr = int(info["default_samplerate"])

    candidates = []
    for sr in (TARGET_RATE, default_sr, 48000, 44100):
        if sr not in candidates:
            candidates.append(sr)

    for sr in candidates:
        for ch in (1, 2):
            if ch > max_ch:
                continue
            try:
                sd.check_input_settings(device=device, channels=ch,
                                        samplerate=sr, dtype="float32")
                return sr, ch
            except Exception:
                continue

    raise RuntimeError("이 장치로 녹음할 수 있는 설정을 찾지 못했습니다.")


# --- 녹음과 인식 ------------------------------------------------------------

def resample(audio, src_rate, dst_rate):
    """샘플레이트를 바꾼다. 선형 보간으로 충분하다."""
    if src_rate == dst_rate:
        return audio
    n = int(round(len(audio) * dst_rate / src_rate))
    old = np.linspace(0.0, 1.0, len(audio), dtype=np.float64)
    new = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return np.interp(new, old, audio).astype(np.float32)


def record(seconds, device, rate, channels):
    """녹음해서 16kHz 모노 배열로 돌려준다."""
    print(f"  녹음 중... ({seconds}초)")
    audio = sd.rec(int(seconds * rate), samplerate=rate,
                   channels=channels, dtype="float32", device=device)
    sd.wait()

    if audio.ndim > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)      # 스테레오면 모노로
    audio = audio.flatten()

    level = float(np.abs(audio).max()) if len(audio) else 0.0
    if level < 0.01:
        print("  (주의) 입력 소리가 거의 없습니다. 마이크를 확인하세요.")

    return resample(audio, rate, TARGET_RATE)


def transcribe(model, audio):
    """
    녹음된 음성을 텍스트로 변환한다.

    파일 경로 대신 numpy 배열을 바로 넘긴다.
    파일로 넘기면 faster-whisper가 내부에서 PyAV로 디코딩하는데,
    av 14 이상에서 폐지된 인자를 쓰고 있어 오류가 난다.
    배열로 넘기면 그 경로를 거치지 않고, 임시 파일 입출력도 없어진다.
    """
    start = time.time()
    segments, _ = model.transcribe(audio, language="ko", beam_size=5)
    text = "".join(seg.text for seg in segments).strip()
    return text, time.time() - start


def save_result(rows):
    """측정 결과를 CSV에 추가한다."""
    is_new = not os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["측정시각", "모델", "기대명령", "말한문장",
                             "STT출력", "매핑결과", "성공", "인식시간(초)"])
        writer.writerows(rows)
    print(f"\n{RESULT_CSV} 에 저장했습니다.")


# --- 실행 모드 --------------------------------------------------------------

def run_measure(model, model_name, device, rate, channels, repeat):
    rows = []
    total = success = stt_exact = 0

    print(f"\n{'=' * 60}")
    print(f"모델: {model_name}   명령 {len(TEST_PROMPTS)}개 × {repeat}회")
    print(f"{'=' * 60}")
    print("문장이 나오면 엔터를 누르고 3초 안에 말하세요.")
    print("건너뛰려면 s, 종료하려면 q 를 입력하세요.\n")

    for round_no in range(1, repeat + 1):
        if repeat > 1:
            print(f"\n--- {round_no}회차 ---")

        for expected, prompt in TEST_PROMPTS:
            cmd = input(f'\n  "{prompt}"  준비되면 엔터 > ').strip().lower()
            if cmd == "q":
                print("\n측정을 중단합니다.")
                if rows:
                    save_result(rows)
                return
            if cmd == "s":
                continue

            audio = record(3, device, rate, channels)
            text, elapsed = transcribe(model, audio)
            intent = parse(text, state="전체")
            mapped = intent.name
            ok = (mapped == expected)

            total += 1
            if ok:
                success += 1
            if text.replace(" ", "") == prompt.replace(" ", ""):
                stt_exact += 1

            mark = "O" if ok else "X"
            note = "" if intent else f", {intent.reason}"
            print(f"  {mark}  STT: '{text}'  ->  {mapped}"
                  f"  ({elapsed:.1f}초{note})")

            rows.append([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name, expected, prompt, text,
                str(mapped), mark, f"{elapsed:.2f}",
            ])

    if total == 0:
        print("\n측정한 항목이 없습니다.")
        return

    cmd_pct = success / total * 100
    stt_pct = stt_exact / total * 100

    print(f"\n{'=' * 60}")
    print(f"모델: {model_name}")
    print(f"명령 인식 성공률   {success}/{total}  ({cmd_pct:.1f}%)")
    print(f"STT 문장 정확도    {stt_exact}/{total}  ({stt_pct:.1f}%)")
    print(f"{'=' * 60}")

    if cmd_pct >= 90:
        print("\n기준(90%)을 충족합니다.")
    else:
        print("\n기준(90%) 미달입니다. 더 큰 모델을 검토해야 합니다.")

    if stt_pct < cmd_pct:
        print(f"매핑 계층이 {cmd_pct - stt_pct:.1f}%p를 보정했습니다.")

    save_result(rows)


def run_free(model, device, rate, channels):
    print("\n자유 입력 모드입니다. 엔터를 누르면 3초간 녹음합니다.")
    print("상태를 바꾸려면 :탐색 :선택 :안내 :전체 중 하나를 입력하세요.")
    print("종료하려면 q 를 입력하세요.\n")

    state = "전체"
    while True:
        cmd = input(f"[{state}] 엔터 > ").strip()
        if cmd.lower() == "q":
            break
        if cmd.startswith(":"):
            new_state = cmd[1:]
            if new_state in STATES:
                state = new_state
                print(f"  상태를 '{state}'(으)로 바꿨습니다.\n")
            else:
                print(f"  알 수 없는 상태입니다. {list(STATES)}\n")
            continue

        audio = record(3, device, rate, channels)
        text, elapsed = transcribe(model, audio)
        intent = parse(text, state=state)

        print(f"  STT 출력 : '{text}'  ({elapsed:.1f}초)")
        if intent:
            print(f"  해석 결과: {intent.kind} / {intent.name}"
                  f"  (매칭 {intent.detail})\n")
        else:
            detail = f", {intent.detail}" if intent.detail else ""
            print(f"  받지 않음  (사유 {intent.reason}{detail})\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true",
                        help="입력 장치 목록만 보고 끝낸다")
    parser.add_argument("--name", default=None,
                        help="마이크 이름의 일부 (예: K66)")
    parser.add_argument("--device", type=int, default=None,
                        help="마이크 장치 번호")
    parser.add_argument("--model", default="tiny",
                        choices=["tiny", "base", "small"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--free", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_input_devices()
        return

    device = args.device
    if args.name:
        found = find_device(args.name)
        if found is None:
            print(f"'{args.name}' 이름을 가진 입력 장치를 찾지 못했습니다.")
            list_input_devices()
            return
        device = found

    try:
        info = sd.query_devices(device, "input")
    except Exception as e:
        print(f"입력 장치를 열 수 없습니다: {e}")
        list_input_devices()
        print("--name 또는 --device 로 입력 장치를 지정하세요.")
        return

    try:
        rate, channels = negotiate(device)
    except Exception as e:
        print(f"녹음 설정을 정하지 못했습니다: {e}")
        return

    print(f"마이크: {info['name']}")
    if rate != TARGET_RATE:
        print(f"녹음 설정: {rate}Hz / {channels}채널"
              f"  ->  {TARGET_RATE}Hz 모노로 변환")
    else:
        print(f"녹음 설정: {rate}Hz / {channels}채널")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("\nfaster-whisper 가 설치되지 않았습니다.")
        print("  py -m pip install faster-whisper sounddevice")
        return

    print(f"\n모델을 불러오는 중입니다... ({args.model})")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    print("준비되었습니다.")

    if args.free:
        run_free(model, device, rate, channels)
    else:
        run_measure(model, args.model, device, rate, channels, args.repeat)


if __name__ == "__main__":
    main()
