"""Headless Hailo/camera smoke test; no hand model, audio, controller or BLE.

Uses local HEF and ordered one-name-per-line labels already prepared by the
operator. It never downloads weights or changes project configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haptic_mvp.config import load_config
from haptic_mvp.vision import CameraSource, HailoDetector


def frame_count(value: str) -> int:
    count = int(value)
    if not 1 <= count <= 300:
        raise argparse.ArgumentTypeError("frames must be between 1 and 300")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="카메라·Hailo 단독 추론 점검 (영상 창 없음)")
    parser.add_argument("--model", type=Path, default=Path("/tmp/kimgane-khs-check/best.hef"))
    parser.add_argument("--labels", type=Path, default=Path("/tmp/kimgane-khs-check/labels.txt"))
    parser.add_argument("--frames", type=frame_count, default=10)
    args = parser.parse_args()
    for path in (args.model, args.labels):
        if not path.is_file() or not path.stat().st_size:
            parser.error(f"파일이 없거나 비어 있습니다: {path}; HEF와 라벨을 먼저 준비하세요.")

    config = dict(load_config()["vision"])
    config.update(object_model=str(args.model.resolve()), labels_file=str(args.labels.resolve()))
    print("HAT에 모델을 로딩합니다.", flush=True)
    try:
        with HailoDetector(config) as detector:
            print(f"모델 로딩 성공 | 입력 {detector.model_width}×{detector.model_height} | "
                  f"클래스 {len(detector.labels)}개", flush=True)
            with CameraSource(config) as camera:
                for index in range(args.frames):
                    item = camera.read()
                    if item is None:
                        raise RuntimeError("카메라 프레임이 없습니다.")
                    rgb, _ = item
                    started = time.perf_counter()
                    try:
                        objects = detector.detect(rgb)
                    except ValueError:
                        # Diagnostic only: never substitute malformed results.
                        print("좌표 해석 실패: 원시 NMS 표본을 출력합니다.", flush=True)
                        try:
                            raw = detector.bindings.output().get_buffer(tf_format=False)
                            for label, rows in zip(detector.labels, raw):
                                if len(rows):
                                    print(label, repr(rows[:3]), flush=True)
                        except Exception as error:
                            print(f"원시 결과 조회 실패: {error}", flush=True)
                        raise
                    elapsed = (time.perf_counter() - started) * 1000
                    print(f"프레임 {index + 1}/{args.frames} | 추론 {elapsed:.1f}ms | "
                          f"탐지 {len(objects)}개", flush=True)
                    for obj in objects:
                        box = tuple(round(value, 3) for value in obj.box)
                        print(f"  {obj.label} | 신뢰도 {obj.score:.2f} | 좌표 {box}", flush=True)
                    time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n사용자 요청으로 점검을 중단했습니다.", flush=True)
        return 130
    print("카메라·HAT 단독 시험 종료", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
