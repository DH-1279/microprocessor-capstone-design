"""Hailo/camera smoke test with optional desktop preview; no hand/audio/BLE.

Uses local HEF and ordered one-name-per-line labels already prepared by the
operator. It never downloads weights or changes project configuration.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from itertools import count
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haptic_mvp.config import load_config
from haptic_mvp.vision import CameraSource, HailoDetector


WINDOW = "KIMGANE objects (q / Esc: quit)"


@contextmanager
def preview_window(enabled):
    if not enabled:
        yield None
        return
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("영상 창은 라즈베리파이 바탕화면의 터미널에서 실행하세요. PuTTY에서는 --preview를 빼세요.")
    import cv2

    if hasattr(cv2, "currentUIFramework") and not cv2.currentUIFramework():
        raise RuntimeError("현재 OpenCV에 GUI 백엔드가 없습니다. cv2.__file__과 getBuildInformation()을 확인하세요.")
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        yield cv2
    finally:
        cv2.destroyAllWindows()


def show_preview(cv2, rgb, objects, elapsed):
    # Convert a copy for OpenCV; never draw on the RGB inference input.
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]
    for obj in objects:
        x1, y1, x2, y2 = obj.box
        first = (max(0, min(width - 1, int(x1 * width))), max(0, min(height - 1, int(y1 * height))))
        last = (max(0, min(width - 1, int(x2 * width))), max(0, min(height - 1, int(y2 * height))))
        cv2.rectangle(frame, first, last, (0, 255, 0), 2)
        cv2.putText(frame, f"{obj.label} {obj.score:.2f}", (first[0], max(18, first[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"detect {elapsed:.1f} ms | objects {len(objects)} | q/Esc: quit", (8, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imshow(WINDOW, frame)
    return cv2.waitKey(1) & 0xFF in (ord("q"), 27)


def frame_count(value: str) -> int:
    count = int(value)
    if not 1 <= count <= 300:
        raise argparse.ArgumentTypeError("frames must be between 1 and 300")
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="카메라·Hailo 단독 추론 점검")
    parser.add_argument("--model", type=Path, default=Path("/tmp/kimgane-khs-check/best.hef"))
    parser.add_argument("--labels", type=Path, default=Path("/tmp/kimgane-khs-check/labels.txt"))
    parser.add_argument("--frames", type=frame_count, help="1~300; 기본은 텍스트 10프레임, 미리보기는 종료 키까지")
    parser.add_argument("--preview", action="store_true", help="Pi 바탕화면에 탐지 박스·이름 표시 (q/Esc 종료)")
    args = parser.parse_args(argv)
    limit = args.frames if args.frames is not None else None if args.preview else 10
    for path in (args.model, args.labels):
        if not path.is_file() or not path.stat().st_size:
            parser.error(f"파일이 없거나 비어 있습니다: {path}; HEF와 라벨을 먼저 준비하세요.")

    config = dict(load_config()["vision"])
    config.update(object_model=str(args.model.resolve()), labels_file=str(args.labels.resolve()))
    print("HAT에 모델을 로딩합니다.", flush=True)
    try:
        with preview_window(args.preview) as preview, HailoDetector(config) as detector:
            print(f"모델 로딩 성공 | 입력 {detector.model_width}×{detector.model_height} | "
                  f"클래스 {len(detector.labels)}개", flush=True)
            with CameraSource(config) as camera:
                for index in count(1):
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
                    if preview is None or index == 1 or index % 30 == 0:
                        progress = f"{index}/{limit}" if limit is not None else str(index)
                        print(f"프레임 {progress} | 추론 {elapsed:.1f}ms | 탐지 {len(objects)}개", flush=True)
                        for obj in objects:
                            box = tuple(round(value, 3) for value in obj.box)
                            print(f"  {obj.label} | 신뢰도 {obj.score:.2f} | 좌표 {box}", flush=True)
                    if preview is not None and show_preview(preview, rgb, objects, elapsed):
                        break
                    if limit is not None and index >= limit:
                        break
                    if preview is None:
                        time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n사용자 요청으로 점검을 중단했습니다.", flush=True)
        return 130
    print("카메라·HAT 단독 시험 종료", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
