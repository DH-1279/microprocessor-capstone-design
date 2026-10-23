"""Run HandTracker on the Pi's camera alone, with no detector.

`haptic_mvp run` starts object detection and hand tracking together, so it
stops before the camera opens when the object model is missing. This opens the
camera by itself and only runs the hand path, so the hand side can be checked
while the detector is not ready.

It prints a line a second and a summary at the end. --show also draws the
skeleton, the fingertips and the palm triangle, which needs a display:

    python -m KMJ_hand_audio_ui.tools.pi_hand_check
    DISPLAY=:0 python -m KMJ_hand_audio_ui.tools.pi_hand_check --show
    DISPLAY=:0 python -m KMJ_hand_audio_ui.tools.pi_hand_check --show --seconds 600
    python -m KMJ_hand_audio_ui.tools.pi_hand_check --worn Left --hand Any

`DISPLAY=:0` sends the window to the screen attached to the Pi. Without it an
SSH session has nowhere to draw and OpenCV aborts.

**Single use only.** It opens the camera, so it must not run while
`haptic_mvp run` is running.
"""

from __future__ import annotations

import argparse
from collections import Counter
import sys
import time

from ..hand_tracker import (
    FINGERTIP_LANDMARKS, HandTracker, fingertip_points, palm_facing_up,
)

# Landmark pairs that outline a hand.
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
PALM = (0, 5, 17)
TIP_NAMES = ("엄지", "검지", "중지", "약지", "소지")


class LandmarkTap:
    """
    Keep the landmarks HandTracker just used, without changing it.

    HandTracker calls the MediaPipe landmarker and hands back only a Hand, so
    the raw joints are gone by the time this tool sees the result. Wrapping
    the landmarker's own call keeps them for drawing. The guidance path is
    untouched: the wrapper returns exactly what it received.
    """

    def __init__(self, tracker):
        self.result = None
        self._original = tracker.tracker.detect_for_video
        tracker.tracker.detect_for_video = self._capture

    def _capture(self, image, timestamp):
        self.result = self._original(image, timestamp)
        return self.result

    def landmarks(self):
        if self.result is None or not self.result.hand_landmarks:
            return None
        return self.result.hand_landmarks[0]

    def reported_hand(self):
        """What MediaPipe called this hand, before any voting."""
        if self.result is None or not self.result.handedness:
            return None
        return self.result.handedness[0][0].category_name


def load_vision_config(config_path):
    """Use the project's own loader so model paths and defaults match."""
    try:
        from haptic_mvp.config import load_config
    except ImportError:
        raise SystemExit("Raspi5_vision 폴더에서 실행하세요.")
    return load_config(config_path)["vision"]


def open_camera(config):
    """
    Picamera2 first, then OpenCV.

    The Pi uses a CSI camera through Picamera2; a laptop webcam goes through
    OpenCV. Returns a function that gives one RGB frame, and a closer.
    """
    width = int(config.get("width", 640))
    height = int(config.get("height", 480))

    try:
        from picamera2 import Picamera2
    except ImportError:
        Picamera2 = None

    if Picamera2 is not None and config.get("source", "picamera2") == "picamera2":
        camera = Picamera2()
        camera.configure(camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}))
        camera.start()
        time.sleep(0.5)
        return camera.capture_array, camera.close, "picamera2"

    import cv2

    capture = cv2.VideoCapture(int(config.get("device", 0)))
    if not capture.isOpened():
        raise SystemExit("카메라를 열 수 없습니다.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read():
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("프레임을 읽지 못했습니다.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return read, capture.release, "opencv"


def facing_text(value):
    return {True: "손등", False: "손바닥", None: "앞뒤 보류"}.get(value, "?")


def draw(cv2, frame, landmarks, hand, facing, reported, rate):
    """Skeleton, palm triangle, fingertips and the point the controller uses."""
    height, width = frame.shape[:2]
    points = [(int(mark.x * width), int(mark.y * height)) for mark in landmarks]

    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 3, (0, 0, 200), -1)

    # The three joints the back/palm decision uses.
    for index in PALM:
        cv2.circle(frame, points[index], 9, (255, 0, 255), 2)
    cv2.line(frame, points[PALM[0]], points[PALM[1]], (255, 0, 255), 2)
    cv2.line(frame, points[PALM[0]], points[PALM[2]], (255, 0, 255), 2)

    for index, name in zip(FINGERTIP_LANDMARKS, TIP_NAMES):
        x, y = points[index]
        cv2.circle(frame, (x, y), 7, (0, 180, 255), -1)
        cv2.putText(frame, name, (x + 9, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 180, 255), 1)

    # What the controller would actually guide with.
    if hand is not None:
        x, y = hand.point
        cv2.circle(frame, (int(x * width), int(y * height)), 12,
                   (255, 255, 0), 3)

    side = hand.handedness if hand is not None else (reported or "-")
    colour = (0, 255, 255) if hand is not None else (140, 140, 140)
    cv2.putText(frame, f"{rate:.0f}%   {side}   {facing_text(facing)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    if reported and hand is not None and reported != hand.handedness:
        cv2.putText(frame, f"MediaPipe: {reported} (보정됨)", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)


def tip_line(tips):
    """Fingertip coordinates, short enough for one terminal line."""
    if not tips:
        return ""
    return "  ".join(f"{name} {x:.2f},{y:.2f}"
                     for name, (x, y) in zip(TIP_NAMES, tips))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--hand", default=None, choices=["Left", "Right", "Any"],
                        help="추적할 손. 설정값보다 우선한다")
    parser.add_argument("--worn", default=None, choices=["Left", "Right"],
                        help="장갑 낀 손. 앞뒤 판정에 쓴다")
    parser.add_argument("--hold", type=float, default=None,
                        help="짧은 소실을 유지할 시간(ms). 0이면 보정 없음")
    parser.add_argument("--show", action="store_true",
                        help="영상 창을 띄운다. DISPLAY 가 있어야 한다")
    parser.add_argument("--tips", action="store_true",
                        help="손가락 끝 좌표를 매 줄에 함께 찍는다")
    args = parser.parse_args()

    config = load_vision_config(args.config)
    if args.hand:
        config = {**config, "hand": args.hand}
    if args.worn:
        config = {**config, "worn_hand": args.worn}
    stability = {"hold_ms": args.hold} if args.hold is not None else None

    cv2 = None
    if args.show:
        import cv2 as cv2_module

        cv2 = cv2_module

    read_frame, close_camera, backend = open_camera(config)
    worn = config.get("worn_hand", config.get("hand", "Right"))
    print(f"카메라 {backend}  "
          f"{config.get('width', 640)}x{config.get('height', 480)}")
    print(f"추적 대상 {config.get('hand', 'Right')}   장갑 손 {worn}")
    if cv2 is not None:
        print("영상 창을 띄웁니다. 창에서 q 를 누르면 끝납니다.")
    print(f"{args.seconds:.0f}초 동안 봅니다. 카메라 앞에서 손을 움직여 주세요.\n")

    frames = detected = 0
    sides = Counter()
    facings = Counter()
    mismatches = 0
    started = time.monotonic()
    next_report = started + 1.0
    stats = None

    try:
        with HandTracker(config, stability) as tracker:
            tap = LandmarkTap(tracker)
            frame_id = 0

            while time.monotonic() - started < args.seconds:
                rgb = read_frame()
                frame_id += 1
                frames += 1
                hand = tracker.detect(rgb, time.monotonic(), frame_id)

                landmarks = tap.landmarks()
                reported = tap.reported_hand()
                facing = (palm_facing_up(landmarks, worn)
                          if landmarks is not None else None)
                tips = (fingertip_points(landmarks)
                        if landmarks is not None else ())

                if hand is not None:
                    detected += 1
                    sides[hand.handedness] += 1
                    facings[facing_text(facing)] += 1
                    if reported and reported != hand.handedness:
                        mismatches += 1

                if cv2 is not None:
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    rate = detected / frames * 100 if frames else 0
                    if landmarks is not None:
                        draw(cv2, frame, landmarks, hand, facing, reported, rate)
                    else:
                        cv2.putText(frame, f"{rate:.0f}%   손 없음", (10, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (140, 140, 140), 2)
                    cv2.imshow("KMJ hand check (q: stop)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                now = time.monotonic()
                if now >= next_report:
                    next_report = now + 1.0
                    rate = detected / frames * 100 if frames else 0
                    if hand is None:
                        print(f"  {now - started:4.0f}초  {rate:5.1f}%  손 없음")
                    else:
                        x, y = hand.point
                        print(f"  {now - started:4.0f}초  {rate:5.1f}%  "
                              f"대표점 {x:.2f},{y:.2f}  {hand.handedness}  "
                              f"{facing_text(facing)}")
                        if args.tips and tips:
                            print(f"        {tip_line(tips)}")

            stats = tracker.stats()
    finally:
        close_camera()
        if cv2 is not None:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - started
    rate = detected / frames * 100 if frames else 0
    print(f"\n{'=' * 56}")
    print(f"프레임 {frames}   손 검출 {detected} ({rate:.1f}%)   "
          f"{frames / elapsed:.1f} fps")
    if sides:
        print("좌우 판정   " + "   ".join(f"{k} {v}" for k, v in sides.items()))
    if facings:
        print("앞뒤 판정   " + "   ".join(f"{k} {v}" for k, v in facings.items()))
    if detected:
        print(f"MediaPipe 좌우 오판   {mismatches} "
              f"({mismatches / detected * 100:.1f}%)")
    if stats is not None:
        print(f"보정   흡수한 뒤집힘 {stats.flips_absorbed}   "
              f"판정 전환 {stats.switches}   "
              f"유지 {stats.held} ({stats.hold_ratio * 100:.1f}%)")
    print("=" * 56)

    if stats is not None and stats.hold_ratio > 0.3:
        print("\n유지 비율이 높습니다. 이미 사라진 손으로 안내할 수 있으니")
        print("--hold 0 으로 다시 재고 비교해 보세요.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
