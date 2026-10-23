"""
뒤집힘 판별과 손가락 끝 좌표 확인 (웹캠 전용)

models.py를 고치기 전에, 두 가지가 실제로 되는지 먼저 본다.

1. 손등인지 손바닥인지 카메라만으로 알 수 있는가
   MediaPipe는 왼손·오른손만 알려주고 앞뒤는 알려주지 않는다.
   손목(0)·검지뿌리(5)·새끼뿌리(17) 세 점이 화면에서 도는 방향으로
   직접 계산한다. 손을 뒤집으면 세 점의 순서가 반대가 되므로 부호가
   바뀐다.

2. 손가락 끝 다섯 점을 쓸 만한가
   4, 8, 12, 16, 20번 관절이다. MediaPipe가 이미 내보내고 있으므로
   뽑기만 하면 되지만, 장갑을 낀 상태에서 값이 튀지 않는지 봐야 한다.

준비
    pip install mediapipe opencv-python
    hand_landmarker.task 를 같은 폴더나 models/ 에 둔다

실행
    py palm_test.py
    py palm_test.py --label 왼손_장갑        조건 이름을 붙여 기록
    py palm_test.py --hand Left             추적할 손

조작
    s   측정 시작 / 중지
    f   현재 자세를 '손등'으로 표시하고 측정  (앞면)
    b   현재 자세를 '손바닥'으로 표시하고 측정 (뒷면)
    q   종료

f 와 b 로 정답을 알려주면, 계산이 그 자세를 맞혔는지 세어 준다.
결과는 palm_results.csv 에 쌓인다.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

RESULT_CSV = "palm_results.csv"

# 손바닥 면을 이루는 세 점. 손가락을 구부려도 거의 움직이지 않는다.
WRIST, INDEX_BASE, PINKY_BASE = 0, 5, 17

# 손가락 끝
FINGERTIPS = (4, 8, 12, 16, 20)
TIP_NAMES = ("엄지", "검지", "중지", "약지", "소지")

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def find_model():
    for path in ("hand_landmarker.task", "models/hand_landmarker.task",
                 os.path.join("models", "hand_landmarker.task")):
        if os.path.isfile(path):
            return path
    return None


def palm_normal(landmarks):
    """
    손목에서 본 두 방향의 외적으로 손바닥 면의 방향을 구한다.

    화면 좌표에서 z 성분만 보면 되므로 2차원 외적으로 충분하다.
    부호가 양수면 세 점이 한 방향으로 돌고, 음수면 반대로 돈다.
    손을 뒤집으면 이 부호가 바뀐다.

    크기도 함께 돌려준다. 세 점이 일직선에 가까우면 크기가 0에
    가까워지고, 그때는 판정을 믿을 수 없다.
    """
    wrist = landmarks[WRIST]
    index = landmarks[INDEX_BASE]
    pinky = landmarks[PINKY_BASE]

    ax, ay = index.x - wrist.x, index.y - wrist.y
    bx, by = pinky.x - wrist.x, pinky.y - wrist.y
    cross = ax * by - ay * bx
    return cross


def classify(cross, worn_hand, threshold):
    """
    부호를 손등·손바닥으로 옮긴다.

    같은 부호라도 왼손과 오른손이 반대다. 다만 MediaPipe의 좌우 판정은
    장갑을 끼면 흔들리므로 그 값을 쓰지 않는다. 장갑을 어느 손에 끼는지는
    우리가 정해 두는 값이므로, 설정에서 받은 worn_hand 를 쓴다.

    크기가 작으면 판정하지 않는다. 세 점이 일직선에 가까울 때다.
    """
    if abs(cross) < threshold:
        return "불확실", abs(cross)
    facing_up = cross > 0
    if worn_hand == "Left":
        facing_up = not facing_up
    return ("손등" if facing_up else "손바닥"), abs(cross)


def draw(frame, landmarks, handedness, verdict, strength, tips):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 3, (0, 0, 200), -1)

    # 손바닥 면을 이루는 세 점을 굵게
    for index in (WRIST, INDEX_BASE, PINKY_BASE):
        cv2.circle(frame, points[index], 9, (255, 0, 255), 2)
    cv2.line(frame, points[WRIST], points[INDEX_BASE], (255, 0, 255), 2)
    cv2.line(frame, points[WRIST], points[PINKY_BASE], (255, 0, 255), 2)

    # 손가락 끝에 이름
    for index, name in zip(FINGERTIPS, TIP_NAMES):
        x, y = points[index]
        cv2.circle(frame, (x, y), 7, (0, 180, 255), -1)
        cv2.putText(frame, name, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 180, 255), 1)

    color = (0, 200, 0) if verdict != "불확실" else (0, 140, 255)
    cv2.putText(frame, f"MP:{handedness}  ->  {verdict}  ({strength:.4f})",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def save(rows):
    is_new = not os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["측정시각", "조건", "정답", "전체프레임",
                             "판정성공", "불확실", "성공률(%)",
                             "좌우오판", "좌우오판률(%)",
                             "외적크기평균", "손가락끝흔들림평균"])
        writer.writerows(rows)
    print(f"\n{RESULT_CSV} 에 저장했습니다.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="측정", help="조건 이름")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--worn", default="Left",
                        choices=["Left", "Right"],
                        help="장갑을 낀 손. MediaPipe 판정 대신 이 값을 쓴다")
    parser.add_argument("--hand", default="Any",
                        choices=["Left", "Right", "Any"],
                        help="추적할 손. Any 는 손이 하나일 때만 받는다")
    parser.add_argument("--threshold", type=float, default=0.004,
                        help="이 값보다 작으면 판정하지 않는다")
    args = parser.parse_args()

    model = find_model()
    if not model:
        print("hand_landmarker.task 를 찾지 못했습니다.")
        print("아래 주소에서 받아 같은 폴더나 models/ 에 두세요.")
        print("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/"
              "hand_landmarker.task")
        return 1

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"카메라 {args.camera} 를 열 수 없습니다.")
        return 1

    print("\n손등을 보이면 f, 손바닥을 보이면 b 를 눌러 측정하세요.")
    print("정답 없이 그냥 재려면 s 를 누릅니다. q 로 끝냅니다.\n")

    rows = []
    measuring = False
    truth = None
    total = correct = unsure = mismatch = 0
    strengths = []
    tip_history = []
    last_timestamp = -1

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = max(last_timestamp + 1, int(time.time() * 1000))
            last_timestamp = timestamp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, timestamp)

            verdict, strength, tips = None, 0.0, None
            for landmarks, handed in zip(result.hand_landmarks,
                                         result.handedness):
                name = handed[0].category_name
                if args.hand != "Any" and name != args.hand:
                    continue
                cross = palm_normal(landmarks)
                verdict, strength = classify(cross, args.worn, args.threshold)
                if name != args.worn:
                    mismatch += 1
                tips = [(lm.x, lm.y) for lm in
                        (landmarks[i] for i in FINGERTIPS)]
                draw(frame, landmarks, name, verdict, strength, tips)
                break

            if measuring:
                total += 1
                if verdict == "불확실" or verdict is None:
                    unsure += 1
                elif truth is None or verdict == truth:
                    correct += 1
                if strength:
                    strengths.append(strength)
                if tips:
                    tip_history.append(tips)

                rate = correct / total * 100 if total else 0
                head = f"MEASURING [{truth or '정답없음'}]  {correct}/{total}"
                cv2.putText(frame, f"{head}  ({rate:.0f}%)  불확실 {unsure}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 255), 2)
            else:
                cv2.putText(frame, "f=손등  b=손바닥  s=정답없이  q=종료",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 200), 1)

            cv2.imshow("palm test", frame)
            key = cv2.waitKey(1) & 0xFF

            if key not in (ord("q"), ord("s"), ord("f"), ord("b")):
                continue

            if key in (ord("s"), ord("f"), ord("b")) or (
                    key == ord("q") and measuring):
                if measuring:
                    # 손가락 끝이 얼마나 흔들렸는지
                    jitter = 0.0
                    if len(tip_history) > 1:
                        diffs = []
                        for before, after in zip(tip_history, tip_history[1:]):
                            for (x1, y1), (x2, y2) in zip(before, after):
                                diffs.append(abs(x2 - x1) + abs(y2 - y1))
                        jitter = sum(diffs) / len(diffs) if diffs else 0.0
                    mean_strength = (sum(strengths) / len(strengths)
                                     if strengths else 0.0)
                    rate = correct / total * 100 if total else 0

                    print(f"\n[{args.label}] 정답 {truth or '없음'}")
                    mis_rate = mismatch / total * 100 if total else 0
                    print(f"  전체 {total}프레임  판정성공 {correct}  "
                          f"불확실 {unsure}")
                    print(f"  MediaPipe 좌우 오판 {mismatch} ({mis_rate:.1f}%)")
                    print(f"  성공률 {rate:.1f}%")
                    print(f"  외적 크기 평균 {mean_strength:.4f}")
                    print(f"  손가락 끝 흔들림 평균 {jitter:.5f}")

                    rows.append([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        args.label, truth or "없음", total, correct, unsure,
                        f"{rate:.1f}", mismatch, f"{mis_rate:.1f}",
                        f"{mean_strength:.4f}", f"{jitter:.5f}",
                    ])
                    measuring = False
                elif key != ord("q"):
                    truth = {"f": "손등", "b": "손바닥"}.get(chr(key))
                    total = correct = unsure = mismatch = 0
                    strengths = []
                    tip_history = []
                    measuring = True
                    print(f"[{args.label}] 측정 시작  정답 {truth or '없음'}")

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    if rows:
        save(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
