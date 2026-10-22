"""
손 인식률 측정 도구 (웹캠 전용)

노트북 웹캠으로 MediaPipe Hand Landmarker의 인식률을 측정한다.

**라즈베리파이의 CSI 카메라에는 쓰지 않는다.** OpenCV 웹캠 입력과 화면
창을 쓰므로 Picamera2 경로와 SSH 환경에 맞지 않는다. 파이에서의 측정은
공통 카메라 입력을 쓰는 별도 도구가 필요하며 아직 만들지 않았다.

이 도구는 카메라를 직접 연다. 통합 실행과 동시에 쓰지 않는다.
맨손 / 장갑 / 색상별 / 자세별 조건을 바꿔가며 같은 방식으로 재고,
결과를 results.csv에 쌓는다.

준비
    pip install mediapipe opencv-python

    모델 파일을 같은 폴더에 받는다 (한 번만):
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

실행
    python hand_test.py --label 맨손
    python hand_test.py --label 검정장갑
    python hand_test.py --label 흰장갑_손등

조작
    s  측정 시작 / 중지
    q  종료
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"
RESULT_CSV = "results.csv"

# 손 랜드마크 연결선 (21개 점을 손 모양으로 잇는다)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),          # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),     # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),   # 약지
    (13, 17), (17, 18), (18, 19), (19, 20),  # 소지
    (0, 17),                                  # 손바닥 아래
]


def draw_hand(frame, landmarks, handedness):
    """인식된 손의 관절과 연결선을 그린다."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    # 손 기준점: 손목(0)과 손가락 뿌리(5, 9, 13, 17)의 평균
    base_idx = [0, 5, 9, 13, 17]
    cx = sum(points[i][0] for i in base_idx) // len(base_idx)
    cy = sum(points[i][1] for i in base_idx) // len(base_idx)
    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), 2)
    cv2.putText(frame, handedness, (cx + 15, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return cx, cy


def save_result(label, total, detected, elapsed):
    """측정 결과를 CSV에 한 줄 추가한다."""
    rate = detected / total * 100 if total else 0
    fps = total / elapsed if elapsed else 0
    is_new = not os.path.exists(RESULT_CSV)

    with open(RESULT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["측정시각", "조건", "전체프레임", "검출프레임",
                             "인식률(%)", "측정시간(초)", "FPS"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label, total, detected,
            f"{rate:.1f}", f"{elapsed:.1f}", f"{fps:.1f}",
        ])

    print(f"\n[{label}] 전체 {total}프레임 중 {detected}프레임 검출")
    print(f"인식률 {rate:.1f}%  |  측정시간 {elapsed:.1f}초  |  {fps:.1f} FPS")
    print(f"{RESULT_CSV} 에 저장했습니다.\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="측정",
                        help="측정 조건 이름 (예: 맨손, 검정장갑)")
    parser.add_argument("--camera", type=int, default=0,
                        help="카메라 번호. 외장 캠이면 1 이상")
    parser.add_argument("--hand", default="Any",
                        choices=["Right", "Left", "Any"],
                        help="추적할 손")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일이 없습니다: {MODEL_PATH}")
        print("아래 주소에서 받아 같은 폴더에 두세요.")
        print("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        return 1

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
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

    measuring = False
    total = 0
    detected = 0
    start_time = 0.0

    print("s 키로 측정 시작/중지, q 키로 종료합니다.")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int(time.time() * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # 지정한 손만 세기
            hands_found = 0
            for landmarks, handed in zip(result.hand_landmarks, result.handedness):
                name = handed[0].category_name
                if args.hand != "Any" and name != args.hand:
                    continue
                draw_hand(frame, landmarks, name)
                hands_found += 1

            if measuring:
                total += 1
                if hands_found > 0:
                    detected += 1

                rate = detected / total * 100
                cv2.putText(frame, f"MEASURING  {detected}/{total}  ({rate:.1f}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
            else:
                status = "DETECTED" if hands_found else "NOT DETECTED"
                color = (0, 200, 0) if hands_found else (0, 0, 255)
                cv2.putText(frame, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, "press s to start", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("hand test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if measuring:
                    save_result(args.label, total, detected,
                                time.time() - start_time)
                    measuring = False
                else:
                    total = detected = 0
                    start_time = time.time()
                    measuring = True
                    print(f"[{args.label}] 측정 시작")

    if measuring and total:
        save_result(args.label, total, detected, time.time() - start_time)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
