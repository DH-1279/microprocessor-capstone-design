import sys
import cv2
import math
import time
from pathlib import Path

# 1. 모듈 임포트를 위한 최상위 경로 추가
PROJECT_ROOT = "/home/dh1279/Desktop/kimGA_raspi/microprocessor-capstone-design/03.김가네_캡스톤_디자인/Raspi5_vision"
sys.path.append(PROJECT_ROOT)

import haptic_mvp.vision
from haptic_mvp.vision import HailoDetector, CameraSource
from haptic_mvp.models import Detection

# ==========================================
# [긴급 패치] 신뢰도 및 박스 좌표 오차 우회 (Monkey Patch)
# ==========================================
def patched_decode_hailo_nms(raw, labels, geometry, threshold=0.5, max_objects=20):
    found = []
    for label, rows in zip(labels, raw):
        for row in rows:
            if len(row) != 5:
                continue
            
            # 1. 신뢰도(Score) 클리핑
            score = float(row[4])
            if not math.isfinite(score):
                continue
            score = max(0.0, min(1.0, score)) 
            if score < threshold:
                continue
            
            # 2. 박스 좌표(ymin, xmin, ymax, xmax) 클리핑 추가
            ymin = max(0.0, min(1.0, float(row[0])))
            xmin = max(0.0, min(1.0, float(row[1])))
            ymax = max(0.0, min(1.0, float(row[2])))
            xmax = max(0.0, min(1.0, float(row[3])))
            
            # 좌표가 비정상적으로 뒤집힌 경우(min이 max보다 큰 경우) 무시
            if xmin >= xmax or ymin >= ymax:
                continue

            # 보정된 좌표로 restore_box 호출
            box = haptic_mvp.vision.restore_box((ymin, xmin, ymax, xmax), geometry)
            if box is not None:
                found.append(Detection(label=label, box=box, score=score))
                
    return tuple(sorted(found, key=lambda item: item.score, reverse=True)[:max_objects])

haptic_mvp.vision.decode_hailo_nms = patched_decode_hailo_nms
# ==========================================

# 2. 파일 경로 설정
BASE_DIR = Path(__file__).resolve().parent
HEF_PATH = str(BASE_DIR / "models" / "yolov8n_custom.hef")
LABELS_PATH = str(BASE_DIR / "config" / "labels.txt")

def main():
    config = {
        "object_model": HEF_PATH,
        "labels_file": LABELS_PATH,
        "score_threshold": 0.5,
        "max_objects": 20,
        "infer_timeout_ms": 1500,
        
        "source": "picamera2", 
        "width": 640,
        "height": 480,
        "fps": 15,
        "device": 0
    }

    print("초기화 중... (카메라 웜업)")

    try:
        with CameraSource(config) as cam, HailoDetector(config) as detector:
            print("추론 시작 - (Ctrl+C를 누르면 강제 종료됩니다)")
            
            frame_count = 0
            while True:
                read_result = cam.read()
                
                if read_result is None:
                    print("프레임을 읽어올 수 없습니다.")
                    break
                    
                frame, timestamp = read_result

                detections = detector.detect(frame)

                if detections:
                    print(f"--- [프레임 {frame_count}] 탐지된 물체 ---")
                    for det in detections:
                        print(f" 🎯 {det.label} (정확도: {det.score:.2f})")
                
                if frame_count % 10 == 0 and detections:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    height, width = display_frame.shape[:2]
                    
                    for det in detections:
                        x1, y1, x2, y2 = det.box
                        pt1 = (int(x1 * width), int(y1 * height))
                        pt2 = (int(x2 * width), int(y2 * height))
                        cv2.rectangle(display_frame, pt1, pt2, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"{det.label}: {det.score:.2f}", (pt1[0], max(pt1[1] - 8, 15)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imwrite("latest_detect.jpg", display_frame)

                frame_count += 1
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n프로그램을 안전하게 종료합니다.")
    except Exception as e:
        print(f"실행 중 에러가 발생했습니다: {e}")

if __name__ == "__main__":
    main()