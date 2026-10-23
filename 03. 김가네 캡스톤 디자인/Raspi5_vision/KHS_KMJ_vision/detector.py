import sys
import cv2
import math
from pathlib import Path

# 1. 모듈 임포트를 위한 최상위 경로 추가
PROJECT_ROOT = "/home/dh1279/Desktop/kimGA_raspi/microprocessor-capstone-design/03. 김가네 캡스톤 디자인/Raspi5_vision"
sys.path.append(PROJECT_ROOT)

import haptic_mvp.vision
from haptic_mvp.vision import HailoDetector, CameraSource
from haptic_mvp.models import Detection

# 손 추적 모듈 임포트[cite: 2]
from hand_tracker import HandTracker, draw_hand

# ==========================================
# [긴급 패치] 신뢰도 및 박스 좌표 오차 우회 (Monkey Patch)
# ==========================================
def patched_decode_hailo_nms(raw, labels, geometry, threshold=0.5, max_objects=20):
    found = []
    for label, rows in zip(labels, raw):
        for row in rows:
            if len(row) != 5:
                continue
            
            score = float(row[4])
            if not math.isfinite(score):
                continue
            score = max(0.0, min(1.0, score)) 
            if score < threshold:
                continue
            
            ymin = max(0.0, min(1.0, float(row[0])))
            xmin = max(0.0, min(1.0, float(row[1])))
            ymax = max(0.0, min(1.0, float(row[2])))
            xmax = max(0.0, min(1.0, float(row[3])))
            
            if xmin >= xmax or ymin >= ymax:
                continue

            box = haptic_mvp.vision.restore_box((ymin, xmin, ymax, xmax), geometry)
            if box is not None:
                found.append(Detection(label=label, box=box, score=score))
                
    return tuple(sorted(found, key=lambda item: item.score, reverse=True)[:max_objects])

haptic_mvp.vision.decode_hailo_nms = patched_decode_hailo_nms
# ==========================================

# 2. 파일 경로 설정
BASE_DIR = Path(__file__).resolve().parent
HEF_PATH = str(BASE_DIR / "models" / "best.hef")
LABELS_PATH = str(BASE_DIR / "config" / "labels.txt")
# PROJECT_ROOT를 기준으로 손 추적 모델 절대 경로 설정
HAND_MODEL_PATH = str(Path(PROJECT_ROOT) / "models" / "hand_landmarker.task")

def main():
    # 3. 통합 Config 설정
    config = {
        # Hailo 객체 인식 설정[cite: 1]
        "object_model": HEF_PATH,
        "labels_file": LABELS_PATH,
        "score_threshold": 0.5,
        "max_objects": 20,
        "infer_timeout_ms": 1500,
        
        # MediaPipe 손 추적 설정[cite: 2]
        "hand_model": HAND_MODEL_PATH,
        "hand": "Right",
        "worn_hand": "Right",
        "hand_score_threshold": 0.6,
        
        # 카메라 설정[cite: 1]
        "source": "picamera2", 
        "width": 1280,
        "height": 720,
        "fps": 15,
        "device": 0
    }

    print("초기화 중... (카메라 및 두 모델 웜업)")

    try:
        # 4. 카메라 및 두 모델 동시 초기화[cite: 1, 2]
        with CameraSource(config) as cam, \
             HailoDetector(config) as detector, \
             HandTracker(config) as tracker:
             
            print("추론 시작 - 비디오 창이 선택된 상태에서 'q'를 누르면 종료됩니다.")
            
            window_name = "Raspberry Pi 5 + Hailo & MediaPipe"
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name, 50, 50)
            
            frame_count = 0
            while True:
                read_result = cam.read()
                
                if read_result is None:
                    print("프레임을 읽어올 수 없습니다.")
                    break
                    
                frame, timestamp = read_result
                
                # 5. 프레임 공유 및 모델 추론[cite: 1, 2]
                obj_detections = detector.detect(frame)
                hand_result = tracker.detect(frame, timestamp)

                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                height, width = display_frame.shape[:2]
                
                # 6. 결과 터미널 출력 및 화면 그리기
                if obj_detections or hand_result is not None:
                    print(f"\n--- [프레임 {frame_count}] 탐지 결과 ---")

                if obj_detections:
                    for det in obj_detections:
                        x1, y1, x2, y2 = det.box
                        norm_center_x = (x1 + x2) / 2
                        norm_center_y = (y1 + y2) / 2
                        
                        pt1 = (int(x1 * width), int(y1 * height))
                        pt2 = (int(x2 * width), int(y2 * height))
                        center_x = int(norm_center_x * width)
                        center_y = int(norm_center_y * height)

                        # 터미널에 객체 정보 출력[cite: 1]
                        print(f" 🎯 객체: {det.label} (정확도: {det.score:.2f}) | 픽셀: ({center_x}, {center_y}) | 비율: X={norm_center_x:.2f}, Y={norm_center_y:.2f}")

                        # 화면에 바운딩 박스 및 중심점 그리기[cite: 1]
                        cv2.rectangle(display_frame, pt1, pt2, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"{det.label}: {det.score:.2f}", (pt1[0], max(pt1[1] - 8, 15)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                if hand_result is not None:
                    hx, hy = hand_result.point
                    h_center_x, h_center_y = int(hx * width), int(hy * height)
                    
                    # 터미널에 손 정보 출력[cite: 2]
                    print(f" 🖐️ 손: {hand_result.handedness} (정확도: {hand_result.score:.2f}) | 픽셀: ({h_center_x}, {h_center_y}) | 비율: X={hx:.2f}, Y={hy:.2f}")

                # 화면에 손 랜드마크 그리기[cite: 2]
                draw_hand(cv2, display_frame, tracker, hand_result)
                
                cv2.imshow(window_name, display_frame)
                frame_count += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n프로그램을 안전하게 종료합니다.")
    except Exception as e:
        print(f"실행 중 에러가 발생했습니다: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()