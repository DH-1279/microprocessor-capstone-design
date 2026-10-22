import sys
import cv2
from pathlib import Path

# 1. 모듈 임포트를 위한 최상위 경로 추가
PROJECT_ROOT = "/home/dh1279/Desktop/kimGA_raspi/microprocessor-capstone-design/03.김가네_캡스톤_디자인/Raspi5_vision"
sys.path.append(PROJECT_ROOT)

# 팀원이 작성한 탐지기 및 카메라 소스 모듈 임포트
from haptic_mvp.vision import HailoDetector, CameraSource

# 2. 파일 경로 설정
BASE_DIR = Path(__file__).resolve().parent
HEF_PATH = str(BASE_DIR / "models" / "best.hef")
LABELS_PATH = str(BASE_DIR / "config" / "labels.txt")

def main():
    # 3. HailoDetector 및 CameraSource가 공통으로 사용할 설정 딕셔너리
    config = {
        # 객체 탐지 모델 설정
        "object_model": HEF_PATH,
        "labels_file": LABELS_PATH,
        "score_threshold": 0.5,
        "max_objects": 20,
        "infer_timeout_ms": 1500,
        
        # 카메라 설정 (라즈베리 파이 카메라 모듈 사용)
        "source": "picamera2",  # USB 카메라일 경우 "opencv"로 변경
        "width": 640,
        "height": 480,
        "fps": 15,
        "device": 0
    }

    print("초기화 중... (카메라 웜업에 몇 초 정도 걸릴 수 있습니다)")

    # 4. NPU 탐지기와 카메라 소스를 함께 엽니다 (ExitStack 역할)
    try:
        with CameraSource(config) as cam, HailoDetector(config) as detector:
            print("추론 시작 - 종료하려면 카메라 창이 선택된 상태에서 'q'를 누르세요.")
            
            while True:
                # 5. 기존 cap.read() 대신 팀의 안정화된 카메라 모듈 사용
                read_result = cam.read()
                
                # 프레임을 읽어오지 못했으면 에러 출력 후 종료
                if read_result is None:
                    print("카메라에서 프레임을 읽어올 수 없습니다. 케이블 연결을 확인하세요.")
                    break
                    
                # read_result는 (frame, captured_at) 튜플을 반환합니다.
                frame, timestamp = read_result

                # 6. 추론 실행
                detections = detector.detect(frame)

                # 7. 시각화 처리 (OpenCV는 BGR을 사용하므로, RGB 프레임을 BGR로 변환하여 출력)
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                height, width = display_frame.shape[:2]
                
                for det in detections:
                    x1, y1, x2, y2 = det.box
                    pt1 = (int(x1 * width), int(y1 * height))
                    pt2 = (int(x2 * width), int(y2 * height))

                    label_text = f"{det.label}: {det.score:.2f}"
                    
                    cv2.rectangle(display_frame, pt1, pt2, (0, 255, 0), 2)
                    cv2.putText(display_frame, label_text, (pt1[0], max(pt1[1] - 8, 15)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("Raspberry Pi 5 + Hailo-10H (PiCamera2)", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"실행 중 에러가 발생했습니다: {e}")

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()