import sys
import cv2
from pathlib import Path

# 1. 모듈 임포트를 위한 최상위 경로 추가
PROJECT_ROOT = "/home/dh1279/Desktop/kimGA_raspi/microprocessor-capstone-design/03.김가네_캡스톤_디자인/Raspi5_vision"
sys.path.append(PROJECT_ROOT)

from haptic_mvp.vision import HailoDetector

# 2. 파일 경로 설정
BASE_DIR = Path(__file__).resolve().parent
HEF_PATH = str(BASE_DIR / "models" / "yolov8n_custom.hef")
LABELS_PATH = str(BASE_DIR / "config" / "labels.txt") # json이 아닌 txt로 변경

def main():
    # 3. HailoDetector가 요구하는 config 딕셔너리 구성
    config = {
        "object_model": HEF_PATH,
        "labels_file": LABELS_PATH,
        "score_threshold": 0.5,
        "max_objects": 20
    }

    # 카메라 오픈
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("추론 시작 - 종료하려면 'q'를 누르세요.")

    # 4. config를 전달하여 NPU 감지기 실행
    with HailoDetector(config) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 팀원이 작성한 탐지 메서드 실행
            detections = detector.detect(frame)

            # 시각화 처리 (탐지된 객체 박스 그리기)
            height, width = frame.shape[:2]
            for det in detections:
                # detection.box는 (x1, y1, x2, y2) 정규화된 좌표입니다
                x1, y1, x2, y2 = det.box
                pt1 = (int(x1 * width), int(y1 * height))
                pt2 = (int(x2 * width), int(y2 * height))

                label_text = f"{det.label}: {det.score:.2f}"
                
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(frame, label_text, (pt1[0], max(pt1[1] - 8, 15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Raspberry Pi 5 + Hailo-10H", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()