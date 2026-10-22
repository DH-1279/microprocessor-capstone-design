import cv2
import sys
import os

# 1. haptic_mvp 모듈을 불러오기 위해 프로젝트 최상위 경로 추가
PROJECT_ROOT = "/home/dh1279/Desktop/kimGA_raspi/microprocessor-capstone-design/03.김가네_캡스톤_디자인/Raspi5_vision"
sys.path.append(PROJECT_ROOT)

# 2. 이미 완벽하게 비동기/GStreamer 처리가 된 기존 클래스 임포트
from haptic_mvp.vision import HailoDetector

# 3. 모델 및 라벨 설정 (json 대신 변환했던 txt 사용)
config = {
    "labels_file": f"{PROJECT_ROOT}/KHS_object_detection/labels.txt",
    "model_file": f"{PROJECT_ROOT}/KHS_object_detection/models/best.hef"
    # vision.py 내부 구조에 따라 키 이름(model_file 등)은 다를 수 있으니 확인 후 맞춰주세요.
}

def main():
    CAMERA_ID = 0
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("추론 시작 - 종료하려면 'q'를 누르세요.")

    # 4. 복잡한 VDevice, ConfigureParams 설정 없이 래퍼(Wrapper) 클래스 활용
    with HailoDetector(config) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 추론 수행 (vision.py 내부에 정의된 실제 메서드명으로 변경하세요. 예: predict, infer, run 등)
            detections = detector.detect(frame)
            
            # 여기서 바운딩 박스를 그리고 화면에 출력합니다.
            # (haptic_mvp 내부에 시각화 함수가 있다면 그것을 호출하셔도 됩니다)
            
            cv2.imshow("Raspberry Pi 5 + Hailo-10H", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()