# Raspi5_vision/KHS_object_detection/detector.py

import cv2
import numpy as np
from ultralytics import YOLO

# TODO: 실제 통합 시에는 haptic_mvp 모듈에서 Detection을 가져옵니다.
# from haptic_mvp.models import Detection

# 단독 테스트를 위한 임시 Detection 클래스 (추후 통합 시 주석 처리)
class Detection:
    def __init__(self, label: str, box: tuple, score: float):
        self.label = label
        self.box = box  # (x1, y1, x2, y2) 0~1 정규화 좌표
        self.score = score
        
    def __repr__(self):
        return f"Detection(label='{self.label}', box={self.box}, score={self.score:.2f})"

class YoloDetector:
    def __init__(self, model_path="models/best.pt", conf_threshold=0.5):
        """
        초기화 시 YOLO 모델을 로드합니다.
        (추후 라즈베리파이 환경에서는 HailoRT 기반의 HEF 모델 로직으로 변경 예정)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = YOLO(self.model_path)
        print(f"[KHS_Detector] YOLO 모델 로드 완료: {self.model_path}")

    def process(self, frame: np.ndarray) -> list:
        """
        통합 런타임에서 넘겨받은 공통 RGB 프레임을 추론하여 Detection 객체 리스트를 반환합니다.
        """
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]
        
        # verbose=False로 설정하여 매 프레임마다 터미널에 로그가 도배되는 것을 방지
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue
                
                # 픽셀 좌표 추출
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 0~1 사이로 정규화 (Normalization)
                norm_x1 = max(0.0, min(1.0, x1 / width))
                norm_y1 = max(0.0, min(1.0, y1 / height))
                norm_x2 = max(0.0, min(1.0, x2 / width))
                norm_y2 = max(0.0, min(1.0, y2 / height))
                
                label_id = int(box.cls[0])
                label_name = self.model.names[label_id]
                
                # 합의된 포맷인 Detection 객체로 감싸서 반환
                det = Detection(
                    label=label_name,
                    box=(norm_x1, norm_y1, norm_x2, norm_y2),
                    score=conf
                )
                detections.append(det)
                
        return detections

# 단독 실행 및 검증 블록
if __name__ == "__main__":
    # 테스트용 사전학습 모델
    detector = YoloDetector(model_path="best.pt", conf_threshold=0.5)
    
    cap = cv2.VideoCapture(0)
    print("[KHS_Detector] 단독 테스트 모드. 종료하려면 'q' 키를 누르세요.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 프레임을 RGB로 변환하여 모델에 전달
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_objects = detector.process(rgb_frame)
        
        h, w = frame.shape[:2]
        
        # 결과 출력 및 시각화
        for det in detected_objects:
            # 정규화된 좌표를 다시 화면 크기에 맞게 픽셀로 복원하여 그리기
            x1, y1, x2, y2 = det.box
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(frame, f"{det.label} {det.score:.2f}", (px1, max(py1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
        cv2.imshow("KHS YOLO Object Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()