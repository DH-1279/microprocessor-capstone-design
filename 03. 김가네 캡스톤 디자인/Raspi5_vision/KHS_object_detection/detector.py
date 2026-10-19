import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo

# 복사해 온 hailo_apps_infra 모듈에서 뼈대 클래스 임포트
from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp

# 기존에 작성하신 Detection 클래스
class Detection:
    def __init__(self, label: str, box: tuple, score: float):
        self.label = label
        self.box = box  # (x1, y1, x2, y2) 0~1 정규화 좌표
        self.score = score
        
    def __repr__(self):
        return f"Detection(label='{self.label}', box={self.box}, score={self.score:.2f})"

# ---------------------------------------------------------
# 콜백 함수: NPU가 프레임 1장의 추론을 마칠 때마다 자동 실행됨
# ---------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # GStreamer 버퍼(Hailo 메타데이터)에서 탐지된 객체 정보 꺼내기
    roi = hailo.get_main_roi(buffer)
    hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    detected_objects = []
    
    for det in hailo_detections:
        confidence = det.get_confidence()
        if confidence < 0.5: # 신뢰도 50% 미만 필터링
            continue
            
        label = det.get_label()
        bbox = det.get_bbox()
        
        # Hailo NPU 출력은 이미 0~1 사이 정규화된 값(norm)입니다.
        det_obj = Detection(
            label=label, 
            box=(bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()), 
            score=confidence
        )
        detected_objects.append(det_obj)
        
    # TODO: 여기서 추출된 detected_objects 리스트를 haptic_mvp 모듈로 전달하여 스마트 글러브 진동 모터 제어!
    if detected_objects:
        print(f"[KHS_Detector] 탐지결과: {detected_objects}")

    return Gst.PadProbeReturn.OK

# ---------------------------------------------------------
# 앱 메인 클래스: GStreamer 파이프라인을 상속받아 콜백 연결
# ---------------------------------------------------------
class KHS_DetectorApp(GStreamerDetectionApp):
    def __init__(self, args, user_data):
        super().__init__(args, user_data)
        
    def on_ready(self):
        # 파이프라인 준비(NPU 로드 완료) 후 콜백 함수 연결
        super().on_ready()
        # NMS 후처리 직후 단계("identity" 패드)에 위에서 만든 콜백 함수를 끼워 넣습니다.
        self.add_probe("identity", app_callback)

# GStreamer 앱 내부 속성 요구사항을 맞추기 위한 UserData 클래스
class UserData:
    def __init__(self):
        self.use_frame = False

# 단독 실행 및 검증 블록
if __name__ == "__main__":
    import argparse
    
    # 현재 스크립트 위치 기준 절대 경로 자동 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_hef_path = os.path.join(current_dir, "best_hailo_model", "best.hef")
    default_labels_path = os.path.join(current_dir, "labels.json")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef-path", default=default_hef_path, help="컴파일된 HEF 모델 경로")
    parser.add_argument("--input", default="rpi", help="입력 소스 (rpi 카메라 모듈)")
    parser.add_argument("--labels-json", default=default_labels_path, help="클래스 이름이 정의된 json 파일 경로")
    
    # YOLOv8n 네트워크 명시
    parser.add_argument("--network", default="yolov8n", help="사용할 네트워크 종류")
    
    args = parser.parse_args()
    
    print("[KHS_Detector] NPU 추론 엔진 가동 시작... (종료하려면 Ctrl+C)")
    
    user_data = UserData()
    app = KHS_DetectorApp(args, user_data=user_data)
    app.run()