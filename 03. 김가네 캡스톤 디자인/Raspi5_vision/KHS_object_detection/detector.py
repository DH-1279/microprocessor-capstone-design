
import os
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst
import hailo

from hailo_apps.python.pipeline_apps.detection.detection_pipeline import (
    GStreamerDetectionApp,
    get_pipeline_parser,
)


# ---------------------------------------------------------
# 탐지 결과를 저장하는 클래스
# ---------------------------------------------------------
class Detection:
    def __init__(self, label: str, box: tuple, score: float):
        self.label = label
        self.box = box  # (x1, y1, x2, y2), 0~1 정규화 좌표
        self.score = score

    def __repr__(self):
        return (
            f"Detection(label='{self.label}', "
            f"box={self.box}, score={self.score:.2f})"
        )


# ---------------------------------------------------------
# Hailo NPU 객체 탐지 콜백
# ---------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()

    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Hailo 메타데이터에서 탐지 결과 가져오기
    roi = hailo.get_main_roi(buffer)
    hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    detected_objects = []

    for det in hailo_detections:
        confidence = det.get_confidence()

        # 신뢰도 50% 미만 제외
        if confidence < 0.5:
            continue

        label = det.get_label()
        bbox = det.get_bbox()

        det_obj = Detection(
            label=label,
            box=(
                bbox.xmin(),
                bbox.ymin(),
                bbox.xmax(),
                bbox.ymax(),
            ),
            score=confidence,
        )

        detected_objects.append(det_obj)

    # 탐지 결과 출력
    if detected_objects:
        print(f"[KHS_Detector] 탐지 결과: {detected_objects}")

    # TODO: 추후 haptic_mvp 모듈과 연결하여 진동 모터 제어

    return Gst.PadProbeReturn.OK


# ---------------------------------------------------------
# 사용자 데이터
# ---------------------------------------------------------
class UserData:
    def __init__(self):
        self.use_frame = False


# ---------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 사용자 HEF 및 라벨 파일의 절대 경로
    default_hef_path = os.path.join(
        current_dir,
        "best_hailo_model",
        "best.hef",
    )

    default_labels_path = os.path.join(
        current_dir,
        "labels.json",
    )

    print("[KHS_Detector] 파일 경로 확인 중...")

    # HEF 파일 확인
    if not os.path.isfile(default_hef_path):
        raise FileNotFoundError(
            f"HEF 파일을 찾을 수 없습니다:\n{default_hef_path}"
        )

    # 라벨 파일 확인
    if not os.path.isfile(default_labels_path):
        raise FileNotFoundError(
            f"라벨 파일을 찾을 수 없습니다:\n{default_labels_path}"
        )

    print(f"[KHS_Detector] HEF: {default_hef_path}")
    print(f"[KHS_Detector] Labels: {default_labels_path}")

    # Hailo 앱에서 제공하는 기본 파서 사용
    parser = get_pipeline_parser()

    # 기본 모델 대신 사용자 모델 경로 지정
    parser.set_defaults(
        hef_path=default_hef_path,
        labels_json=default_labels_path,
    )

    # 명령행 인자 파싱
    args = parser.parse_args()

    print("[KHS_Detector] NPU 추론 엔진 가동 시작...")
    print("[KHS_Detector] 종료하려면 Ctrl+C를 누르세요.")

    user_data = UserData()

    # 확인된 생성자 인터페이스:
    # GStreamerDetectionApp(app_callback, user_data, parser=None)
    app = GStreamerDetectionApp(
        app_callback,
        user_data,
        parser=parser,
    )

    app.run()
