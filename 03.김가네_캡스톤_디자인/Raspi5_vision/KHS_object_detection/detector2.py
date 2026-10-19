import os
import sys
import cv2
import numpy as np
import json

# =====================================================================
# 1. 환경 변수 강제 설정 (라즈베리파이 5 디스플레이 출력용)
# =====================================================================
os.environ["DISPLAY"] = ":0"
os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo

# =====================================================================
# 2. 파일 구조에 맞춘 자동 경로 설정
# =====================================================================
# 현재 스크립트가 위치한 KHS_object_detection 디렉토리의 절대 경로
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 모델 및 설정 파일 경로 지정
MODEL_PATH = os.path.join(CURRENT_DIR, "best_hailo_model", "best.hef")
CONFIG_PATH = os.path.join(CURRENT_DIR, "best_hailo_model", "nms_config.json")
LABELS_PATH = os.path.join(CURRENT_DIR, "labels.json")

# 라즈베리파이5 Hailo 기본 YOLO 후처리 라이브러리(C++) 절대 경로
SO_PATH = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailort.so"

# 라벨 JSON 파일 읽기 (클래스 ID에 해당하는 이름을 가져오기 위함)
class_labels = {}
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        class_labels = json.load(f)

# =====================================================================
# 3. 객체 인식 및 중심 좌표 렌더링 (콜백 함수)
# =====================================================================
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    caps = pad.get_current_caps()
    if caps:
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
    else:
        width, height = 640, 480

    success, map_info = buffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)
    if not success:
        return Gst.PadProbeReturn.OK

    try:
        frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=map_info.data)

        for detection in detections:
            # 라벨명 가져오기 (JSON에 정의된 이름이 있으면 우선 사용)
            class_id = str(detection.get_class_id())
            label = class_labels.get(class_id, detection.get_label())
            
            bbox = detection.get_bbox()
            
            xmin_px = int(bbox.xmin() * width)
            ymin_px = int(bbox.ymin() * height)
            w_px = int(bbox.width() * width)
            h_px = int(bbox.height() * height)
            
            center_x = int(xmin_px + (w_px / 2))
            center_y = int(ymin_px + (h_px / 2))
            
            # 1. 중심에 빨간색 점(원) 그리기
            cv2.circle(frame, (center_x, center_y), radius=6, color=(255, 0, 0), thickness=-1)
            
            # 2. 중심 좌표 텍스트 렌더링
            ratio_x = bbox.xmin() + (bbox.width() / 2)
            ratio_y = bbox.ymin() + (bbox.height() / 2)
            text = f"{label} ({ratio_x:.2f}, {ratio_y:.2f})"
            
            cv2.putText(
                frame, text, (center_x + 10, center_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA
            )
            
            # 콘솔에 좌표 출력 (5채널 진동 모터 연동용 데이터)
            print(f"객체: {label} | 중심 X 비율: {ratio_x:.3f} | 중심 Y 비율: {ratio_y:.3f}")

    finally:
        buffer.unmap(map_info)

    return Gst.PadProbeReturn.OK

# =====================================================================
# 4. GStreamer 파이프라인 실행 메인 함수
# =====================================================================
def main():
    Gst.init(sys.argv)

    if not os.path.exists(MODEL_PATH):
        print(f"오류: 모델 파일을 찾을 수 없습니다! 경로를 확인하세요: {MODEL_PATH}")
        return
        
    if not os.path.exists(CONFIG_PATH):
        print(f"경고: nms_config.json 파일을 찾을 수 없습니다: {CONFIG_PATH}")

    # GStreamer 파이프라인 구성
    pipeline_str = f"""
        libcamerasrc ! video/x-raw, width=640, height=480 ! 
        videoconvert ! 
        videoscale ! 
        video/x-raw, format=RGB ! 
        hailonet hef-path={MODEL_PATH} ! 
        hailofilter so-path={SO_PATH} config-path={CONFIG_PATH} qos=false ! 
        videoconvert ! video/x-raw, format=RGB ! 
        identity name=cb_probe ! 
        hailooverlay ! 
        videoconvert ! 
        autovideosink sync=false
    """
    
    print("GStreamer 파이프라인 초기화 중...")
    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except GLib.Error as e:
        print(f"파이프라인 생성 오류: {e}")
        return

    probe_element = pipeline.get_by_name("cb_probe")
    if probe_element:
        pad = probe_element.get_static_pad("src")
        pad.add_probe(Gst.PadProbeType.BUFFER, app_callback, None)

    pipeline.set_state(Gst.State.PLAYING)
    loop = GLib.MainLoop()
    
    try:
        print("\n=== 객체 인식 및 중심 좌표 추적 시작 ===")
        print("종료하려면 Ctrl+C를 입력하세요.")
        loop.run()
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()