import cv2
import numpy as np
from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)

# ==========================================
# 1. 설정 및 커스텀 라벨 정의
# ==========================================
HEF_PATH = "yolov8n_custom.hef"  # 준비된 커스텀 HEF 파일 경로로 변경하세요
CAMERA_ID = 0                    # 카메라 인덱스
SCORE_THRESHOLD = 0.5            # 탐지 임계값

# 학습하신 커스텀 데이터셋의 클래스 (0~10번)
CUSTOM_CLASSES = [
    "Earbud",
    "bottle",
    "cell phone",
    "cup",
    "remote",
    "Glasses",
    "Medicine",
    "key",
    "Pen",
    "Tumbler",
    "Wallet"
]

def preprocess(frame: np.ndarray, target_shape: tuple) -> np.ndarray:
    """입력 프레임을 모델 입력 크기(H, W)에 맞춰 리사이즈 및 BGR->RGB 변환"""
    target_h, target_w = target_shape
    resized = cv2.resize(frame, (target_w, target_h))
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb_frame, axis=0)

def postprocess_and_draw(frame: np.ndarray, detections: list, score_threshold: float = 0.5):
    """Hailo NMS 출력을 파싱하여 원본 프레임에 바운딩 박스 시각화"""
    h, w, _ = frame.shape

    # detections 구조: [class_id, [ymin, xmin, ymax, xmax, score]]
    for class_id, class_detections in enumerate(detections):
        if len(class_detections) == 0:
            continue

        for det in class_detections:
            ymin, xmin, ymax, xmax, score = det

            if score < score_threshold:
                continue

            # 정규화 좌표(0~1)를 원본 이미지 픽셀 좌표로 변환
            pt1 = (int(xmin * w), int(ymin * h))
            pt2 = (int(xmax * w), int(ymax * h))

            # 커스텀 클래스 참조
            if class_id < len(CUSTOM_CLASSES):
                class_name = CUSTOM_CLASSES[class_id]
            else:
                class_name = f"Unknown_{class_id}"
                
            label = f"{class_name}: {score:.2f}"

            # 바운딩 박스 및 텍스트 렌더링
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (pt1[0], max(pt1[1] - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

def main():
    # 2. HEF 로드
    print(f"[{HEF_PATH}] 모델 로딩 중...")
    hef = HEF(HEF_PATH)

    # 3. PCIe 인터페이스 기반 VDevice 할당 및 네트워크 설정
    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(
            hef=hef, interface=HailoStreamInterface.PCIe
        )
        network_group = target.configure(hef, configure_params)[0]
        
        # VStream 파라미터 구성 (입력: UINT8, 출력: FLOAT32 자동 변환)
        input_vstream_params = InputVStreamParams.make(
            network_group, quantized=False, format_type=FormatType.UINT8
        )
        output_vstream_params = OutputVStreamParams.make(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        # 모델의 입력 해상도 동적 추출 (일반적으로 YOLOv8은 640x640)
        input_vstream_info = hef.get_input_vstream_infos()[0]
        model_h, model_w = input_vstream_info.shape[1], input_vstream_info.shape[2]

        # 4. 카메라 스트림 오픈
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            print(f"카메라({CAMERA_ID})를 열 수 없습니다. 연결 상태를 확인하세요.")
            return

        print(f"추론 시작 (입력 해상도: {model_w}x{model_h}) - 종료하려면 'q'를 누르세요.")

        # 5. 추론 파이프라인 실행
        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("프레임을 읽어올 수 없습니다.")
                    break

                # 전처리 수행
                input_tensor = preprocess(frame, (model_h, model_w))

                # 추론 실행
                raw_output = infer_pipeline.infer(input_tensor)

                # NMS 출력 추출 (첫 번째 출력 레이어 키를 가져옴)
                output_name = next(iter(raw_output))
                detections = raw_output[output_name][0]

                # 시각화 수행
                postprocess_and_draw(frame, detections, score_threshold=SCORE_THRESHOLD)

                # 결과 화면 출력
                cv2.imshow("Raspberry Pi 5 + Hailo NPU (Custom YOLOv8n)", frame)
                
                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()