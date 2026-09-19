# 카메라 · AI HAT+ 2 · 손 추적 준비

이 문서는 `haptic_mvp/vision.py`의 실제 영상 입력을 준비하는 절차다. 전체 실행 명령과 음성·BLE 준비는 이 폴더의 `README.md`를 따른다. 모델 파일을 한 번 내려받고 설치를 완료한 뒤에는 영상 추론에 인터넷을 사용하지 않는다.

## 지원 범위와 검증 상태

- 카메라: CSI 카메라는 Picamera2, USB 카메라와 로컬 영상 파일은 OpenCV.
- 물체 탐지: 기본값은 **Hailo-10H용 RGB YOLOv8n HEF**와 HailoRT. `detector: ultralytics`를 직접 선택하면 로컬 `.pt` 모델을 CPU에서 실행한다.
- 손 추적: MediaPipe Tasks Hand Landmarker CPU 실행. 로컬 `.task` 모델 파일을 사용한다.
- 모든 모델은 같은 RGB 프레임에서 실행한다. Hailo 입력에 추가한 여백을 제거한 뒤 원본 영상의 정규화 좌표로 복구한다.
- 호스트에서는 좌표 복원, 잘못된 NMS 출력 거부, 손 선택, 최신 프레임 교체, SDK 대역을 통한 호출을 시험했다. **실제 CSI 카메라·Hailo-10H·장갑을 낀 손에 대한 인식률과 FPS는 아직 측정하지 않았다.**

## 1. Pi OS와 HAT 확인

공식 가이드 기준은 **Raspberry Pi 5 + 64-bit Raspberry Pi OS Trixie**다. 먼저 다음 정보부터 확인한다.

```bash
cat /etc/os-release
uname -m
python3 --version
```

`aarch64`인지 확인하고, 운영체제가 다르면 먼저 [공식 AI 설치 안내](https://www.raspberrypi.com/documentation/computers/ai.html)의 해당 환경 지원을 확인한다. 다음은 AI HAT+ 2용 설치 명령이다.

```bash
sudo apt update
sudo apt install -y dkms hailo-h10-all python3-picamera2 python3-opencv python3-numpy python3-venv rpicam-apps
sudo reboot
```

재부팅 후:

```bash
hailortcli fw-control identify
hailortcli --version
rpicam-hello --list-cameras
```

장치 아키텍처가 `HAILO10H`인지 확인한다. `hailo-all`은 Hailo-8/8L용 패키지이므로 AI HAT+ 2에는 사용하지 않는다. 두 패키지는 함께 설치할 수 없다. CSI 카메라를 연결할 때는 Pi 전원을 끈 상태에서 연결한다.

## 2. 시스템 카메라·Hailo Python을 사용하는 가상환경

프로젝트의 `Raspi5_vision` 폴더에서 실행한다. 기존 BLE 시험 폴더의 `.venv`와 별도다.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-pi.txt
python -c "import picamera2, hailo_platform, mediapipe, cv2, numpy; print('vision imports OK')"
```

`--system-site-packages`는 apt로 설치한 Picamera2/libcamera/Hailo Python 모듈을 현재 환경에서도 사용할 수 있게 한다. 시스템 Python을 pip로 덮어쓰거나 `--break-system-packages`로 설치하지 않는다.

MediaPipe는 `opencv-contrib-python`과 NumPy 등을 의존성으로 설치한다. 가상환경의 OpenCV가 apt의 OpenCV보다 먼저 로드될 수 있다. 이 조합의 Pi ABI 호환성은 아직 실기기로 확인하지 않았으므로 `python -m haptic_mvp doctor --deep --console`과 위 import를 실행한다. 오류가 나면 SDK·NumPy 버전과 가상환경을 확인하며 시스템 패키지를 임의로 삭제하지 않는다.

2026-09-19에 [공식 PyPI 1.0.1 메타데이터](https://pypi.org/pypi/mediapipe/1.0.1/json)에서 `mediapipe-1.0.1-py3-none-manylinux_2_28_aarch64.whl` 제공을 확인했다. `mediapipe==1.0.1`의 `mp.tasks.vision.HandLandmarker` API도 패키지 소스로 확인했다. 예전 0.10.18의 `cp311`/`cp312` ARM wheel을 Trixie Python 3.13에 억지로 설치하지 않는다. wheel 제공 확인은 실기기 동작 검증을 대신하지 않으므로 위 import 확인과 아래 모델 생성 확인까지 수행한다.

## 3. 모델을 미리 다운로드

프로젝트는 시작할 때 모델을 자동 다운로드하지 않는다. 설정의 경로에 있는 로컬 파일을 확인한 뒤 실행하며 파일이 없으면 오류를 출력하고 종료한다.

```bash
mkdir -p models
curl --fail --location \
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task' \
  --output models/hand_landmarker.task
```

이 주소는 [Google Hand Landmarker 모델 안내](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker#models)의 공식 모델 링크다. 모델의 가중치를 저장소에 commit하지 않는다.

다음은 [Hailo Model Zoo HAILO10H 목록](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_object_detection.rst)에 명시된 RGB YOLOv8n 모델이다. 이 링크의 컴파일 버전은 **5.4.0**이다. 설치된 HailoRT와 이 HEF가 호환되는지 확인한다. 버전 불일치 오류가 발생하면 현재 HailoRT에 맞는 HAILO10H 모델을 공식 목록에서 선택하며, 다른 아키텍처 HEF로 교체하지 않는다.

```bash
curl --fail --location \
  'https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov8n.hef' \
  --output models/yolov8n_hailo10h.hef

sha256sum models/hand_landmarker.task models/yolov8n_hailo10h.hef
hailortcli parse-hef models/yolov8n_hailo10h.hef
```

`sha256sum` 출력은 재현을 위한 로컬 파일 기록용이며, 게시된 신뢰할 수 있는 체크섬과 비교하지 않은 상태에서는 진위 검증을 의미하지 않는다.

HEF는 다음 계약을 만족해야 한다.

| 항목 | 요구 조건 |
|---|---|
| 대상 장치 | Hailo-10H |
| 입력 | RGB, 640×640×3, 단일 NHWC 입력, UINT8 픽셀 |
| 출력 | 단일 `HAILO_NMS_BY_CLASS` 탐지 출력 |
| 탐지 형식 | 클래스별 N×5 배열, `ymin, xmin, ymax, xmax, score` |
| 라벨 | 모델 학습 때의 정확한 클래스 순서. 제공된 COCO 모델은 `models/coco.txt`의 80개 순서 |

`NV12`, `RGBX`, 분할 모델, `bbox_decoding_only`, NMS 없는 YOLO 원시 출력은 이 어댑터의 입력 계약에 해당하지 않는다. 해당 모델을 주면 프로그램이 변환 방식을 추측하지 않고 중지한다. HailoRT가 HEF-장치 호환성도 검사한다.

모델 생성까지 확인하려면 `Raspi5_vision` 폴더에서 실행한다.

```bash
python - <<'PY'
from haptic_mvp.config import load_config
from haptic_mvp.vision import HailoDetector, MediaPipeHandTracker

config = load_config()["vision"]
with HailoDetector(config), MediaPipeHandTracker(config):
    print("models OK")
PY
```

## 4. 실제 카메라와 손 방향 맞추기

`config/mvp.json`의 `vision` 설정에서 선택한다.

| 설정 | 값과 용도 |
|---|---|
| `source` | CSI는 `picamera2`, USB/로컬 영상은 `opencv` |
| `device` | 카메라 번호(숫자, 기본 0) 또는 기존 로컬 영상 파일 경로 |
| `width`, `height`, `fps` | 기본 640, 480, 15. USB가 실제 지원하는 해상도를 확인 |
| `hand` | `Right` 또는 `Left`. 모델이 표시하는 좌우 판정을 실제 손과 비교해 선택 |
| `mirror` | 기본 false. true이면 영상 전체를 좌우 반전한 다음 두 모델에 함께 전달 |
| `show_preview` | 기본 false. Pi 데스크톱에서 true로 설정하면 물체 상자와 손바닥 대표점 표시 |

Tailscale SSH 터미널처럼 그래픽 디스플레이가 없는 환경에서는 `show_preview: false`를 유지한다. preview의 `q`는 전체 실행 중지 요청이다.

MediaPipe의 좌우 분류와 실제 손의 관계는 카메라 배치/반전에 따라 시각적으로 확인해야 한다. 좌우 판정이 틀렸는데 단지 `Any`로 우회하지 않는다. `Any`는 후보 손이 한 개일 때만 허용하며 두 손이 잡히면 안내를 중단한다. 같은 `Right` 후보가 두 개여도 선택하지 않는다. 맨손, 장갑, 손바닥/손등, 부분 가림을 각각 확인한다.

손의 대표점은 손목과 네 손가락 뿌리 관절 5개의 평균이다. `Hand.score`는 MediaPipe가 반환한 **좌우 분류 점수**이며 관절 좌표의 정확도 점수가 아니다. 손 검출·존재·추적 임계값도 별도로 같은 설정값을 전달한다. 이 결과는 영상 평면의 방향/근접도를 위한 것이며 실제 cm 거리나 파지 완료를 측정하지 않는다.

USB나 영상 파일로 바꾸면 실제 화면 비율에 맞춰 `vision.width/height`와 `controller.image_aspect`도 맞춘다. 로컬 영상은 `vision.fps` 속도로 재생하며, 추론이 느리면 중간 프레임을 건너뛰어 최신 프레임을 처리한다. 녹화 시각은 복원하지 않고 재생 시점의 호스트 단조 시계를 사용한다.

## 5. CPU 대체 경로가 필요한 경우

HAT 없이 Pi CPU로 시험하려면 제공된 `config/cpu.json`을 선택한다. 이 경로에는 `hailo-h10-all`, HailoRT, `.hef` 모델이 필요 없다. 카메라와 MediaPipe 손 추적 모델은 필요하다. 카메라가 없으면 `python3 -m haptic_mvp simulate`로 가상 입력 시나리오부터 확인한다.

```json
{
  "vision": {
    "detector": "ultralytics",
    "object_model": "models/yolov8n.pt"
  }
}
```

아래 명령은 `Raspi5_vision` 폴더에서 실행한다. 아직 이 폴더의 `.venv`가 없다면 생성한다. 이미 가상환경을 만들었다면 생성 줄은 생략한다.

```bash
sudo apt update
sudo apt install -y python3-venv python3-picamera2 python3-opencv python3-numpy libportaudio2 curl
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-pi.txt
python -m pip install ultralytics
mkdir -p models
python -c "from ultralytics import YOLO; YOLO('models/yolov8n.pt')"
curl --fail --location \
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task' \
  --output models/hand_landmarker.task
python -m haptic_mvp doctor --deep --console --config config/cpu.json
python -m haptic_mvp run --console --config config/cpu.json
```

모델 준비용 `YOLO(...)` 명령은 파일이 없으면 공식 배포 가중치를 내려받는다. 이 설치 단계에는 인터넷이 필요하며, MVP 실행 단계에서는 기존 파일만 사용한다. [Ultralytics Pi 설치 안내](https://docs.ultralytics.com/guides/raspberry-pi/)와 [YOLOv8 사용법](https://docs.ultralytics.com/models/yolov8/)을 참고했다. 실제 Pi의 OS/Python과 PyTorch·MediaPipe 조합은 `doctor --deep`으로 확인한다. 이 조합을 현재 Windows 호스트에서 Pi 실기기로 검증한 것은 아니다.

`--console`은 마이크·STT·TTS를 끄고 키보드/문자 안내를 사용한다. CSI 카메라가 기본이며, USB 카메라나 영상 파일을 쓰려면 `cpu.json`을 별도 로컬 설정 파일로 복사하여 `vision.source: opencv`, `vision.device`와 해상도/화면 비율을 함께 설정한다. 실제 BLE 전송은 새 `haptic_receiver`를 업로드한 뒤 위 실행 명령에 `--ble`를 추가한다.

이 경로는 CPU로 실행하며 Hailo 오류가 났을 때 자동으로 바뀌지 않는다. CPU 추론이 기본 `controller.max_frame_age_s` 0.5초보다 오래 걸리면 오래된 영상으로 안내하지 않고 `stale_frame`으로 중지할 수 있다. HAT 없이 같은 FPS나 음성·영상 동시 처리 성능을 보장하지 않으므로 먼저 콘솔 모드에서 기능을 확인한다. 실제 처리 시간 측정 후 모델/추론 설정을 조정하며, 오래된 입력을 허용하기 위해 만료 시간을 무작정 늘리지 않는다.

## 구현 근거

- [Hailo 공식 InferModel 예제](https://github.com/hailo-ai/Hailo-Application-Code-Examples/blob/main/runtime/python/common/hailo_inference.py): 장치를 한 번 설정하고 입력/출력 버퍼를 사용한다.
- [HailoRT Python API 구현](https://github.com/hailo-ai/hailort/blob/master/hailort/libhailort/bindings/python/platform/hailo_platform/pyhailort/pyhailort.py): `get_buffer(tf_format=False)`의 NMS-by-class 행 순서와 동기 추론 호출 확인.
- [Picamera2 공식 픽셀 변환 구현](https://github.com/raspberrypi/picamera2/blob/main/picamera2/request.py): libcamera의 `BGR888`은 numpy 메모리에서 RGB 순서다.
- [MediaPipe Python 사용법](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python): VIDEO 모드와 증가하는 밀리초 타임스탬프 사용.

프레임 타임스탬프는 카메라 읽기 직전의 `time.monotonic()` 값이다. 센서 노출 시각을 정확히 측정했다고 간주하지 않는다. CSI의 과거 프레임 재사용을 끄고 최신 프레임 하나만 보관해 추론 대기열이 늘어나지 않게 했다. 드라이버 자체가 멈추거나 지연되는 경우 컨트롤러의 프레임 만료 처리와 ESP32 명령 만료가 마지막 출력을 정지시킨다.
