# 2026-09-19 Raspberry Pi MVP 구현·검증

## 이번 작업의 결과

[9월 18일 설계](2026-09-18_Raspberry_Pi_온디바이스_MVP_설계.md)를 바탕으로 카메라 한 대, 물체·손 인식, 한국어 음성, 목표 선택, BLE 5채널 출력을 연결하는 코드를 구현했다. 소프트웨어 정적 검사와 가상 입력/오류 상황 시험을 수행하고 발견한 문제를 수정했다.

실행 진입점은 `Raspi5_vision/haptic_mvp`이며, [설치·실행 안내](../Raspi5_vision/README.md)에 Pi 준비부터 ESP32 LED 시험까지 정리했다. 기존 `board_bringup`, `ble_led_control` 코드와 9월 17일 실기기 BLE 시험 기록은 유지했다.

오늘은 Windows 개발 PC에서 구현·정적 검사·시뮬레이션을 마친 뒤, 브랜치를 GitHub에 올리고 라즈베리파이에서도 가상 입력 시뮬레이션과 기본 라이브러리 설치를 진행했다.

**Pi에서 시뮬레이션 6개 통과와 패키지 설치까지 확인했다. 카메라·AI HAT+ 2·마이크·스피커의 실제 통합 실행이나 새 ESP32 수신 펌웨어의 보드 업로드는 아직 하지 않았다. 라이브러리 import 검사 결과도 아직 전달받지 않았다.**

## 오늘 확정한 진행 방향

- 물체 탐지는 예정대로 **AI HAT+ 2의 Hailo-10H**에서 실행한다. 현재 HAT는 배송 대기 중이다.
- 배송을 기다리는 동안에는 가상 입력으로 제어 흐름을 확인하고 Pi 실행 환경을 준비한다. CPU용 YOLO를 설치해서 대신 돌리는 작업은 이번 준비 범위에 포함하지 않는다.
- 카메라 한 대에서 물체와 손을 인식하고, 한국어 음성으로 목표를 선택한 뒤 장갑에 5채널 BLE 명령을 보내는 MVP 구성을 유지한다.
- 초기 객체는 컵·물병·휴대전화·책·마우스이며 기존 사전학습 모델을 사용한다. 개인 책상 데이터셋 학습은 우선 진행하지 않는다.

## 구현한 기능

| 구성 | 현재 구현 |
|---|---|
| 영상 입력 | CSI Picamera2, USB/영상 파일 OpenCV, 최신 프레임만 유지 |
| 물체 탐지 | Hailo-10H용 YOLOv8n HEF 기본 경로, 명시적 Ultralytics CPU 대체 경로 |
| 손 추적 | MediaPipe Hand Landmarker, 지정한 한 손만 선택 |
| 좌표 처리 | 같은 RGB 프레임에서 추론, YOLO 패딩 제거, 화면 비율과 회전 반영 |
| 목표 선택 | 컵·물병·휴대폰·책·마우스, 여러 후보의 좌우 선택, 보수적인 IoU 추적 |
| 한국어 음성 | 로컬 whisper.cpp STT, eSpeak NG TTS, 입력 에너지 기반 발화 구간 추출 |
| 안내 | 탐색 중 약 5초 간격 물체 목록, 목표 선택 확인, 손·목표 손실과 근접 안내 |
| 병행 실행 | 영상/마이크/STT/TTS 작업 분리, 메인 상태 처리와 비동기 BLE 송신 |
| 통신 | 전용 UUID, 11바이트 명령, 번호·TTL·5채널 세기, STOP 확인 후 연결 |
| 수신기 | ESP32-C3, 기본 DRY_RUN, 내장 LED·시리얼 확인, 전용 만료 검사 작업 |
| 준비 점검 | `doctor`, SDK import/API를 별도 프로세스로 검사하는 `doctor --deep` |

설치와 모델 다운로드가 끝난 뒤에는 클라우드 STT/TTS나 원격 추론을 호출하지 않는다. YOLO만 HAT를 사용하고 MediaPipe·STT·TTS는 Pi CPU에서 동작한다. 기본 STT 모델은 다국어 `tiny`이며 실제 한국어 정확도와 지연을 비교해 `base` 등으로 조정할 수 있다.

계산하는 것은 **영상 평면의 방향과 상대적인 근접도**다. 단안 카메라에서 실제 cm 거리·깊이·파지 성공을 측정한다고 표시하지 않는다. MediaPipe 좌우 분류 점수 역시 관절 위치의 정확도를 측정한 값으로 취급하지 않는다.

## 오류 상황의 처리

- 손을 놓치면 5채널 출력을 0으로 한다. 같은 목표가 유지된 상태에서 손이 다시 인식되면 안내할 수 있다.
- 목표를 놓치거나 동일 물체 후보의 연결이 모호해지면 이전 목표로 자동 복귀하지 않고 재선택을 요구한다.
- 기본 0.5초 이상 오래된 영상은 안내에 사용하지 않는다. 반복 프레임만으로 안정 인식 횟수를 채우지 않는다.
- BLE 연결 상태가 바뀌면 기존 목표·대기 명령·이전 STT 결과를 무효화한다. 재연결 후 새 목표를 선택해야 한다.
- 잘못된 패킷, 중복/과거 명령, 연결 종료 또는 명령 TTL 만료 시 수신 출력이 꺼진다. 유효한 STOP은 과거 번호여도 출력을 끄되 번호를 되돌리지 않는다.
- BLE 전송은 동시에 하나만 진행한다. 전송에 실패하면 연결을 정리하고 다시 연결한다. 메인 판단 자체가 오래된 경우에도 STOP을 보낸다.
- 음성/영상 작업 오류 시 정리 절차를 수행한다. 한 구성 요소의 종료가 실패해도 다른 구성 요소를 정리한다.

기본 송신 간격 100ms, TTL 500ms는 시작 설정이다. 실제 BLE 지연이나 전체 지연의 보장값이 아니다. 수신 출력 작업은 약 5ms마다 검사하지만 물리 출력에는 스케줄링 지연도 존재한다.

## 개발 PC 검증 결과

아래 검사는 2026-09-19에 Windows x64, Python 3.12.14 개발 환경에서 수행했다. 라즈베리파이에서 확인한 내용은 다음 별도 절에 기록한다.

| 검사 | 결과 | 범위 |
|---|---|---|
| pytest | **932 passed, 24 subtests passed** | 컨트롤러, 음성, 영상 어댑터, BLE 대역, 런타임 정리, doctor, 입력 경계 |
| Ruff 0.16.8 | 통과 | Python 구문/이름/기본 오류 검사, 수신기 교차 검사 스크립트 포함 |
| Mypy 2.3.1 | 12개 소스 파일 통과 | 하드웨어 SDK의 타입 정보가 없는 부분은 동적 타입으로 처리 |
| compileall | 통과 | Python 모듈 바이트코드 컴파일 |
| 가상 통합 시나리오 | **6개 모두 PASS** | 좌표·명령 → 컨트롤러 → BLE 인코딩 → 수신기 모델 |
| C++ 호스트 시험 | **6,810개 조건 통과** | 실제 수신기 상태기계의 패킷·번호·TTL·연결 처리 |
| C++ 정적 분석 | 진단 없음 | MSVC C++17 `/W4 /WX /analyze` |
| Python/C++ 교차 비교 | **20,000개 동작 통과** | 같은 입력의 수락 여부, 상태, 번호, 남은 TTL, 5채널 출력 비교 |
| Arduino 최종 컴파일 | 성공, `--warnings all` 진단 없음 | ESP32 코어 3.3.11, NimBLE-Arduino 2.5.1 |

Python 테스트 수에는 잘못된 패킷 길이·플래그 등 여러 입력값을 나눠 검사한 사례가 포함된다. 932개의 서로 다른 사용자 시나리오 또는 932회의 실기기 시험을 뜻하지 않는다. 영상 SDK 및 BLE API 시험은 대역을 사용했으며, 음성 시험은 일부 실제 자식 프로세스의 시간 초과·취소까지 검사했다. 모델의 인식 정확도를 측정한 시험은 아니다.

최종 수신기 컴파일 사용량:

- 프로그램: **584,737 / 1,310,720바이트, 44%**.
- 정적 RAM: **23,720 / 327,680바이트, 7%**.
- 동적 힙·BLE 실행 중 메모리·작업 스택 최대값은 실제 보드에서 별도로 측정해야 한다.

가상 통합 시나리오:

1. 목표 선택, 근접, 손 손실과 복귀, 명시적 정지.
2. 목표가 사라졌다 나타나도 재선택 전까지 정지 유지.
3. 영상 갱신 중단 시 정지.
4. 컵 여러 개의 좌우 선택과 탐지 순서 변경.
5. 패킷 손실, 수신 만료, 연결 종료와 재연결 초기 상태.
6. 잘못된 패킷·지원하지 않는 물체·복수 목표·잘못된 좌표 거부.

## 검사 중 수정한 문제

| 발견한 문제 | 수정 |
|---|---|
| Python에서 `13.220 + 0.300`의 부동소수점 오차로 정확한 만료 시점에 출력 유지 | 시뮬레이터의 만료 계산을 정수 밀리초로 변경, C++과 20,000회 재비교 |
| 재연결 전 녹음한 STT 결과가 늦게 도착하면 이전 목표 재선택 가능 | 음성 세션 무효화 후 명령 보관함 비우기 |
| 정지 명령을 꺼내는 순간 완료된 STT 결과가 뒤이어 처리될 가능성 | 정지 시 음성 무효화와 보관함 비우기를 함께 수행 |
| 손 가림·근접도 변화로 목표 선택 확인 음성이 사라질 가능성 | 유효한 목표 확인 안내를 유지하고 현재 상태 안내와 구분 |
| 이전 BLE 클라이언트의 늦은 연결 해제 콜백 | 현재 클라이언트 여부를 검사 |
| BLE 연결 성공을 충분히 확인하지 못하는 초기 응답 검사 | STOP 응답의 번호·상태·남은 TTL·출력까지 검사 |
| 음성 종료 오류가 카메라 정리를 건너뛰게 할 가능성 | 정리 단계의 오류를 보관하고 나머지 정리 계속 수행 |
| 기본 Whisper 경로의 `~`를 doctor가 확장하지 않음 | 런타임과 동일하게 사용자 홈 경로를 확장하여 검사 |

## 라즈베리파이에서 직접 확인한 결과

이 절은 사용자가 전달한 실제 Pi 터미널 출력에 근거한다. 개발 PC의 전체 테스트를 Pi에서도 수행했다고 간주하지 않는다.

### Git으로 코드 받기

초기 MVP 구현·테스트·문서 35개 파일을 `DH-1279/2026-09-19` 브랜치의 [`e50c7a3`](https://github.com/DH-1279/microprocessor-capstone-design/commit/e50c7a3) 커밋으로 push했다. 커밋 메시지는 `feat: implement on-device vision voice and haptic MVP`이다.

Pi 프로젝트 경로는 다음과 같다.

```text
~/Desktop/kimGA_raspi/microprocessor-capstone-design/
└── 03. 김가네 캡스톤 디자인/
    └── Raspi5_vision/
```

### 부품 없이 시뮬레이션 실행

Pi에서 다음 명령을 실행했다. 이 모드는 Python 기본 라이브러리만 사용하므로 가상환경이나 YOLO 설치 없이 실행할 수 있다.

```bash
cd "$HOME/Desktop/kimGA_raspi/microprocessor-capstone-design/03. 김가네 캡스톤 디자인/Raspi5_vision"
python3 -m haptic_mvp simulate --report reports/simulation.json
```

사용자가 전달한 실행 결과:

```text
PASS guidance_hand_loss_stop (8 steps)
PASS target_loss_requires_reselection (10 steps)
PASS stale_frames_stop (9 steps)
PASS multiple_cups_keep_identity (7 steps)
PASS packet_loss_watchdog_reconnect (5 steps)
PASS invalid_packet_and_request (7 steps)
Report: reports/simulation.json
```

**6개 시나리오, 총 46단계가 통과**했고 결과 파일 생성 메시지를 확인했다. 보고서 파일 내용 자체는 아직 전달받지 않았다. 이 시험은 물체·손 좌표와 수신기 모두 가상이므로 실제 카메라, STT/TTS, ESP32 LED 또는 모터가 동작했다는 의미는 아니다.

### 실행 환경과 기본 패키지 설치

설치 로그에서 확인한 환경은 **Raspberry Pi OS Trixie 계열, ARM64, Python 3.13 가상환경**이다. Trixie apt 저장소, arm64 패키지, `.venv/lib/python3.13/site-packages` 경로로 확인했다. Python 런타임의 정확한 패치 버전은 별도 출력으로 확인하지 않았다.

실행한 설치 명령:

```bash
sudo apt update
sudo apt install -y python3-venv python3-picamera2 python3-opencv python3-numpy libportaudio2 alsa-utils espeak-ng

cd "$HOME/Desktop/kimGA_raspi/microprocessor-capstone-design/03. 김가네 캡스톤 디자인/Raspi5_vision"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-pi.txt
python -m pip show bleak mediapipe sounddevice
```

`--system-site-packages`는 apt로 설치한 카메라·시스템 Python 모듈을 가상환경에서도 사용하기 위한 설정이다. 이전 `ble_led_control/.venv`와 이번 `Raspi5_vision/.venv`는 별도 환경이다.

| 구성 요소 | 로그에 표시된 버전 | 확인 범위 |
|---|---|---|
| Bleak | 3.0.2 | `.venv` 설치 및 `pip show` 확인 |
| MediaPipe | 1.0.1 | ARM64 wheel 설치 및 `pip show` 확인 |
| sounddevice | 0.5.3 | `.venv` 설치 및 `pip show` 확인 |
| Picamera2 | 0.3.37-1 | apt에서 기존 설치 확인 |
| NumPy | 2.2.4 | 시스템 패키지를 pip 의존성 확인에 사용 |
| 시스템 OpenCV | 4.10.0+dfsg-5 | `python3-opencv` 설치 완료 |
| 가상환경 OpenCV contrib | 5.0.0.93 | MediaPipe 의존성으로 설치 완료 |
| eSpeak NG | 1.52.0+dfsg-5 | apt 설치 완료 |
| PortAudio | 19.6.0-1.2+b3 | `libportaudio2` 설치 완료 |

OpenCV가 시스템과 가상환경에 각각 설치되었다. 실제로 로딩되는 모듈과 NumPy/MediaPipe/Picamera2 조합의 호환성은 아래 import 검사에서 확인해야 한다. 패키지 설치 성공만으로 실제 SDK 로딩이나 장치 동작까지 검증한 것으로 기록하지 않는다.

### 설치 중 나온 의존성 메시지

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
types-seaborn 0.13.2 requires pandas-stubs, which is not installed.
```

기존에 보이는 `types-seaborn`의 의존성 `pandas-stubs`가 누락되었다는 메시지다. `types-seaborn`은 [seaborn 타입 검사용 패키지](https://pypi.org/project/types-seaborn/)이며, 현재 MVP 코드에서는 seaborn/pandas를 사용하지 않는다. 로그의 `Successfully installed`와 이후 `pip show`로 핵심 3개 패키지의 설치는 확인했다.

이 누락을 해결했다는 결과는 아직 없다. 전체 Python 환경에 의존성 문제가 전혀 없다고 판단하지 않으며, 이 메시지와 실제 MVP 라이브러리 로딩 결과를 구분한다.

### 다음 확인 명령 — 아직 결과 미확인

사용자에게 아래 import 검사를 안내했지만 실행 결과는 아직 전달받지 않았다. 카메라나 HAT를 연결하지 않고 수행하는 라이브러리 로딩 검사다.

```bash
source .venv/bin/activate
python -c "import bleak, mediapipe, sounddevice, cv2, numpy; from picamera2 import Picamera2; print('라이브러리 로딩 OK')"
```

`라이브러리 로딩 OK`가 출력되면 기본 SDK import까지 확인한 것으로 기록한다. 이 명령은 Hailo 모델 추론이나 카메라 촬영·음성 녹음·BLE 통신을 시험하지 않는다.

## 개발 검사 재현 방법

Pi 프로젝트의 `Raspi5_vision` 폴더에서:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check haptic_mvp tests ../esp32_device/firmware/haptic_receiver/tests/parity_test.py
python -m mypy haptic_mvp
python -m compileall -q haptic_mvp
python -m haptic_mvp simulate --report reports/simulation.json
```

순수 시나리오는 별도 패키지 없이도 `python3 -m haptic_mvp simulate`로 실행된다. 전체 Python 시험에는 pytest와 NumPy가 필요하다. C++ 호스트 시험과 Python/C++ 비교 실행법은 [수신기 안내](../esp32_device/firmware/haptic_receiver/README.md)에 있다. 생성되는 모델·녹음·캐시·빌드 파일·JSON 보고서는 Git에서 제외한다.

## 부품 수령 후 확인할 것

오늘 확인한 Pi 시뮬레이션과 기본 패키지 설치 이후의 진행 상태는 다음과 같다.

| 항목 | 현재 상태 |
|---|---|
| MVP 초기 구현 브랜치 push | 완료 — `e50c7a3` |
| Pi 가상 입력 시뮬레이션 | 6개 시나리오 통과 |
| Pi 기본 라이브러리 설치 | 설치 로그와 핵심 3개 `pip show` 확인 |
| 기본 SDK import 검사 | 명령 안내 완료, 실행 결과 미확인 |
| Hailo 드라이버·HEF 모델·손 추적 모델 준비 | 준비 완료 여부 미확인 |
| whisper.cpp·다국어 STT 모델 준비 | 준비 완료 여부 미확인 |
| 카메라·HAT·마이크·스피커 통합 실행 | 미검증 |
| 새 haptic_receiver 업로드·실제 BLE 시험 | 미검증 — 기존 9월 17일 LED 시험과 구분 |

| 순서 | 확인 내용 | 완료 판단 |
|---|---|---|
| 1 | Pi OS/ARM64, Hailo-10H 드라이버, Python SDK와 모델 설치 | `doctor --deep` 및 HAT 식별·모델 생성 성공 |
| 2 | 실제 카메라와 두 모델 | 컵 상자와 선택한 손이 같은 화면에서 올바른 위치로 표시됨 |
| 3 | 영상 방향·회전·손 좌우 | 실제 움직임과 출력 방향이 일치, 장갑 착용/가림도 확인 |
| 4 | 마이크·STT·스피커·TTS | 한국어 대상 명령이 인식되고 스피커 음성이 자기 명령으로 들어가지 않음 |
| 5 | 새 ESP32 수신기 DRY_RUN | 목표 안내 시 LED 켜짐, 손/목표 손실·정지·연결 종료 시 꺼짐 |
| 6 | 장시간 동시 실행 | 프레임 나이, STT 처리 시간, BLE 쓰기 지연, CPU/온도/메모리를 실제 측정 |
| 7 | 회로 확인 후 모터 | 적정 전원·드라이버·다이오드·풀다운·핀 배정 확인 뒤 채널별 구동 |

Hailo-10H HEF와 설치된 HailoRT 버전 호환성, MediaPipe/OpenCV/NumPy의 Pi ABI, 실제 장갑 인식률, 한국어 TTS 발음과 STT 지연은 남은 실기기 검증 사항이다. eSpeak NG 음질과 초기 Whisper의 명령별 모델 로드는 성능 평가 후 개선할 수 있다.

작업 브랜치는 `DH-1279/2026-09-19`이다. AI HAT+ 2 사용은 확정되어 있으며 배송 전에는 가상 입력 시뮬레이션과 실행 환경 준비를 진행한다. 별도 CPU YOLO 실행은 이번 준비에 필수 사항이 아니다.
