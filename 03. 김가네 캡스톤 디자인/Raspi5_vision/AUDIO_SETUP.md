# Raspberry Pi 로컬 한국어 음성 준비

카메라·BLE와 병행하는 음성 모듈은 **USB 마이크 → 발화 구간 추출 → whisper.cpp 한국어 STT → 명령 처리**, **eSpeak NG → WAV 캐시 → 스피커** 순서로 동작한다. 음성 파일은 인터넷으로 보내지 않는다. 최초 패키지·모델 다운로드에는 인터넷이 필요하다.

이 문서의 설치 명령은 Raspberry Pi OS 64-bit에서 실행한다. 소프트웨어 로직은 하드웨어 없이 검사했으며 실제 USB 장치 호환성, 한국어 인식 정확도, 영상 처리와 동시 실행 시 지연은 부품 수령 후 측정해야 한다.

## 1. OS 패키지와 Python 라이브러리

```bash
sudo apt update
sudo apt install -y git cmake build-essential curl libportaudio2 alsa-utils espeak-ng
cd ~/Desktop/kimGA_raspi/microprocessor-capstone-design/"03. 김가네 캡스톤 디자인/Raspi5_vision"
source .venv/bin/activate
python -m pip install sounddevice==0.5.3
```

상위 실행 안내에서 프로젝트 `.venv`를 먼저 만든 상태를 전제로 한다. 이전 `ble_led_control/.venv`와 이번 `Raspi5_vision/.venv`는 경로가 다르다.

## 2. whisper.cpp 빌드와 다국어 모델 다운로드

공식 릴리스 `v1.9.4`를 사용한다. Pi CPU용으로 빌드하며, HAT 가속이 자동 적용되는 구성이 아니다. 별도 폴더가 이미 있다면 기존 작업을 덮어쓰지 말고 경로를 확인한다.

```bash
cd ~/Desktop/kimGA_raspi
git clone --depth 1 --branch v1.9.4 https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 2
sh ./models/download-ggml-model.sh tiny
./build/bin/whisper-cli --help
```

`tiny`는 다국어 모델이고, `tiny.en`은 영어 전용이다. 이 프로그램은 `.en` 모델 파일 이름을 오류로 처리한다. 한국어 정확도가 부족하면 `base` 다국어 모델을 추가 다운로드하여 정확도·지연을 비교한다.

```bash
cd ~/Desktop/kimGA_raspi/microprocessor-capstone-design/"03. 김가네 캡스톤 디자인/Raspi5_vision"
mkdir -p models
cp ~/Desktop/kimGA_raspi/whisper.cpp/models/ggml-tiny.bin models/
```

`config/mvp.json`의 `audio` 설정에서 `whisper_executable`을 `~/Desktop/kimGA_raspi/whisper.cpp/build/bin/whisper-cli`, `whisper_model`을 `models/ggml-tiny.bin`으로 맞춘다. Python 프로그램을 `Raspi5_vision` 폴더에서 실행한다. 최초 실행에서 모델을 로드하는 시간이 들며, 현재 구현은 명령마다 자식 프로세스를 실행해 모델을 로드한다. 속도 검증 후 상주형 STT로 바꿀 수 있다. `whisper_threads: 2`는 영상 처리와 CPU를 나눠 쓰기 위한 초기값이며 실측 성능 보장은 아니다.

## 3. 마이크·스피커를 각각 확인

```bash
arecord -l
aplay -l
python -m sounddevice
```

`audio.input_device`는 `python -m sounddevice`의 마이크 장치 번호 또는 이름이다. `null`이면 기본 입력 장치를 사용한다. `audio.output_device`는 ALSA 이름이며, `aplay -L`에서 확인한 `plughw:CARD=...,DEV=0` 등을 지정한다. PortAudio 입력 번호와 ALSA 출력 이름은 서로 다른 체계다. `null`이면 기본 출력 장치를 사용한다.

마이크 기본 장치를 선택한 뒤 3초 동안 “컵 찾아줘”를 말해 녹음한다.

```bash
arecord -f S16_LE -r 16000 -c 1 -d 3 /tmp/kimgane-mic.wav
aplay /tmp/kimgane-mic.wav
~/Desktop/kimGA_raspi/whisper.cpp/build/bin/whisper-cli -m models/ggml-tiny.bin -l ko -t 2 -f /tmp/kimgane-mic.wav -nt
espeak-ng -b 1 -v ko -s 155 -w /tmp/kimgane-tts.wav "컵을 찾아 안내하겠습니다"
aplay /tmp/kimgane-tts.wav
```

기본 입력이 원하는 USB 마이크가 아니면 `arecord -D 'plughw:CARD=해당카드이름,DEV=0' ...`처럼 확인한 실제 장치를 지정한다. 샘플링 주파수 오류가 나면 16 kHz 변환을 지원하는 ALSA `plughw` 또는 기본 장치를 선택하고, 프로그램의 `input_device`도 같은 마이크를 가리키는지 확인한다. 실제 USB 마이크가 도착하면 가장 먼저 해야 할 확인이다.

## 4. 주요 설정과 제약

| 설정 | 초기값 | 의미 |
|---|---:|---|
| `enabled` | 하드웨어 실행에서 `true` | 음성 장치 사용 |
| `sample_rate` | 16000 | mono PCM16 입력, 변경 불가 |
| `block_ms` | 20 | 마이크 처리 단위 |
| `rms_threshold` | 0.015 | 발화 후보 음량 기준(0~1 정규화 RMS) |
| `min_speech_ms` | 200 | 너무 짧은 클릭 소리 제외 |
| `silence_ms` | 650 | 이 길이만큼 조용하면 발화 완료 |
| `max_utterance_ms` | 5000 | 한 음성 입력의 최대 길이 |
| `pre_roll_ms` | 200 | 첫 음절 앞부분 보존 |
| `echo_cooldown_ms` | 300 | TTS 종료 후 잔향 입력 제외 |
| `stt_timeout_s` | 30 | STT 자식 프로세스 제한 시간 |
| `tts_timeout_s` | 20 | 합성·재생 각 작업의 제한 시간 |
| `cache_dir` | `~/.cache/kimgane-haptic/tts` | 고정 안내 WAV 저장 위치 |

발화 검출은 훈련된 음성 분류기 대신 PCM 에너지 기준을 사용한다. 팬 소음이나 음악도 발화로 잡힐 수 있으므로 실제 책상 환경에서 기준값을 조정한다. 한국어 STT 결과는 메인 프로그램의 허용 물체·정지 명령 사전으로 다시 검사한다.

TTS 재생 중과 종료 직후에는 마이크 입력을 버린다. 이미 사용자가 말하기 시작했거나 STT가 처리 중이면 `say()`는 `false`를 반환하여 안내가 말을 끊지 않도록 한다. 안내 대기열은 1개로 제한하여 오래된 물체 목록이 누적되지 않는다. 이 초기 버전은 안내 중 음성 끼어들기를 지원하지 않으므로, 종료가 필요할 때 터미널의 Ctrl+C를 사용할 수 있다.

마이크 캡처·STT와 TTS는 별도 스레드에서 동작한다. `get_error()`로 하드웨어·프로세스 오류를 메인에 전달한다. `close()`는 입력 스트림을 닫고 실행 중인 자식 프로세스 및 작업 스레드를 종료한다. 녹음과 STT 임시 파일은 작업 종료 후 삭제되고, TTS 캐시는 같은 문장을 재생할 때 재사용한다.

BLE 세션 변경이나 명시적 정지 시 `discard_input()`으로 대기 중인 음성과 이미 처리 중인 STT 결과를 무효화할 수 있다. 호출 뒤 메인 명령 보관함도 비워야 이전 세션의 목표 선택이 새 연결에 재생되지 않는다. `on_text` 콜백은 다른 스레드에서 호출되므로 문자열을 메인 보관함에 넣는 일만 하고, `AudioService` 메서드를 다시 호출하지 않는다.

eSpeak NG의 한국어 발음은 자연스러운 음성 모델과 차이가 있다. 대상 물체 이름이 실제 사용자에게 명확하게 들리는지 확인하고, 필요하면 한국어 TTS 어댑터나 사전 녹음 안내로 교체한다.

## 5. 검증 범위

```bash
python -m unittest discover -s tests -p test_audio.py -v
```

테스트는 마이크·스피커·Whisper 모델을 요구하지 않는다. PCM 발화/무음 경계, 짧은 잡음 제거, 최대 버퍼 길이, TTS 에코 차단, 언어 모델 설정 검증, 안전한 프로세스 인자, WAV 캐시, STT 출력 파일 처리, 오류 전달, 실제 자식 프로세스 시간 초과 및 종료를 검사한다. 음성 인식 정확도 자체는 실제 WAV/마이크를 사용해 별도로 확인한다.

## 공식 참고 자료

- [whisper.cpp v1.9.4 릴리스](https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.4)
- [whisper.cpp CLI 옵션](https://github.com/ggml-org/whisper.cpp/tree/v1.9.4/examples/cli)
- [whisper.cpp 모델 다운로드·빌드](https://github.com/ggml-org/whisper.cpp/tree/v1.9.4)
- [eSpeak NG 명령 옵션](https://github.com/espeak-ng/espeak-ng/blob/master/src/espeak-ng.1.ronn)
- [sounddevice 0.5.3 RawInputStream](https://python-sounddevice.readthedocs.io/en/0.5.3/api/raw-streams.html)
