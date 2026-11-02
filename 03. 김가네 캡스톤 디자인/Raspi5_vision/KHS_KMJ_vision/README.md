# 통합 영상·음성 실행기

`Raspi5_vision` 폴더에서 가상환경을 활성화한다. 음성을 처음 준비할 때는 detector를 종료한 상태에서 실행한다.

```bash
source .venv/bin/activate
python -u -m KHS_KMJ_vision.prepare_audio
python -m KHS_KMJ_vision.prepare_audio --check
```

첫 명령은 Supertonic으로 고정 안내와 물체별 안내를 합성해 `~/.cache/kimgane-haptic/tts`에 저장한다. 마이크·카메라·BLE를 열거나 소리를 재생하지 않는다. 첫 실행 시 기존 Supertonic 모델 다운로드에 네트워크가 필요할 수 있다. 다시 실행하면 정상 파일은 재사용하고 누락·손상된 파일만 만든다. `--check`는 파일 검사만 하며 준비할 파일이 있으면 종료 코드 1을 반환한다. `--list`는 문장 목록만 표시한다.

물체 선택 안내는 **물체별 한 문장 + 공통 방향 설명**을 한 묶음으로 재생한다. 중간에 준비음이 끼어들지 않으며, 안내가 끝난 뒤 준비음이 한 번 울린다. 물체 목록처럼 조합이 달라지는 문장은 처음 요청할 때 합성하고 이후 캐시를 재사용한다. 따라서 모든 상황의 실시간 합성이 없어지는 것은 아니다.

```bash
export DISPLAY=:0
python -u KHS_KMJ_vision/detector.py \
  --address 9C:CC:01:82:75:8A \
  --mode 3 --reach-px 75 --reach-frames 8
```

`voice_profile.py`에서 실행기와 사전 생성기가 같은 물체 목록·음성·캐시 설정을 사용한다. 준비음 기본값은 880Hz, **250ms, 진폭 12,000**이다. `ready_tone_ms`, `ready_tone_amplitude`를 조절하면 준비음 파일도 구분해 생성한다. 예전 80ms 준비음 파일은 재사용하지 않는다. 준비음이 끝난 뒤 짧은 잔향 대기를 거쳐 입력을 받는다. 실제 청감은 스피커와 시스템 볼륨에서도 확인해야 한다.

터미널의 `TTS 대기`, `TTS 합성`/`TTS 캐시`, `TTS 재생 요청` 로그로 지연 구간을 구분한다. 재생 요청은 `aplay`를 실행하기 직전이며 실제 스피커에서 소리가 시작된 시각을 측정한 값은 아니다. 기존 TTS 모델 로딩과 음성 인식 시간은 별도다.

ESP 펌웨어와 BLE 패킷 규격은 이번 변경으로 바뀌지 않는다.
