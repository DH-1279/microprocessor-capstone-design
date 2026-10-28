# Raspberry Pi BLE v2 송신부

시동어·STT·TTS·카메라와 분리된 송신 프로세스다. ESP의 `ble_final_receiver`용이며 기존 `haptic_mvp` BLE v1과 함께 사용하지 않는다.

- [설치·실행·해수 코드 연결 방법](../../docs/2026-10-28_DH_라즈베리_BLE_송신부.md)
- [ESP32 16바이트 규격](../../docs/2026-10-27_DH_ESP32_BLE_송수신규격.md)

`Raspi5_vision`에서 기존 가상환경을 활성화한 뒤 실행한다.

```bash
source .venv/bin/activate
python -m pip install -r ble_sender/requirements.txt
python -m ble_sender --mode 2
```

실행 중 터미널 명령:

```text
mode 2
mode 3
status
stop
resume
quit
```

영상 프로그램이 `VisionPublisher`로 관측을 넘기기 전까지 정지 상태로 기다린다. 기존 `KHS_KMJ_vision/detector.py`에 자동 연결되어 있지는 않다. 독립 모터 시험은 `--allow-test`, BLE 없는 송신부 시험은 `--dry-run`을 사용한다. 자세한 예제는 위 문서에 있다.

```bash
python -m unittest discover -s ble_sender/tests -v
```

`HAPTIC_V2_LIB`를 지정하면 실제 ESP C++ 패턴 엔진과의 패킷 호환성 검사도 실행한다. 지정하지 않으면 해당 검사 한 개만 건너뛴다. Pi와 ESP의 무선 연결·모터는 실제 기기로 별도 확인해야 한다.
