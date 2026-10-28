# Raspberry Pi BLE v2 송신부

ESP의 `ble_final_receiver`용 송신부다. `KHS_KMJ_vision/detector.py`를 실행하면 내부 BLE 작업도 자동 시작한다. 기존 `haptic_mvp` BLE v1과 함께 사용하지 않는다.

- [설치·실행·해수 코드 연결 방법](../../docs/2026-10-28_DH_라즈베리_BLE_송신부.md)
- [ESP32 16바이트 규격](../../docs/2026-10-27_DH_ESP32_BLE_송수신규격.md)

`Raspi5_vision`에서 기존 가상환경을 활성화한 뒤 실행한다.

```bash
source .venv/bin/activate
python -m pip install -r ble_sender/requirements.txt
python -u KHS_KMJ_vision/detector.py --mode 2
```

실행 중 터미널 명령:

```text
mode 2
mode 3
status
stop
resume
reconnect
quit
```

ESP 연결 후 시동어·목표 선택을 기다리며, 락온한 목표와 이번 프레임의 유효한 손이 있을 때만 안내한다. 취소·도달·상실·종료 때 정지한다. `quit`는 BLE와 detector를 함께 종료한다.

안내 중 통신이 늦거나 끊기면 진동 연결을 해제하고 영상·음성은 유지한다. 터미널의 `reconnect`로 다시 연결한 뒤 `resume`을 입력해야 새 프레임으로 진동을 재개한다. 자동 재연결·재개는 하지 않는다.

`--dry-run`은 BLE 없이 영상과 패킷 매핑을 검사하고, `--no-ble`은 영상·음성만 실행한다. ESP의 `DRY_RUN` 펌웨어 설정과는 다르다. 독립 모터 시험은 `python -m ble_sender --allow-test`로 가능하며, detector와 동시에 켜지 않는다.

연결·GATT·초기 STOP만 먼저 확인하려면 `python -m ble_sender.check_link --address ESP주소`를 실행한다. 카메라와 음성은 열지 않고 STOP과 상태 읽기만 수행한 뒤 종료한다. 초기 I/O는 5초, 안내 중 쓰기는 0.3초로 제한한다.

```bash
python -m unittest discover -s ble_sender/tests -v
```

`HAPTIC_V2_LIB`를 지정하면 실제 ESP C++ 패턴 엔진과의 패킷 호환성 검사도 실행한다. 지정하지 않으면 해당 검사 한 개만 건너뛴다. Pi와 ESP의 무선 연결·모터는 실제 기기로 별도 확인해야 한다.
