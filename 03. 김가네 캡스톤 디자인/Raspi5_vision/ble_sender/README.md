# Raspberry Pi BLE v2 송신부

ESP의 `ble_final_receiver`용 송신부다. `KHS_KMJ_vision/detector.py`를 실행하면 내부 BLE 작업도 자동 시작한다. 기존 `haptic_mvp` BLE v1과 함께 사용하지 않는다.

현재 기본값은 **mode 3 방향 안내**다. 물체가 가려져도 잠금 위치로 안내하며, 손바닥/목표 중심이 50px 이내에서 5프레임 유지되면 획득으로 판정한다. 다섯 채널 duty 255로 350ms씩 두 번 울린다. mode 2 다중 주기도 유지한다. **ESP도 최신 펌웨어를 다시 업로드해야 한다.** 구형 수신기는 초기 STOP 이후 기능 확인에서 차단한다.

- [현재 규격·획득 조건·업로드 및 실기기 시험](../../docs/2026-10-29_DH_도달알림_다섯손가락_BLE.md)

- [설치·실행·해수 코드 연결 방법](../../docs/2026-10-28_DH_라즈베리_BLE_송신부.md)
- [ESP32 16바이트 규격](../../docs/2026-10-27_DH_ESP32_BLE_송수신규격.md)

`Raspi5_vision`에서 기존 가상환경을 활성화한 뒤 실행한다.

```bash
source .venv/bin/activate
python -m pip install -r ble_sender/requirements.txt
python -u KHS_KMJ_vision/detector.py --mode 3
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

ESP 연결 후 시동어·목표 선택을 기다리며, 락온한 목표와 이번 프레임의 유효한 손이 있을 때만 거리 안내한다. 도달 시 거리 안내를 완료 알림으로 교체하고 한 회 실행 후 정지한다. 취소·손 관측 상실·종료는 정지한다. 물체 미검출만으로 잠금 위치 안내를 끄지 않는다. `quit`는 BLE와 detector를 함께 종료한다.

안내 중 통신이 늦거나 끊기면 진동 연결을 해제하고 영상·음성은 유지한다. 터미널의 `reconnect`로 다시 연결한 뒤 `resume`을 입력해야 새 프레임으로 진동을 재개한다. 자동 재연결·재개는 하지 않는다.

`--dry-run`은 BLE 없이 영상과 패킷 매핑을 검사하고, `--no-ble`은 영상·음성만 실행한다. ESP의 `DRY_RUN` 펌웨어 설정과는 다르다. 독립 모터 시험은 `python -m ble_sender --allow-test`로 가능하며, detector와 동시에 켜지 않는다.

연결·GATT·초기 STOP만 먼저 확인하려면 `python -m ble_sender.check_link --address ESP주소`를 실행한다. 카메라와 음성은 열지 않고 STOP과 상태 읽기만 수행한 뒤 종료한다. 초기 I/O는 5초, 안내 중 쓰기 응답은 1초로 제한한다. 일반 안내 TTL은 500ms, 획득 이벤트는 1000ms이며 ACK 대기가 모터 출력 시간을 연장하지 않는다.

```bash
python -m unittest discover -s ble_sender/tests -v
```

`HAPTIC_V2_LIB`를 지정하면 실제 ESP C++ 패턴 엔진과의 패킷 호환성 검사도 실행한다. 지정하지 않으면 네이티브 엔진 검사를 건너뛴다. Pi와 ESP의 무선 연결·모터는 실제 기기로 별도 확인해야 한다.
