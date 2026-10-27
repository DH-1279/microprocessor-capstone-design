# ESP32-C3 BLE 수신기 — 2안 / 3안

작성일: 2026-10-27. 라즈베리 통합 브랜치에 추가할 ESP 전용 펌웨어다. **이 변경은 라즈베리 영상·음성·목표 추적 코드를 수정하지 않는다.** 송신부는 추후 이 수신 규격에 맞춰 연결한다.

한 개의 스케치에서 BLE 명령의 `mode=2` 또는 `mode=3`으로 선택한다. 모드를 바꿀 때 ESP를 다시 업로드하지 않는다. USB 시리얼은 상태 출력용이며 `mode 2`를 입력하는 명령 인터페이스는 없다.

## 업로드할 파일

- [ble_final_receiver.ino](firmware/ble_final_receiver/ble_final_receiver.ino): BLE 연결, GATT 수신, 출력 작업, PWM.
- [pattern_protocol.h](firmware/ble_final_receiver/pattern_protocol.h): 패킷 검증, 두 안의 패턴, TTL·정지 상태기계.
- [ESP32 BLE 송수신 규격](../../docs/2026-10-27_DH_ESP32_BLE_송수신규격.md): 나중에 라즈베리 송신부를 작성할 기준. UUID·16바이트 배치·시퀀스·STOP·Hex 예시 포함.
- [pattern_test.cpp](tests/pattern_test.cpp): 하드웨어 없는 C++ 검사.

기존 `firmware/pcb_bringup`과 `firmware/haptic_receiver`는 별도 스케치다. 이 폴더에는 기존 v1과 구별되는 BLE v2 수신기가 들어 있다.

## 두 안의 차이

| 항목 | 2안: 고정 세기 + 간격 | 3안: 방향 + 근접도 |
|---|---|---|
| 채널 | 송신부가 한 손가락으로 고정 | 송신부가 물체 방향에 따라 선택 |
| far | 200ms ON / 1000ms OFF | 방향 모터 200ms ON / 1000ms OFF |
| middle | 200ms ON / 500ms OFF | 방향 모터 200ms ON / 500ms OFF |
| near | 200ms ON / 200ms OFF | 중지 120ms ON → 120ms OFF → 120ms ON 한 회 |
| 세기 | 채널별 보정 duty 고정 | 채널별 보정 duty 고정 |

3안의 방향 약속은 엄지 쪽→TH, 소지 쪽→PI, 손끝 쪽→IN, 손목 쪽→RI, 근접 알림→MI다. 손바닥/손등·회전에 따른 최종 채널 선택은 추후 라즈베리가 계산한다. ESP는 px 좌표나 손 자세를 판단하지 않는다.

같은 근접 상태를 주기적으로 갱신해도 3안의 중지 두 번 진동은 반복하지 않는다. 구간을 벗어났다가 재진입하거나 STOP 뒤 재개하면 새 이벤트가 된다. 이는 영상상의 접근 알림이며 실제 접촉·집기 완료를 뜻하지 않는다.

모든 시점에 최대 한 모터만 ON이다. 다섯 모터 동시 구동 시험은 기존 `pcb_bringup`에서 수행한다.

## 보드 설정

| 항목 | 값 |
|---|---|
| 보드 | ESP32C3 Dev Module |
| Arduino-ESP32 | 3.3.11 |
| USB CDC On Boot | Enabled |
| 라이브러리 | NimBLE-Arduino 2.5.1 |
| 시리얼 속도 | 115200 |
| TH / IN / MI / RI / PI GPIO | 0 / 1 / 3 / 4 / 5 |
| PWM | 200Hz, 8비트 |
| 내장 LED | GPIO8, LOW 활성 |

Arduino IDE에서 `firmware/ble_final_receiver/ble_final_receiver.ino`를 연다. 같은 폴더의 헤더도 함께 있어야 한다.

기본값은 `DRY_RUN=true`이며 GPIO는 LOW로 유지하고 LED·논리 상태만 확인한다. 점검한 PCB에서 실제 모터를 시험할 때 스케치 상단을 `DRY_RUN=false`로 바꿔 업로드한다. 모터용 적정 전원·공통 GND·MOSFET·게이트 풀다운·수정한 다이오드 방향(K→모터+, A→드레인)을 전제로 한다. GPIO6/7 및 IMU는 사용하지 않는다.

시작 메시지:

```text
BLE V2 | READY | DRY_RUN | modes=2,3 | TH=0 IN=1 MI=3 RI=4 PI=5
```

## 수신과 정지

- BLE 이름: `KIMGANE-HAPTIC-V2`. 명령은 16바이트, GATT Write with response다.
- 송신부가 약 100ms마다 새 번호로 갱신하고 TTL=500ms로 시작하는 구성을 권장한다.
- 같은 mode/zone/channel의 갱신은 TTL만 연장하며 진동 위상을 초기화하지 않는다.
- STOP, 무효 패킷, TTL 만료, 연결 종료는 출력 OFF다. 출력 하드웨어 오류는 재부팅까지 유지된다.
- 출력 작업은 약 5ms 주기이며 Serial/BLE I/O와 분리되어 있다. 실제 반영에는 작업 주기와 스케줄링 지연이 있다.
- far 반복 주기는 최대 TTL보다 길므로 단발 명령만으로 지속 반복하지 않는다. **새 sequence를 계속 보내야 한다.**

상태 패킷과 LED는 수신기가 결정한 논리 출력이다. 모터 진동·전류 센서 측정값은 아니다. 송신부가 아직 연결되지 않은 상태에서 업로드만 하면 진동 없이 수신 대기한다.

## 검증

라즈베리 또는 C++17 컴파일러가 있는 PC에서:

```bash
g++ -std=c++17 -Wall -Wextra -Werror tests/pattern_test.cpp -o /tmp/kimgane-v2-test
/tmp/kimgane-v2-test
```

2026-10-27 PC 확인:

- MSVC C++17 `/W4 /WX /analyze`: **12,523개 검사 통과**.
- 명령 길이·필드, TTL 경계, 시퀀스 순환·재전송, STOP, 연결 종료, 하드웨어 오류 유지, 갱신 시 위상 유지, 3안 근접 알림 비반복을 검사했다.
- Arduino-ESP32 3.3.11 / NimBLE-Arduino 2.5.1 / C3 USB CDC: DRY_RUN=true 및 false 모두 컴파일 통과, `--warnings all` 진단 없음.
- DRY_RUN=true: 프로그램 585,339바이트(44%), 정적 RAM 23,728바이트(7%).
- DRY_RUN=false: 프로그램 594,439바이트(45%), 정적 RAM 23,856바이트(7%).
- 규격 문서의 Hex 예시 4개도 실제 C++ 상태기계가 수락하고 STOP이 전체 출력을 0으로 만드는 것을 확인했다.

보드 업로드·실제 BLE·모터·Pi 연동 시험은 아직 수행하지 않았다. 라즈베리의 현재 이슈 수정 후 이 규격에 맞춰 송신부를 작성하고 두 안의 실기 감각을 비교한다.
