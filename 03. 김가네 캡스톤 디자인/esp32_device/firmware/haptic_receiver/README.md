# ESP32-C3 5채널 BLE 수신기

라즈베리파이 MVP에서 손가락별 진동 세기 5개를 받아 검증하는 별도 스케치다. 기존 `ble_led_control` 시험 코드는 유지한다.

## 기본 동작: 모터 없이 시험

`haptic_receiver.ino`의 `DRY_RUN = true`가 기본이다.

- 모터 GPIO를 설정하거나 PWM을 출력하지 않는다.
- TH(엄지), IN(검지), MI(중지), RI(약지), PI(소지)의 수신 세기를 시리얼에 출력한다.
- 세기 중 하나라도 0보다 크면 GPIO8 내장 LED를 켠다. 모두 0이면 끈다.
- 전원 투입, 연결 종료, 유효시간 만료, 잘못된 명령 수신 시 논리 출력과 LED를 끈다.
- 내장 LED로는 개별 손가락이나 진동 강도를 검증할 수 없다. 시리얼과 상태 패킷으로 5개 값을 확인한다.

Arduino IDE 설정은 `ESP32C3 Dev Module`, `USB CDC On Boot: Enabled`, `NimBLE-Arduino 2.5.1`, 시리얼 115200이다. `haptic_receiver.ino`와 같은 폴더에 `haptic_protocol.h`를 둔다. 업로드는 사용자가 새 스케치를 선택해서 진행한다.

시작 메시지:

```text
BLE ready | name=KIMGANE-HAPTIC | mode=DRY_RUN | channels=TH,IN,MI,RI,PI
```

## BLE 프로토콜 v1

이전 LED 스케치와 UUID가 다르다. 이 수신기에 `led_control.py`를 연결하는 방식은 지원하지 않는다.

| 항목 | 값 |
|---|---|
| 장치 이름 | `KIMGANE-HAPTIC` |
| 서비스 UUID | `328e0010-8c2a-4e58-9a48-512fc6ab1279` |
| 명령 UUID | `328e0011-8c2a-4e58-9a48-512fc6ab1279` |
| 상태 UUID | `328e0012-8c2a-4e58-9a48-512fc6ab1279` |
| 명령 전송 | GATT Write **with response** |
| 상태 확인 | GATT Read |
| 연결 | 주변장치 ESP32 한 대, 중앙장치 Pi 한 대 |

명령과 상태는 모두 정확히 11바이트다. Python 표현은 `struct.Struct("<BBHH5B")`이며 다중 바이트 정수는 little endian이다.

### 명령

| 바이트 위치 | 형식 | 의미 |
|---|---|---|
| 0 | uint8 | 버전, 반드시 `1` |
| 1 | uint8 | 플래그: bit0 `STOP`; 다른 비트는 0 |
| 2–3 | uint16 | 명령 번호, 0–65535 순환 |
| 4–5 | uint16 | 유효시간 ms, 100–1000 포함 |
| 6–10 | uint8 × 5 | TH, IN, MI, RI, PI 세기, 각각 0–255 |

`STOP` 명령은 세기 5개가 모두 0이어야 한다. 정상 일반 명령은 최초 명령을 제외하고 이전 번호와의 순환 차이가 1–32767일 때만 받는다. 같은 번호나 과거 명령은 잘못된 명령으로 처리하고 출력을 끈다. 번호가 65535에서 0으로 돌아가는 것은 정상이다.

유효한 `STOP`은 같은 번호나 과거 번호여도 출력을 끄지만, 마지막 번호를 뒤로 돌리지 않는다. 길이, 버전, 플래그, TTL 또는 번호가 잘못되면 출력을 끄고 `invalid` 상태를 기록하며 마지막 번호를 유지한다. 유효하지 않은 패킷은 유효시간을 연장하지 않는다. 이후 정상 명령을 받으면 오류/만료 플래그를 지운다. 연결 종료 시 번호 추적도 초기화한다.

예: 명령 번호 4660, TTL 300ms, 세기 `[0, 1, 127, 254, 255]`:

```text
01 00 34 12 2C 01 00 01 7F FE FF
```

### 상태

| 바이트 위치 | 형식 | 의미 |
|---|---|---|
| 0 | uint8 | 버전 `1` |
| 1 | uint8 | bit0 연결, bit1 활성, bit2 시간 만료, bit3 잘못된 명령, bit4 dry-run |
| 2–3 | uint16 | 마지막 유효 명령 번호. 아직 없으면 0 |
| 4–5 | uint16 | 남은 유효시간 ms. 비활성이면 0 |
| 6–10 | uint8 × 5 | 현재 논리 출력 세기. dry-run에서는 실제 모터 출력이 아님 |

상태 읽기는 번호와 적용 상태를 확인하기 위한 기능이다. GATT 쓰기 성공만으로 애플리케이션 수준의 명령 수락을 보장하지 않는다. 상태값은 센서로 측정한 모터 전류·회전·진동값이 아니다.

## 타임아웃과 동시 실행

TTL은 ESP32가 명령을 수신한 시점부터 센다. Pi는 활성 안내 중 새로운 번호로 주기적으로 전송해야 한다. 예를 들어 100ms 간격 송신/500ms TTL로 시작해 실제 BLE 지연에 따라 확인한다. Pi에서 오래된 카메라 결과를 새로운 명령으로 계속 전송하지 않도록 하는 검사는 송신기에서도 필요하다.

출력 전용 FreeRTOS 작업이 약 5ms 주기로 만료를 검사하고 LED/PWM에 반영한다. BLE 콜백과 출력 상태 접근은 짧은 critical section으로 보호한다. 이 구간에는 Serial, BLE 읽기/쓰기, 동적 메모리 할당을 넣지 않는다. Serial 보고는 별도 Arduino `loop()`에서 수행하므로 시리얼 로그가 지연되어도 watchdog 작업이 로그를 기다리지 않는다.

출력 반영에는 다음 작업 주기와 스케줄링 지연이 있다. 상태 패킷은 수신기가 결정한 출력이며, 실제 GPIO 변화보다 한 작업 주기 정도 앞설 수 있다. 연결 종료 이벤트 자체는 BLE가 연결 상실을 감지한 뒤 발생한다. 그 이전에도 마지막 명령의 TTL로 출력을 중단한다. 이는 하드 실시간 또는 하드웨어 차단 회로의 보장을 뜻하지 않는다.

## 실제 모터 모드로 전환할 때

회로 확인 후에만 `DRY_RUN = false`로 변경한다. GPIO0/1/3/4/5 순서가 TH/IN/MI/RI/PI에 대응하는 **임시 핀 배정**이며 PCB 확정값이 아니다. PWM은 우선 200Hz, 8비트지만 모터의 기동 세기와 소음에 맞춰 실제 부품에서 조정해야 한다.

모터는 GPIO나 보드 3V3에 직접 연결하지 않는다. 별도 적정 전원, 공통 GND, 채널별 MOSFET, 게이트 저항·풀다운, 역기전력 다이오드 구성을 확인해야 한다. 기동 시에는 외부 게이트 풀다운으로 OFF를 유지한다. 하드웨어 모드의 PWM 초기화가 실패하면 출력을 끈 상태에서 BLE를 시작하지 않는다.

## 호스트 검사

`haptic_protocol.h`는 하드웨어 의존성이 없는 C++ 상태기계다. Raspberry Pi 또는 C++ 컴파일러가 있는 PC에서 테스트한다.

```bash
g++ -std=c++17 -Wall -Wextra -Werror tests/protocol_test.cpp -o /tmp/kimgane-protocol-test
/tmp/kimgane-protocol-test
```

검사 대상: 바이트 배치, 잘못된 패킷 길이/버전/플래그, TTL 경계와 `millis()` 순환, 중복·과거 번호, STOP, 연결 종료와 재연결. Arduino 컴파일과 호스트 검사가 성공해도 실제 BLE 주기·GPIO·모터·전원 동작 검증은 별도다.

### Python 시뮬레이터와 C++ 수신기의 교차 검증

`protocol_bridge.cpp`로 실제 C++ 상태기계를 공유 라이브러리로 빌드한 다음 `parity_test.py`에서 Pi의 `SimulatedReceiver`와 비교한다. 추가 Python 패키지는 필요 없다. 기본값은 seed 1279로 재현되는 20,000개 동작이다. 수신 성공 여부, 상태 플래그, 명령 번호, 남은 TTL, 5개 세기를 확인한다.

Raspberry Pi/Linux에서 현재 스케치 폴더 기준:

```bash
g++ -std=c++17 -Wall -Wextra -Werror -shared -fPIC tests/protocol_bridge.cpp -o /tmp/kimgane-protocol-bridge.so
python3 tests/parity_test.py --library /tmp/kimgane-protocol-bridge.so
```

Windows에서는 Visual Studio 개발자 PowerShell에서:

```powershell
cl /nologo /std:c++17 /EHsc /W4 /WX /LD tests/protocol_bridge.cpp "/Fo$env:TEMP/kimgane-protocol-bridge.obj" "/Fe$env:TEMP/kimgane-protocol-bridge.dll" /link "/IMPLIB:$env:TEMP/kimgane-protocol-bridge.lib"
python tests/parity_test.py --library "$env:TEMP/kimgane-protocol-bridge.dll"
```

기본 Python 프로젝트 경로는 저장소의 `Raspi5_vision`이다. 다른 위치에서 비교할 때는 `--python-root`로 해당 폴더를 지정한다. DLL/공유 라이브러리는 호스트용 검사 파일이며 ESP32에 업로드하지 않는다. 빌드 결과물은 Git에 올리지 않는다.

### 2026-09-19 실행 결과

- Arduino ESP32 코어 3.3.11 + NimBLE-Arduino 2.5.1로 최종 스케치 컴파일 성공. `--warnings all`에서 진단 없음.
- 프로그램 584,737바이트 / 1,310,720바이트(44%), 정적 RAM 23,720바이트 / 327,680바이트(7%). 실행 중 힙·작업 스택 사용량을 포함한 측정값은 아니다.
- MSVC C++17 `/W4 /WX`로 호스트 테스트 6,810개 검사 통과. `/analyze` 정적 분석도 진단 없음.
- C++ 실제 상태기계와 Python 시뮬레이터의 재현 가능한 20,000개 교차 동작 검사 통과. 이 검사에서 Python의 부동소수점 TTL 경계 오차를 발견하여 시뮬레이터를 정수 밀리초 기준으로 수정한 뒤 재검사했다.
- 보드 업로드, 실제 BLE 연결, 모터 전원·PWM 측정은 수행하지 않았다. 기존 LED 스케치의 실기 시험 결과와 구분한다.
