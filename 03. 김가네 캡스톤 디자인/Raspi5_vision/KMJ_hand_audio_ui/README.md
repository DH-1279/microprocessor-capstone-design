# KMJ_hand_audio_ui — 손 인식 보정 · 명령 해석 · 상태 화면

담당: 김민주 (KMJ)
작성일: 2026-10-13

공통 데이터 형식(`haptic_mvp/models.py`)만 참조한다. `runtime`이나 `controller`를 import하지 않는다.

```
KMJ_hand_audio_ui/
├── hand_tracker.py     손 대표점·좌우 판단 (RGB 프레임 -> Hand | None)
├── speech.py           명령 해석 (text -> Intent)
├── ui/
│   ├── server.py       상태 화면 서버
│   └── static/         화면
└── tools/              단독 측정 도구
```

---

## 1. hand_tracker.py — 손 대표점과 좌우 판단

### 입력과 출력

```python
from KMJ_hand_audio_ui.hand_tracker import HandTracker

with HandTracker(config["vision"]) as tracker:
    hand = tracker.detect(rgb, captured_at, frame_id)

# 보정 설정을 바꾸려면 두 번째 인자로 넘긴다
with HandTracker(config["vision"], {"hold_ms": 0}) as tracker:
    ...
```

| | 형식 |
|---|---|
| 입력 | 공통 카메라의 RGB 프레임, 그 프레임의 `captured_at`과 `frame_id` |
| 출력 | `models.Hand` 또는 `None` |

**`frame_id`는 결과가 어느 프레임의 것인지 식별한다.** 같은 번호나 더 오래된 번호가 들어오면 오류를 낸다. 병렬 추론을 넣을 때 물체 인식과 손 인식이 다른 시점의 결과를 섞지 않게 하기 위한 것이다. 넘기지 않아도 동작하므로 기존 두 인자 호출도 그대로 쓸 수 있다.

`last()`로 마지막 결과와 그 프레임 번호를 함께 볼 수 있다.

**카메라를 열지 않는다.** 공통 파이프라인이 한 번 연 프레임을 받는다.

`vision.MediaPipeHandTracker`를 대체한다. 설정 키와 계약은 같고, 측정으로 확인한 보정이 안에 들어 있다.

### 세 층

**`hand_candidates`** — 한 프레임의 쓸 수 있는 손을 모두 돌려준다. 좌우로 거르지 않는다.

**`select_for_vote`** — 투표에 넘길 손을 고른다. 손이 하나면 MediaPipe가 뭐라고 했든 그대로 넘겨서, 뒤집힌 프레임이 버려지지 않고 투표에서 밀리게 한다. 손이 여럿이면 기존 정책이 결정하므로 두 번째 손이 들어오면 여전히 안내가 멈춘다.

**`hand_from_result`** — 기존 호출부를 위해 남겨둔 것. 좌우로 거르고 모호하면 `None`이다. 대표점은 손목과 네 손가락 뿌리 관절 5개의 평균이다. 손이 둘이거나, 요청한 손이 아니거나, 좌표가 범위를 벗어나면 `None`을 돌려준다.

**`HandStabilizer`** — 아래 보정을 한다. 따로 쓸 수도 있다.

```python
stabilizer = HandStabilizer(config.get("hand_stability"), "Right")
hand = stabilizer.update(raw_hand, captured_at)
```

프레임마다 한 번씩 호출해야 한다. 손이 없는 프레임도 넘겨야 유지 시간이 제때 끝난다.

### 하는 일

**좌우 판정 다수결.** 최근 프레임을 모아 결정한다. 한 프레임이 뒤집혀도 결과가 바뀌지 않는다.

**짧은 소실 유지.** 마지막으로 측정된 점을 기본 400ms까지 반복한다. 측정된 적 없는 손은 유지하지 않으며, 시간이 지나면 버린다.

### 설정

**공통 설정 파일에는 넣을 수 없다.** `config.validate()`가 모르는 키를 거부하므로 `vision.hand_stability`를 `config/mvp.json`에 적으면 `Unknown configuration key` 오류가 난다. 공통 설정에 키를 추가하기 전까지는 생성자 인자로 넘긴다.

| 키 | 기본값 | 뜻 |
|---|---|---|
| `vote_frames` | 5 | 좌우 판정에 쓸 프레임 수 |
| `vote_margin` | 2 | 판정을 바꾸려면 벌어져야 하는 표 차이 |
| `hold_ms` | 400 | 마지막 점을 반복할 시간. 0이면 유지 없음 |
| `min_score` | 0.0 | `vision.hand_score_threshold` 위에 더하는 문턱 |

`stability_config()`가 카메라를 열기 전에 검사한다.

`stats()`로 보정이 얼마나 개입했는지 볼 수 있다. `hold_ratio`가 크면 유지가 과한 것이다.

### 왜 필요한가

2026-10-07 측정(노트북 웹캠, 면장갑, 오른손).

| 자세 | 맨손 | 분홍 | 흰색 | 검정 |
|---|---:|---:|---:|---:|
| 손등 | 100% | 100% | — | 100% |
| 손바닥 | 100% | 100% | — | 100% |
| 세운 손 | 99.6% | 32.8% | 36.2% | 0.0% |

손등·손바닥은 색상과 무관하게 100%다. **세운 손에서만 무너진다.** 물건을 집으려고 손을 기울이면 그 조건이 된다.

그리고 **장갑을 끼고 손을 완전히 펴면 좌우 판정이 흔들린다.** 분홍과 검정 모두에서 오른손이 `Left`로 나왔고, 손가락을 살짝 구부리면 제대로 잡았다. `vision.hand`가 `Right`이므로 뒤집힌 프레임은 버려지고 안내가 멈춘다. 랜드마크 자체는 정상인데도 그렇다.

MediaPipe의 `Hand.score`는 좌우 분류 점수다. 관절 좌표 정확도나 검출 확률이 아니다.

측정 기록: `docs/2026-10-07_KMJ_장갑_손인식률_측정.md`

### vision.py 연결

`MediaPipeHandTracker`를 이 모듈로 바꾸면 된다.

```python
from KMJ_hand_audio_ui.hand_tracker import HandTracker

# run_vision 안
tracker = resources.enter_context(HandTracker(config))
```

`detect()` 호출부는 그대로 두어도 되고, `frame_id`를 함께 넘기면 프레임 섞임을 막는다. 인자를 넘기지 않으면 기본값으로 동작하고, `HandTracker(config, {"hold_ms": 0})`으로 두면 유지가 꺼진다.

**보정 전후를 비교해 측정한 뒤에 켜는 것을 제안한다.** 유지가 과하면 이미 사라진 손으로 안내하게 되므로, 실제 장갑으로 `stats().hold_ratio`를 보고 정하면 좋겠다.

---

## 2. speech.py — 명령 해석

### 입력과 출력

```python
from KMJ_hand_audio_ui import speech

intent = speech.parse_input(text, targets, state)
if intent:
    intent.kind   # target | control | direction
    intent.name   # COCO 라벨 / stop, list, start, resume, done, repeat / left, right
else:
    intent.reason           # 왜 받지 않았는지
    speech.refusal_speech(intent)   # 사용자에게 할 말
```

| | 형식 |
|---|---|
| 입력 | STT 출력 문자열 또는 화면 버튼 식별자, `controller.targets`, 상태 이름 |
| 출력 | `Intent` (frozen dataclass) |

**라벨과 별칭은 `controller.targets`에서 읽는다.** COCO 라벨(`cell phone`)과 설정의 별칭(`머그`, `생수`)이 그대로 동작하며 사전이 두 곳으로 갈라지지 않는다.

### 하는 일

문장의 모든 어절이 명령 문법에 맞아야 받는다. 모르는 말이 하나라도 섞이면 거부한다.

| 입력 | 결과 |
|---|---|
| 책상 정리했어 | 거부 (unknown_word) |
| 컵 말고 책 | 거부 (negation) |
| 컵이랑 책 | 거부 (multiple_targets) |
| 마우스패드 | 거부 |
| 컵 정지 | 거부 (mixed_command) |

한 어절 안의 오인식은 자모 편집거리로 되돌린다. 아래는 2026-10-12 마이크 측정에서 실제로 나온 출력이다.

```
컴 찾아줘 -> cup        눈병 찾아줘 -> bottle    물농 찾아줘 -> bottle
휴데폰    -> cell phone 마우스 타자줘 -> mouse   전지       -> stop
모른 쪽   -> right      컵찾아줘    -> cup
```

어절 안에서만 보정하므로 "책상"은 "책"이 되지 않는다.

**상태별로 받을 명령을 제한한다.** 안내 중에는 새 목표를 받지 않고, 손 복구 대기에서는 "계속"만 받는다. 중지는 모든 상태에서 받는다.

**화면 버튼도 같은 함수를 지난다.** `parse_input("select_cup", targets)`가 음성과 같은 경로로 들어간다.

### 상태 이름

| 이름 | 받는 명령 |
|---|---|
| `scanning` | 목표 선택, 목록 |
| `selecting` | 왼쪽, 오른쪽 |
| `confirming` | 시작, 목표 변경 |
| `guiding` | 목록, 다시 |
| `paused` | 다시 |
| `waiting` | 계속 |
| `near` | 찾았어 |
| `stopped` | 목표 선택, 목록 |

모든 상태에 중지가 포함된다.

---

## 3. ui/ — 상태 화면

### 연결 방법

```python
from KMJ_hand_audio_ui.ui.server import serve

serve(lambda: tuple(latest), on_text, thread_stop,
      config["controller"]["targets"])
```

`transmit_loop`와 같은 계약이다. **같은 `Decision`을 읽으므로 화면이 장갑과 다른 상태를 보여줄 수 없다.** 버튼은 `on_text`로 들어가 음성과 같은 경로로 컨트롤러에 도달한다.

화면은 별도 목표 상태를 갖지 않고 모터를 직접 켜지 않는다.

### 설치

```
python -m pip install fastapi "uvicorn[standard]"
```

`uvicorn[standard]`가 필요하다. 기본 `uvicorn`에는 WebSocket 백엔드가 없어 실제 서버에서 `/ws` 연결이 되지 않는다.

**없어도 프로그램은 돌아간다.** 메시지만 남기고 넘어간다. 브라우저가 끊겨도 안내는 계속된다. 메인 루프를 막지 않도록 별도 스레드에서 돈다.

### 화면 구성

현재 상태 → 고른 물건 → 할 수 있는 것 → 물건 고르기 순서다. 카메라 영상과 기술 지표는 접힌 점검 영역에 있다.

상태에 따라 버튼이 바뀐다. 안내 중에는 물건 버튼이 꺼지고 이유를 알린다. 중지는 모든 상태에서 보인다. `near`에서만 "찾았어요"가 나온다.

### 색과 대비

**저시력 사용자를 기준으로 고대비 다크 모드를 기본값으로 둔다.**

흰 배경은 빛 번짐과 눈부심을 만든다. 노랑은 망막이 가장 민감하게 반응하는 파장대라 어두운 배경 위에서 잘 보인다. 주변 시야가 좁거나 눈부심에 약한 경우에도 어두운 배경이 낫다.

| 조합 | 대비비 |
|---|---:|
| 검정 `#000` + 노랑 `#FFD400` | 14.7:1 |
| 검정 `#000` + 흰색 `#FFF` | 21.0:1 |
| 패널 `#111` + 노랑 `#FFD400` | 13.2:1 |
| 노랑 버튼 + 검정 글자 | 14.7:1 |
| 노랑 배경 `#FFE800` + 검정 글자 | 16.8:1 |

WCAG 2.2 AA의 일반 텍스트 기준은 4.5:1이다. 연한 색끼리의 조합과 비슷한 명도의 조합은 쓰지 않는다.

밝은 화면이 필요한 경우를 위해 노랑 배경 + 검정 글자를 선택지로 둔다.

**모든 버튼에 4px 이상의 테두리를 둔다.** 경계가 뚜렷해야 형태를 찾기 쉽다.

### 그 밖의 접근성

기본 글자 22px에 4단계 조절, 버튼 최소 64px, 키보드 조작, 건너뛰기 링크, `role=status`로 상태 변화만 전달, 색만으로 상태를 구분하지 않고 글자를 함께 씀, 520px 아래에서 세로 배치, 강제 색상 모드 대응.

**개발용 수치는 이 화면에 두지 않는다.** 명세 1장의 "사용자 화면과 개발 점검 화면을 구분한다"를 따랐다. 상태·사유·채널 값은 터미널 출력으로 확인한다.

---

## 4. tools/ — 단독 측정 도구

통합 실행에 쓰지 않는다. **`hand_bench.py`는 카메라를 직접 연다.** 공통 프로그램과 동시에 실행하지 않는다.

```
python KMJ_hand_audio_ui/tools/hand_bench.py --label 조건이름
python KMJ_hand_audio_ui/tools/stt_bench.py --name K66 --model tiny
python -m KMJ_hand_audio_ui.tools.prebuild_tts
```

| 도구 | 하는 일 | 마이크·카메라 |
|---|---|---|
| `console_check.py` | 터미널로 명령 경로 확인 | 필요 없음 |
| `hand_bench.py` | 웹캠으로 손 인식률 측정 | 카메라 |
| `stt_bench.py` | 마이크로 명령 인식률 측정 | 마이크 |
| `prebuild_tts.py` | 안내 음성을 미리 만들어 런타임 캐시에 넣음 | 필요 없음 |

### console_check.py

**마이크가 오기 전까지 명령 경로를 확인하는 도구다.** `run --console`이 이미 키보드 입력을 받으므로, 사용자가 말할 문장을 그대로 넣어 `speech.py`가 어떻게 판단하는지 볼 수 있다.

```
python -m KMJ_hand_audio_ui.tools.console_check          37개 문장 일괄 확인
python -m KMJ_hand_audio_ui.tools.console_check --live   직접 입력
python -m KMJ_hand_audio_ui.tools.console_check --state guiding
```

문장 목록에 2026-10-12 마이크 측정에서 whisper가 실제로 내보낸 출력이 들어 있다. 해석이 바뀌면 여기서 바로 드러난다.

설정을 읽을 수 있으면 `controller.targets`를 쓰고, 없으면 기본 목록으로 돌아간다.

### prebuild_tts.py

`audio.py`를 고치지 않는다. `tts_cache_path()`가 `sha256("espeak-ng\0voice\0rate\0text")`로 파일명을 만들고 파일이 있으면 재사용하므로, **같은 규칙으로 미리 WAV를 넣어두면 실행 중에 합성하지 않는다.**

**문장은 `controller.py`가 실제로 내보내는 것과 글자까지 같아야 한다.** 해시가 같아도 텍스트가 다르면 런타임이 다시 합성한다. 그래서 이 도구는 `controller.step()`과 `_request()`의 문자열을 그대로 쓴다.

`controller`가 조사를 `을`로 고정해 "마우스을", "휴대전화을"을 내보낸다. 한국어로는 틀렸지만 여기서 고치면 캐시가 어긋나므로 그대로 따른다. 조사를 고치는 것은 `controller.py` 쪽 일이다.

주변 물체 목록 안내는 보이는 라벨을 이어 붙이므로 조합 수가 많다. 물체 하나짜리 형태만 미리 만들고 나머지는 기존대로 실행 중에 합성한다.

파이에서 espeak 프로세스가 뜨지 않고 첫 발화 지연이 사라진다.

문장은 `controller.targets`에서 만들고, 조사는 받침에 따라 갈린다. 컵을/마우스를, 책이/마우스가.

`audio.tts_voice`나 `tts_rate`를 바꾸면 캐시 키가 바뀌므로 다시 만들어야 한다.

---

## 5. 시험

```
python3 -m unittest discover -s tests -p 'test_kmj_*.py'
```

전체 시험은 개발 의존성이 있는 환경에서 `python -m pytest -q`로 수행한다. `python -m unittest discover tests`는 pytest를 쓰는 기존 시험까지 불러온다.

| 파일 | 건수 |
|---|---|
| `tests/test_kmj_speech.py` | 31 |
| `tests/test_kmj_ui.py` | 22 (WebSocket 4건은 fastapi 필요) |
| `tests/test_kmj_hand_tracker.py` | 26 |

카메라·마이크·Hailo·BLE 없이 실행된다.

---

## 6. 실기기 검증 범위

### 확인한 것

- 명령 해석, 상태 전이, 화면 계약: 단위 시험 69건
- 터미널 명령 경로: `console_check.py` 37개 문장
- STT 인식률: 데스크톱 + ZEALSOUND K66S-C, faster-whisper, 명령 9종 각 3회
- 손 인식률: 노트북 웹캠 + 면장갑 5종
- TTS 캐시 해시: `audio.tts_cache_path`와 일치

### 확인하지 못한 것

- **라즈베리파이에서의 동작 전반.** 모든 측정은 노트북·데스크톱에서 했다.
- **whisper.cpp 실측.** STT 측정은 faster-whisper로 했다. 인식률은 모델이 좌우하지만 속도는 구현에 따라 다르다.
- **Camera Module 3와 AI HAT+ 2에서의 손 인식률.** 웹캠 측정과 다를 수 있다.
- **모터와 PCB가 부착된 장갑.** 측정에 쓴 장갑에는 부착물이 없었다.
- **실제 화면낭독기 발화와 당사자 사용.** UX 명세 UX-10은 아직 수행하지 않았다.
- **`hand_tracker`의 유지 설정이 적절한지.** 과하면 이미 사라진 손으로 안내하게 된다. 실제 장갑으로 `stats().hold_ratio`를 보고 정해야 한다.

---

## 7. 알려진 문제

**"목록"이 STT에서 자주 실패한다.** 2026-10-12 측정에서 tiny·base 모두 3회 중 1회 틀렸고, 다른 8개 명령은 전부 맞았다. **"리스트"로 말하면 둘 다 성공한다.** `speech.py`에 둘 다 별칭으로 넣어 두었으니 사용자 안내만 "리스트"로 하면 된다.

**eSpeak NG 한국어 음질이 아슬아슬하다.** 못 알아들을 정도는 아니지만 집중해서 들어야 하는 편이다. 사용자가 시각장애인이고 음성이 주 정보 채널이므로 개선 여지가 있다.

Piper를 검토했으나 지금 쓰기 어렵다. 2024년 12월 한국어 지원 요청에 2025년 9월에야 커뮤니티 모델 링크 하나가 달렸고, 2026년 7월 자료도 Piper 커뮤니티가 한국어에 쓸 만한 음성을 만들지 못했다고 기술한다. 유일한 모델(neurlang/piper-onnx-kss-korean)은 추론에 Rust 구현이 필요하고 음소 처리가 표준 Piper와 다르며 라이선스가 cc-by-nc-sa-4.0이다.

사전 생성 구조에서는 합성 엔진이 파이에서 돌 필요가 없으므로, 나중에 더 나은 엔진을 쓰더라도 `audio.py` 변경 없이 WAV만 교체하면 된다.

**세운 손은 보정으로 해결되지 않는다.** `hold_ms`는 짧은 소실을 메울 뿐이다. 검정 장갑의 세운 손은 283프레임 중 0프레임이었다. 밝은 색 장갑과 표면 무늬 없는 재질이 함께 필요하다.

**공통 파일이 아직 이 모듈을 쓰지 않는다.** `vision.py`는 `MediaPipeHandTracker`를, `controller.py`는 부분 문자열 판정을 그대로 쓰고 있다. 연결 방법은 각 절과 PR 설명에 적어 두었다. 담당 파일이 아니라 상의 후에 진행한다.
