"""Single-event-loop state: fresh vision -> mapping -> desired BLE cue."""

import json
import math
import secrets
from collections import deque
from time import monotonic

from .guidance import Mapper, fresh_cue
from .protocol import Channel, Cue, Mode, Zone
from .publisher import MAX_DATAGRAM
from .arrival import valid_palm_observation


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


class Controller:
    def __init__(
        self,
        *,
        mode=2,
        duties=(180,) * 5,
        rate_channel=Channel.IN,
        max_age=0.35,
        stable_frames=3,
        near=0.35,
        middle=1.5,
        mirrored=False,
        allow_test=False,
        clock=monotonic,
        multi_rate=False,
    ):
        self.clock = clock
        self.mapper_options = dict(
            duties=duties,
            rate_channel=rate_channel,
            max_age=max_age,
            stable_frames=stable_frames,
            near=near,
            middle=middle,
            mirrored=mirrored,
            multi_rate=multi_rate,
        )
        self.mapper = Mapper(mode=mode, **self.mapper_options)
        self.mode = Mode(mode)
        self.allow_test = allow_test
        self.paused = False
        self.link_ready = True
        self.session = None
        self.last_sequence = -1
        self.last_frame = -1
        self.last_capture = -math.inf
        self.last_packet_at = -math.inf
        self.target_id = None
        self.guidance = None
        self.test_cue = None
        self.test_until = 0
        self.floor = self.clock()
        self.reason = "영상 입력 대기"
        self.completed = deque(maxlen=128)
        self.arrival_cue = None
        self.arrival_until = 0
        self.event_id = secrets.randbelow(65535)

    def clear(self, reason):
        self.arrival_cue = None
        self.mapper.reset()
        self.guidance = None
        self.target_id = None
        self.test_cue = None
        self.reason = reason

    def set_link_ready(self, connected):
        """Link recovery never overrides an operator's explicit stop."""
        self.link_ready = connected
        self.floor = self.clock()
        self.clear("BLE 연결: 새 프레임 대기" if connected else "장갑 연결 대기")

    def receive(self, data, now=None):
        now = self.clock() if now is None else now
        try:
            if len(data) > MAX_DATAGRAM:
                raise ValueError("oversize")
            packet = json.loads(data)
            if (
                not isinstance(packet, dict)
                or packet.get("version") != 1
                or type(packet.get("version")) is not int
                or type(packet.get("active")) is not bool
                or not isinstance(packet.get("session"), str)
                or not 1 <= len(packet["session"]) <= 64
                or not integer(packet.get("sequence"), 1)
            ):
                raise ValueError("bad envelope")
            session = packet["session"]
            if session != self.session:
                # A second producer cannot steal an active stream. A restarted
                # producer can take over only after the previous one is silent.
                if now - self.last_packet_at <= self.mapper.max_age:
                    return False
                self.clear("새 영상 프로세스 연결")
                self.session, self.last_sequence = session, -1
                self.last_frame, self.last_capture = -1, -math.inf
                self.mapper.last_capture = -math.inf
            if packet["sequence"] <= self.last_sequence:
                return False
            self.last_sequence = packet["sequence"]
            self.last_packet_at = now
            if not packet["active"]:
                if packet.get("idle") is True and self.arrival_cue is not None:
                    return True  # Routine WAIT_WAKE must not truncate the event.
                self.clear("시동어 대기 / 취소 / 종료 / 관측 상실")
                return True
            captured = packet.get("captured_at")
            frame_id = packet.get("frame_id")
            target_id = packet.get("target_id")
            if (
                type(captured) not in (float, int)
                or not math.isfinite(captured)
                or not 0 <= now - captured <= self.mapper.max_age
                or not integer(frame_id)
                or frame_id <= self.last_frame
                or captured <= self.last_capture
                or captured <= self.floor
                or not isinstance(target_id, str)
                or not 1 <= len(target_id) <= 128
            ):
                raise ValueError("stale, unmatched or invalid observation")
            self.last_frame, self.last_capture = frame_id, captured
            seen = packet.get("target_seen_at", captured)
            if (type(seen) not in (int, float) or not math.isfinite(seen)
                    or not 0 <= captured - seen or not 0 <= now - seen):
                raise ValueError("invalid locked target timestamp")
            if not self.link_ready:
                if packet.get("event") == "arrival" and target_id not in self.completed:
                    self.completed.append(target_id)  # Never replay an offline completion.
                self.guidance = self.arrival_cue = self.test_cue = None
                return True
            if self.paused:
                return True
            if target_id != self.target_id:
                self.clear("새 목표 안정화 중")
                self.target_id = target_id
            # Actual incoming frames take precedence over a timed bench cue.
            self.test_cue = None
            self.guidance = self.mapper.update(
                packet.get("landmarks"),
                packet.get("target_box"),
                captured,
                now,
                packet.get("size"),
                observed=True,
            )
            if packet.get("event") == "arrival":
                if target_id in self.completed:
                    self.guidance = None
                    return True
                # Acquisition needs a valid palm, not a palm/back direction axis.
                if not valid_palm_observation(packet.get("landmarks"), packet.get("target_box"), captured,
                                              now, packet.get("size"), self.mapper.max_age):
                    raise ValueError("invalid arrival observation")
                self.completed.append(target_id)
                self.event_id = self.event_id % 65535 + 1
                self.arrival_cue = Cue(self.mode, Zone.NEAR,
                                       Channel.MI if self.mode == Mode.DIRECTION else Channel.IN,
                                       (255,) * 5, arrival_id=self.event_id)
                # A finite event is not a stream of refreshed camera positions.
                # Firmware runs exactly 900ms once; allow delivery/ACK latency.
                self.arrival_until = now + 1.9
                self.guidance = None
                self.reason = "획득: 다섯 손가락 최대 PWM, 350ms씩 두 번"
                return True
            if target_id in self.completed:
                self.guidance = None
            self.reason = ("잠금 위치 안내 (목표 가림/미검출)" if seen < captured else "안내") if self.guidance else "손 좌표 검증 / 방향 안정화 중"
            return True
        except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
            self.clear("잘못되거나 오래된 영상 입력: 정지")
            return False

    def cue(self, now=None):
        now = self.clock() if now is None else now
        if self.paused or not self.link_ready:
            return None
        if self.arrival_cue is not None:
            if now < self.arrival_until:
                return self.arrival_cue
            self.clear("도달 알림 완료: 정지")
        if self.test_cue is not None:
            if now < self.test_until:
                return self.test_cue
            self.clear("시험 종료: 정지")
        result = fresh_cue(self.guidance, now, self.mapper.max_age)
        if result is None and self.guidance is not None:
            self.clear("영상 입력 시간 초과: 정지")
        return result

    def command(self, line):
        """Return (reply, force_stop, quit). CLI sends STOP before next cue."""
        words = line.lower().split()
        if not words:
            return "", False, False
        if words[0] in ("quit", "exit", "q"):
            self.clear("종료")
            return "종료합니다.", True, True
        if words == ["stop"]:
            self.paused = True
            self.clear("터미널 정지: resume 필요")
            return self.reason, True, False
        if words == ["resume"]:
            self.paused = False
            self.floor = self.clock()
            self.clear("재개: 새 영상 입력 대기")
            return self.reason, True, False
        if len(words) == 2 and words[0] == "mode" and words[1] in ("2", "3"):
            mode = Mode(int(words[1]))
            if mode == self.mode:
                return f"현재 mode={mode.value}", False, False
            self.mode = mode
            self.mapper = Mapper(mode=mode, **self.mapper_options)
            self.floor = self.clock()
            self.clear("모드 변경: 새 영상 입력 대기")
            return f"mode={mode.value} 적용. {self.reason}", True, False
        if words == ["status"]:
            current = self.cue()
            text = (
                "STOP"
                if current is None
                else current.describe()
            )
            return f"mode={self.mode.value} paused={self.paused} | {text} | {self.reason}", False, False
        if words[0] == "test":
            if not self.link_ready:
                return "장갑 연결 전에는 모터 시험을 실행하지 않습니다.", False, False
            if not self.allow_test:
                return "독립 모터 시험은 --allow-test로 실행해야 합니다.", False, False
            try:
                if len(words) != 4 or self.paused:
                    raise ValueError("test 형식 또는 resume 필요")
                zone = Zone[words[1].upper()]
                channel = Channel[words[2].upper()]
                seconds = float(words[3])
                if not math.isfinite(seconds) or not 0 < seconds <= 2:
                    raise ValueError("시험 시간은 0초 초과, 2초 이하")
                # Fixed channel in mode 2; MI is reserved for mode 3 near.
                if self.mode == Mode.RATE and not self.mapper.multi_rate and channel != self.mapper.rate_channel:
                    raise ValueError("mode 2는 --rate-channel로 지정한 채널만 사용")
                cue = Cue(self.mode, zone, channel, self.mapper.duties)
                cue.validate()
            except (ValueError, KeyError) as error:
                return f"시험 거절: {error}; 예: test middle th 2", False, False
            self.clear("터미널에서 요청한 단독 모터 시험")
            self.test_cue, self.test_until = cue, self.clock() + seconds
            return f"시험 {seconds:g}초 | {zone.name}/{channel.name}", True, False
        return "명령: mode 2 | mode 3 | status | stop | resume | quit", False, False
