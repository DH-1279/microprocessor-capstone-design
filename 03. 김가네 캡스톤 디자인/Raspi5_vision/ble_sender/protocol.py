"""BLE v2 only. Deliberately incompatible UUIDs with haptic_mvp protocol v1."""

import struct
from dataclasses import dataclass
from enum import IntEnum

NAME = "KIMGANE-HAPTIC-V2"
SERVICE = "328e0020-8c2a-4e58-9a48-512fc6ab1279"
COMMAND = "328e0021-8c2a-4e58-9a48-512fc6ab1279"
STATUS = "328e0022-8c2a-4e58-9a48-512fc6ab1279"
PACKET = struct.Struct("<BBHHBBBB5BB")


class Mode(IntEnum):
    RATE = 2
    DIRECTION = 3


class Zone(IntEnum):
    FAR = 0
    MIDDLE = 1
    NEAR = 2


class Channel(IntEnum):
    TH = 0
    IN = 1
    MI = 2
    RI = 3
    PI = 4


@dataclass(frozen=True)
class Cue:
    mode: Mode
    zone: Zone
    channel: Channel
    duties: tuple[int, ...] = (180, 180, 180, 180, 180)
    zones: tuple[int, ...] | None = None
    arrival_id: int | None = None

    def describe(self):
        if self.arrival_id is not None:
            return f"도달 id={self.arrival_id} | ALL duty=255 | 150/150/150ms 한 회"
        if self.zones is not None:
            zones = ",".join(f"{ch.name}={Zone(z).name}" for ch, z in zip(Channel, self.zones))
            return f"mode=2 | {zones} | duty={self.duties}"
        return f"mode={int(self.mode)} {Zone(self.zone).name}/{Channel(self.channel).name} duty={self.duties[self.channel]}"

    def validate(self):
        if (
            any(not isinstance(n, int) or isinstance(n, bool) for n in (self.mode, self.zone, self.channel))
            or self.mode not in (2, 3)
            or self.zone not in (0, 1, 2)
            or self.channel not in range(5)
        ):
            raise ValueError("invalid mode, zone or channel")
        if len(self.duties) != 5 or any(type(n) is not int or not 1 <= n <= 255 for n in self.duties):
            raise ValueError("five calibrated duties, each 1..255, are required")
        if self.arrival_id is not None:
            if type(self.arrival_id) is not int or not 1 <= self.arrival_id <= 65535 or self.zones is not None:
                raise ValueError("invalid arrival event")
            if self.duties != (255,) * 5:
                raise ValueError("arrival uses maximum PWM on all five channels")
        if self.zones is not None:
            if self.mode != Mode.RATE or len(self.zones) != 5 or any(type(z) is not int or z not in range(3) for z in self.zones):
                raise ValueError("mode 2 needs five zones, each 0..2")
        if self.mode == Mode.DIRECTION and ((self.zone == Zone.NEAR) != (self.channel == Channel.MI)):
            raise ValueError("plan 3 reserves MI for the near event")


def encode(sequence: int, cue: Cue | None, ttl_ms: int = 500) -> bytes:
    if type(sequence) is not int or not 0 <= sequence <= 65535:
        raise ValueError("sequence must be uint16")
    if type(ttl_ms) is not int or not 100 <= ttl_ms <= 1000:
        raise ValueError("TTL must be 100..1000 ms")
    if cue is None:
        return PACKET.pack(2, 1, sequence, ttl_ms, 0, 0, 0, 0, *([0] * 5), 0)
    cue.validate()
    if cue.arrival_id is not None:
        return PACKET.pack(2, 4, sequence, ttl_ms, int(cue.mode), cue.arrival_id & 255,
                           cue.arrival_id >> 8, 0, *cue.duties, 0)
    if cue.zones is not None:
        packed = sum(z << (2 * i) for i, z in enumerate(cue.zones))
        return PACKET.pack(2, 2, sequence, ttl_ms, 2, packed & 255, packed >> 8, 0, *cue.duties, 0)
    return PACKET.pack(2, 0, sequence, ttl_ms, cue.mode, cue.zone, cue.channel, 0, *cue.duties, 0)


@dataclass(frozen=True)
class Status:
    flags: int
    sequence: int
    remaining_ms: int
    mode: int
    zone: int
    channel: int
    on: bool
    duties: tuple[int, ...]
    capabilities: int = 0

    @classmethod
    def decode(cls, data):
        if len(data) != PACKET.size:
            raise ValueError("upload ble_final_receiver: status must be 16 bytes")
        version, flags, seq, ttl, mode, zone, channel, phase, *tail = PACKET.unpack(data)
        if (
            version != 2
            or flags & ~63
            or ttl > 1000
            or mode not in (2, 3)
            or zone > 2
            or channel > 4
            or phase > 1
            or tail[-1] & ~3
        ):
            raise ValueError("invalid v2 status")
        return cls(flags, seq, ttl, mode, zone, channel, bool(phase), tuple(tail[:5]), tail[-1])
