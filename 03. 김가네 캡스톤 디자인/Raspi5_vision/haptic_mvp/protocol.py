"""Versioned 11-byte BLE protocol; shared with haptic_receiver firmware."""

from dataclasses import dataclass
import struct

SERVICE_UUID = "328e0010-8c2a-4e58-9a48-512fc6ab1279"
COMMAND_UUID = "328e0011-8c2a-4e58-9a48-512fc6ab1279"
STATUS_UUID = "328e0012-8c2a-4e58-9a48-512fc6ab1279"
PACKET = struct.Struct("<BBHH5B")
VERSION = 1
STOP = 1
MIN_TTL_MS = 100
MAX_TTL_MS = 1000


@dataclass(frozen=True)
class Command:
    sequence: int
    ttl_ms: int
    levels: tuple[int, int, int, int, int]
    stop: bool = False

    def encode(self) -> bytes:
        if type(self.sequence) is not int or not 0 <= self.sequence <= 65535:
            raise ValueError("sequence must be uint16")
        if type(self.ttl_ms) is not int or not MIN_TTL_MS <= self.ttl_ms <= MAX_TTL_MS:
            raise ValueError("ttl_ms must be 100..1000")
        if len(self.levels) != 5 or any(type(x) is not int or not 0 <= x <= 255 for x in self.levels):
            raise ValueError("five uint8 levels required")
        if type(self.stop) is not bool or self.stop and any(self.levels):
            raise ValueError("STOP requires zero levels")
        return PACKET.pack(VERSION, STOP if self.stop else 0, self.sequence, self.ttl_ms, *self.levels)

    @classmethod
    def decode(cls, data: bytes) -> "Command":
        if len(data) != PACKET.size:
            raise ValueError("packet must be 11 bytes")
        version, flags, seq, ttl, *levels = PACKET.unpack(data)
        if version != VERSION or flags & ~STOP:
            raise ValueError("unsupported version or flags")
        command = cls(seq, ttl, tuple(levels), bool(flags & STOP))
        command.encode()
        return command


def is_newer(sequence: int, previous: int) -> bool:
    return 0 < ((sequence - previous) & 0xFFFF) < 0x8000


@dataclass(frozen=True)
class Status:
    flags: int
    sequence: int
    remaining_ms: int
    levels: tuple[int, int, int, int, int]

    @classmethod
    def decode(cls, data: bytes) -> "Status":
        if len(data) != PACKET.size:
            raise ValueError("status must be 11 bytes; upload haptic_receiver")
        version, flags, seq, remaining, *levels = PACKET.unpack(data)
        if version != VERSION or flags & ~31 or remaining > MAX_TTL_MS:
            raise ValueError("invalid status")
        return cls(flags, seq, remaining, tuple(levels))


class SimulatedReceiver:
    """Deterministic receiver model for packet loss/timeout simulation."""

    def __init__(self) -> None:
        self.sequence: int | None = None
        self.levels = (0, 0, 0, 0, 0)
        self.deadline = 0
        self.connected = False
        self.expired = False
        self.invalid = False

    def connect(self) -> None:
        self.connected = True
        self.sequence = None
        self.levels = (0, 0, 0, 0, 0)
        self.deadline = 0
        self.expired = self.invalid = False

    def disconnect(self) -> None:
        self.connected = False
        self.levels = (0, 0, 0, 0, 0)
        self.sequence = None
        self.deadline = 0
        self.expired = self.invalid = False

    def _reject(self) -> bool:
        self.levels = (0, 0, 0, 0, 0)
        self.deadline = 0
        self.invalid = True
        return False

    def receive(self, data: bytes, now: float) -> bool:
        if not self.connected:
            return self._reject()
        try:
            command = Command.decode(data)
        except ValueError:
            return self._reject()
        newer = self.sequence is None or is_newer(command.sequence, self.sequence)
        if command.stop:
            self.levels = (0, 0, 0, 0, 0)
            self.deadline = 0
            if newer:
                self.sequence = command.sequence
            self.expired = self.invalid = False
            return True
        if not newer:
            return self._reject()
        self.sequence = command.sequence
        self.levels = command.levels
        # Firmware uses integer milliseconds; float addition can miss the exact
        # timeout boundary (13.220 + 0.300 > 13.520 in binary floating point).
        self.deadline = round(now * 1000) + command.ttl_ms if any(command.levels) else 0
        self.expired = self.invalid = False
        return True

    def tick(self, now: float) -> None:
        if self.deadline and round(now * 1000) >= self.deadline:
            self.levels = (0, 0, 0, 0, 0)
            self.deadline = 0
            self.expired = True
