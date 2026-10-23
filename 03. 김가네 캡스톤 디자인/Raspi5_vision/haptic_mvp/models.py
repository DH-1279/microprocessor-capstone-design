"""Hardware-independent messages. All image coordinates are normalized."""

from dataclasses import dataclass
import math


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class Detection:
    label: str
    box: tuple[float, float, float, float]
    score: float

    def valid(self) -> bool:
        if not isinstance(self.box, (tuple, list)) or len(self.box) != 4 or not all(finite_number(x) for x in self.box):
            return False
        x1, y1, x2, y2 = self.box
        return (isinstance(self.label, str) and bool(self.label)
                and 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
                and finite_number(self.score) and 0 <= self.score <= 1)


@dataclass(frozen=True)
class Hand:
    point: tuple[float, float]
    score: float
    handedness: str

    def valid(self) -> bool:
        return (isinstance(self.point, (tuple, list)) and len(self.point) == 2
                and all(finite_number(v) and 0 <= v <= 1 for v in self.point)
                and finite_number(self.score) and 0 <= self.score <= 1
                and self.handedness in ("Right", "Left", "Unknown"))


@dataclass(frozen=True)
class Observation:
    frame_id: int
    captured_at: float
    objects: tuple[Detection, ...]
    hand: Hand | None

    def valid(self) -> bool:
        return (type(self.frame_id) is int and self.frame_id >= 0
                and finite_number(self.captured_at) and self.captured_at >= 0
                and isinstance(self.objects, (tuple, list))
                and all(isinstance(d, Detection) and d.valid() for d in self.objects)
                and (self.hand is None or isinstance(self.hand, Hand) and self.hand.valid()))


@dataclass(frozen=True)
class Decision:
    levels: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    state: str = "scanning"
    reason: str = "idle"
    target: str | None = None
    speech: str | None = None
    direction: str | None = None
    proximity: str | None = None

    @property
    def stopped(self) -> bool:
        return not any(self.levels)
