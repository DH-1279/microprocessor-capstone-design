"""Confirmed object list for speech/display, never a source of control coordinates."""

from collections import deque
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class ListedObject:
    detection: Any
    seen_at: float


class ObjectInventory:
    def __init__(self, *, window=5, hits=3, hold_s=3.0, max_age=0.35):
        if (type(window) is not int or type(hits) is not int
                or not 1 <= hits <= window <= 30
                or not all(math.isfinite(v) and v > 0 for v in (hold_s, max_age))):
            raise ValueError("invalid inventory window/hits/hold time")
        self.window, self.hits, self.hold_s, self.max_age = window, hits, hold_s, max_age
        self._history = deque(maxlen=window)
        self._objects = {}
        self._confirmed = set()
        self._last_capture = -math.inf

    def snapshot(self, now):
        for label, item in tuple(self._objects.items()):
            if not 0 <= now - item.seen_at <= self.hold_s:
                del self._objects[label]
                self._confirmed.discard(label)
        return tuple(self._objects[label] for label in sorted(self._confirmed))

    def update(self, detections, captured_at, now):
        self.snapshot(now)
        if (not all(math.isfinite(v) for v in (captured_at, now))
                or not 0 <= now - captured_at <= self.max_age
                or captured_at <= self._last_capture):
            return self.snapshot(now)
        if captured_at - self._last_capture > self.max_age:
            self._history.clear()
        self._last_capture = captured_at
        best = {}
        for detection in detections or ():
            label = detection.label
            if label not in best or detection.score > best[label].score:
                best[label] = detection
        self._history.append(frozenset(best))
        for label, detection in best.items():
            self._objects[label] = ListedObject(detection, captured_at)
            if sum(label in frame for frame in self._history) >= self.hits:
                self._confirmed.add(label)
        return self.snapshot(now)
