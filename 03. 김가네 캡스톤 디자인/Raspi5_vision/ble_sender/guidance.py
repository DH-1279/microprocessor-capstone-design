"""Left hand only. Project target direction into the observed hand's 2D axes.

Coordinates must be from the SAME frame, normalized 0..1, with original capture
time. Never feed stabilizer-held landmarks as a new observation. No depth or
grasp claims are made from image-plane proximity.
"""

import math
from dataclasses import dataclass

from .protocol import Channel, Cue, Mode, Zone


@dataclass(frozen=True)
class Guidance:
    cue: Cue
    captured_at: float
    facing: str
    distance: float  # image-plane gap / observed palm width
    side: float  # positive toward index/thumb side
    forward: float  # positive toward fingertips


class Mapper:
    def __init__(
        self,
        mode=Mode.RATE,
        duties=(180,) * 5,
        rate_channel=Channel.IN,
        max_age=0.25,
        near=0.35,
        middle=1.5,
        hysteresis=0.10,
        stable_frames=3,
        mirrored=False,
        multi_rate=False,
    ):
        Cue(mode, Zone.FAR, rate_channel if mode == Mode.RATE else Channel.TH, duties).validate()
        if (
            not all(math.isfinite(n) for n in (max_age, near, middle, hysteresis))
            or max_age <= 0
            or not 0 < hysteresis < near < middle - hysteresis
            or type(stable_frames) is not int
            or stable_frames < 1
        ):
            raise ValueError("invalid guidance thresholds")
        self.mode, self.duties, self.rate_channel = mode, tuple(duties), rate_channel
        self.max_age, self.near, self.middle, self.hysteresis = (
            max_age,
            near,
            middle,
            hysteresis,
        )
        self.stable_frames, self.mirrored = stable_frames, mirrored
        self.multi_rate = multi_rate
        self.last_capture = -math.inf
        self.reset()

    def reset(self):
        self.zone = None
        self.channel = None
        self.pending = None
        self.count = 0
        self.confirmed_cue = None
        self.finger_zones = [None] * 5

    def _zone(self, d):
        n, m, h = self.near, self.middle, self.hysteresis
        if self.zone == Zone.NEAR and d <= n + h:
            return Zone.NEAR
        if self.zone == Zone.FAR and d >= m - h:
            return Zone.FAR
        if self.zone == Zone.MIDDLE:
            if d < n - h:
                return Zone.NEAR
            if d > m + h:
                return Zone.FAR
            return Zone.MIDDLE
        return Zone.NEAR if d <= n else Zone.MIDDLE if d <= m else Zone.FAR

    def update(self, landmarks, box, captured_at, now, size, *, observed=True):
        """Return Guidance, or None (STOP). Duplicate frames never win a vote.

        landmarks: 21 objects with x/y attributes, or 21 (x,y) pairs.
        size: actual (width,height), essential for correct non-square geometry.
        mirrored: set True only when inference itself uses a mirrored image.
        """
        if (
            not observed
            or landmarks is None
            or box is None
            or not math.isfinite(captured_at)
            or not math.isfinite(now)
            or not 0 <= now - captured_at <= self.max_age
            or captured_at <= self.last_capture
        ):
            self.reset()
            return None
        if captured_at - self.last_capture > self.max_age:
            self.reset()
        self.last_capture = captured_at
        try:
            width, height = size
            if (
                not math.isfinite(width)
                or not math.isfinite(height)
                or min(width, height) <= 0
                or len(landmarks) != 21
                or len(box) != 4
            ):
                raise ValueError("bad dimensions")
            points = [(float(p.x), float(p.y)) if hasattr(p, "x") else (float(p[0]), float(p[1])) for p in landmarks]
            box = tuple(float(n) for n in box)
            if any(not math.isfinite(n) or not 0 <= n <= 1 for pair in points for n in pair):
                raise ValueError("invalid landmarks")
            if any(not math.isfinite(n) or not 0 <= n <= 1 for n in box) or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("invalid box")
            if self.mirrored:
                points = [(1 - x, y) for x, y in points]
                box = (1 - box[2], box[1], 1 - box[0], box[3])
            # KMJ's wrist/index-base/pinky-base signed triangle; left-hand sign.
            w, i, p = points[0], points[5], points[17]
            cross = (i[0] - w[0]) * (p[1] - w[1]) - (i[1] - w[1]) * (p[0] - w[0])
            if abs(cross) < 0.004:
                raise ValueError("edge-on / uncertain facing")
            facing = "back" if cross < 0 else "palm"
            aspect = width / height
            points = [(x * aspect, y) for x, y in points]
            x1, y1, x2, y2 = box[0] * aspect, box[1], box[2] * aspect, box[3]
            # Side points anatomically toward index/thumb, independently of flip.
            ux, uy = points[5][0] - points[17][0], points[5][1] - points[17][1]
            fx, fy = points[9][0] - points[0][0], points[9][1] - points[0][1]
            palm_width, palm_length = math.hypot(ux, uy), math.hypot(fx, fy)
            if min(palm_width, palm_length) < 0.04:
                raise ValueError("hand too small")
            ux, uy, fx, fy = (
                ux / palm_width,
                uy / palm_width,
                fx / palm_length,
                fy / palm_length,
            )
            determinant = ux * fy - uy * fx
            if abs(determinant) < 0.25:
                raise ValueError("degenerate hand axes")
            if self.mode == Mode.RATE:
                hx, hy = points[(4, 8, 12, 16, 20)[self.rate_channel]]
            else:
                hx = sum(points[j][0] for j in (0, 5, 9, 13, 17)) / 5
                hy = sum(points[j][1] for j in (0, 5, 9, 13, 17)) / 5
            dx, dy = (x1 + x2) / 2 - hx, (y1 + y2) / 2 - hy
            # Solve a 2x2 basis (not two dots: axes need not be perpendicular).
            side = (dx * fy - dy * fx) / determinant / palm_width
            forward = (ux * dy - uy * dx) / determinant / palm_width
            distance = math.hypot(max(x1 - hx, 0, hx - x2), max(y1 - hy, 0, hy - y2)) / palm_width
            self.zone = self._zone(distance)
            if self.mode == Mode.RATE:
                candidate = self.rate_channel
            elif self.zone == Zone.NEAR:
                candidate = Channel.MI
            else:
                candidate = (
                    (Channel.TH if side > 0 else Channel.PI)
                    if abs(side) >= abs(forward)
                    else (Channel.IN if forward > 0 else Channel.RI)
                )
            if candidate == self.pending:
                self.count += 1
            else:
                self.pending, self.count = candidate, 1
            if self.count < self.stable_frames:
                # Keep the previous confirmed cue during a brief direction vote,
                # only while the CURRENT hand/lock geometry remains valid.
                if self.mode == Mode.DIRECTION and self.confirmed_cue is not None:
                    return Guidance(self.confirmed_cue, captured_at, facing, distance, side, forward)
                return None
            self.channel = candidate
            zones = None
            if self.mode == Mode.RATE and self.multi_rate:
                zones = []
                saved_zone = self.zone
                for finger, index in enumerate((4, 8, 12, 16, 20)):
                    tx, ty = points[index]
                    gap = math.hypot(max(x1 - tx, 0, tx - x2), max(y1 - ty, 0, ty - y2)) / palm_width
                    self.zone = self.finger_zones[finger]
                    zones.append(int(self._zone(gap)))
                self.finger_zones = zones
                self.zone = saved_zone
                zones = tuple(zones)
            self.confirmed_cue = Cue(self.mode, self.zone, candidate, self.duties, zones=zones)
            return Guidance(
                self.confirmed_cue,
                captured_at,
                facing,
                distance,
                side,
                forward,
            )
        except (ValueError, TypeError, IndexError, AttributeError, OverflowError):
            self.reset()
            return None


def fresh_cue(guidance, now, max_age=0.25):
    """Sender-side gate; reusing the latest record must not renew its timestamp."""
    if guidance is None or not math.isfinite(now):
        return None
    age = now - guidance.captured_at
    return guidance.cue if math.isfinite(age) and 0 <= age <= max_age else None
