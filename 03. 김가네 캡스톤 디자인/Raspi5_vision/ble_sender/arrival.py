"""Palm/locked-target center overlap. This is an image-plane acquisition rule."""

import math


def valid_palm_observation(landmarks, box, captured_at, now, size, max_age):
    """Palm overlap does not require a reliable palm/back direction basis."""
    try:
        if not (math.isfinite(captured_at) and math.isfinite(now) and 0 <= now - captured_at <= max_age):
            return False
        if landmarks is None or box is None or len(landmarks) != 21 or len(box) != 4 or len(size) != 2:
            return False
        points = [(p.x, p.y) if hasattr(p, "x") else p[:2] for p in landmarks]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for pair in points for v in pair):
            return False
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in box):
            return False
        if not box[0] < box[2] or not box[1] < box[3] or any(not math.isfinite(v) or v <= 0 for v in size):
            return False
        # Reject collapsed landmarks, but allow a hand seen from its side.
        return max(math.hypot(points[i][0] - points[0][0], points[i][1] - points[0][1])
                   for i in (5, 9, 13, 17)) > 0.005
    except (TypeError, ValueError, IndexError, AttributeError):
        return False

def palm_center(landmarks):
    points = [landmarks[i] for i in (0, 5, 9, 13, 17)]
    pairs = [(p.x, p.y) if hasattr(p, "x") else p[:2] for p in points]
    return sum(p[0] for p in pairs) / 5, sum(p[1] for p in pairs) / 5


class ArrivalGate:
    def __init__(self, threshold=50.0, frames=5, max_age=0.35):
        if not math.isfinite(threshold) or threshold <= 0 or type(frames) is not int or frames < 1:
            raise ValueError("invalid acquisition threshold/frames")
        self.threshold, self.frames, self.max_age = threshold, frames, max_age
        self.reset()

    def reset(self):
        self.target_id = None
        self.count = 0
        self.distance = None
        self.last_capture = None
        self.done = False

    def update(self, target_id, landmarks, box, captured_at, now, size, *, target_seen_at=None):
        if target_id != self.target_id:
            self.reset()
            self.target_id = target_id
        seen = captured_at if target_seen_at is None else target_seen_at
        lock_valid = (type(seen) in (float, int) and math.isfinite(seen)
                      and 0 <= captured_at - seen and 0 <= now - seen)
        valid = lock_valid and valid_palm_observation(landmarks, box, captured_at, now, size, self.max_age)
        if not valid or (self.last_capture is not None and captured_at <= self.last_capture):
            self.count = 0
            self.distance = None
            self.last_capture = None
            return False
        if self.last_capture is not None and captured_at - self.last_capture > self.max_age:
            self.count = 0
        self.last_capture = captured_at
        if self.done:
            return False
        px, py = palm_center(landmarks)
        tx, ty = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        self.distance = math.hypot((tx - px) * size[0], (ty - py) * size[1])
        self.count = self.count + 1 if self.distance <= self.threshold else 0
        if self.count >= self.frames:
            self.done = True
            return True
        return False
