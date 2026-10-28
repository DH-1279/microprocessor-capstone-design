"""Image-plane arrival only; never infer a grasp from a lost detection."""

from .guidance import Mapper


class ArrivalGate:
    def __init__(self, dwell=0.4, max_age=0.35, inset=0.1):
        self.dwell, self.max_age, self.inset = dwell, max_age, inset
        self.validator = Mapper(mode=3, stable_frames=1, max_age=max_age)
        self.reset()

    def reset(self):
        self.target_id = None
        self.since = [None] * 5
        self.last_capture = None
        self.done = False
        self.validator.reset()
        self.validator.last_capture = -float("inf")

    def update(self, target_id, landmarks, box, captured_at, now, size):
        if target_id != self.target_id:
            self.reset()
            self.target_id = target_id
        valid = self.validator.update(landmarks, box, captured_at, now, size)
        if valid is None:
            self.since = [None] * 5
            self.last_capture = None
            return False
        if self.last_capture is not None and captured_at - self.last_capture > self.max_age:
            self.since = [None] * 5
        self.last_capture = captured_at
        if self.done:
            return False
        x1, y1, x2, y2 = box
        ix, iy = (x2 - x1) * self.inset, (y2 - y1) * self.inset
        for finger, index in enumerate((4, 8, 12, 16, 20)):
            point = landmarks[index]
            x, y = (point.x, point.y) if hasattr(point, "x") else point[:2]
            inside = x1 + ix <= x <= x2 - ix and y1 + iy <= y <= y2 - iy
            if not inside:
                self.since[finger] = None
            elif self.since[finger] is None:
                self.since[finger] = captured_at
            elif captured_at - self.since[finger] >= self.dwell:
                self.done = True
                return True
        return False
