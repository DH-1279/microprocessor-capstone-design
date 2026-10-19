"""Deterministic target selection and image-plane guidance; no hardware imports."""

from dataclasses import dataclass
import math
import re

from .models import Decision, Detection, Observation


def iou(a: tuple, b: tuple) -> float:
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    detection: Detection
    count: int = 1


class Tracker:
    """Conservative IoU association. Lost or ambiguous instances get new IDs."""

    def __init__(self, minimum_iou: float = 0.2, ambiguity_margin: float = 0.1):
        self.minimum_iou = minimum_iou
        self.ambiguity_margin = ambiguity_margin
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        proposals: dict[int, list[tuple[int, float]]] = {}
        for j, detection in enumerate(detections):
            matches = sorted(
                ((i, iou(track.detection.box, detection.box))
                 for i, track in enumerate(self.tracks)
                 if track.detection.label == detection.label), key=lambda pair: pair[1], reverse=True)
            if matches and matches[0][1] >= self.minimum_iou:
                if len(matches) < 2 or matches[0][1] - matches[1][1] >= self.ambiguity_margin:
                    proposals.setdefault(matches[0][0], []).append((j, matches[0][1]))
        assignments = {candidates[0][0]: i for i, candidates in proposals.items() if len(candidates) == 1}
        updated = []
        for j, detection in enumerate(detections):
            if j in assignments:
                previous = self.tracks[assignments[j]]
                updated.append(Track(previous.id, detection, previous.count + 1))
            else:
                updated.append(Track(self.next_id, detection))
                self.next_id += 1
        self.tracks = updated
        return updated


class Controller:
    def __init__(self, config: dict):
        self.config = config
        self.targets = config["targets"]
        self.tracker = Tracker(config["tracking_iou"], config["ambiguity_margin"])
        self.observation: Observation | None = None
        self.last_frame = -1
        self.last_capture = -1.0
        self.target_id: int | None = None
        self.target_label: str | None = None
        self.pending_label: str | None = None
        self.requires_reselect = False
        self.user_stopped = False
        self.last_announcement = -math.inf
        self.last_reason = ""

    def _name(self, label: str) -> str:
        return self.targets[label]["name"]

    def _clear_target(self) -> None:
        self.target_id = None
        self.target_label = None
        self.pending_label = None
        self.requires_reselect = False

    def _fresh(self, now: float) -> bool:
        return (self.observation is not None
                and -0.05 <= now - self.observation.captured_at <= self.config["max_frame_age_s"])

    def _ingest(self, observation: Observation | None, now: float) -> None:
        if observation is None or observation is self.observation:
            return
        if not observation.valid() or observation.captured_at > now + 0.05:
            self.observation = None
            self.tracker.update([])
            if self.target_id is not None:
                self.requires_reselect = True
            return
        if observation.frame_id <= self.last_frame or observation.captured_at < self.last_capture:
            return  # Repeated frames must not manufacture stability or replace newer data.
        gap = observation.captured_at - self.last_capture
        if self.last_frame >= 0 and gap > self.config["max_frame_age_s"]:
            self.tracker.update([])
            if self.target_id is not None:
                self.requires_reselect = True
        self.last_frame, self.last_capture = observation.frame_id, observation.captured_at
        self.observation = observation
        detections = [d for d in observation.objects
                      if d.label in self.targets and d.score >= self.config["object_score"]]
        tracks = self.tracker.update(detections)
        if self.target_id is not None and not any(t.id == self.target_id for t in tracks):
            self.requires_reselect = True

    def _request(self, text: str, now: float) -> str:
        text = re.sub(r"[\s.,!?]", "", text.lower())
        if any(word in text for word in ("정지", "멈춰", "그만", "중지", "종료")) or text in ("stop", "off"):
            self._clear_target()
            self.user_stopped = True
            return "안내를 중지했습니다."
        if text in ("목록", "다시안내", "뭐가보여", "주변안내", "list"):
            self._clear_target()
            self.user_stopped = False
            self.last_announcement = -math.inf
            return "주변 물체를 다시 확인하겠습니다."
        text = text.replace("책상", "")
        labels = [label for label, item in self.targets.items()
                  if any(alias.lower().replace(" ", "") in text for alias in item["aliases"])]
        side = "left" if "왼쪽" in text else "right" if "오른쪽" in text else None
        if not labels and side and self.pending_label:
            labels = [self.pending_label]
        # Any request to change targets first turns previous guidance off.
        self._clear_target()
        self.user_stopped = False
        if len(labels) != 1:
            return "찾을 물체 하나를 다시 말씀해 주세요."
        label = labels[0]
        candidates = [t for t in self.tracker.tracks
                      if t.detection.label == label and t.count >= self.config["stable_frames"]]
        if not self._fresh(now) or not candidates:
            return f"현재 {self._name(label)}이 안정적으로 보이지 않습니다."
        if len(candidates) > 1:
            if side:
                candidates.sort(key=lambda t: t.detection.box[0] + t.detection.box[2])
                if side == "right":
                    candidates.reverse()
                distance = abs(sum(candidates[0].detection.box[::2]) - sum(candidates[1].detection.box[::2])) / 2
                if distance < self.config["side_separation"]:
                    self.pending_label = label
                    return "좌우 구분이 어렵습니다. 물체를 떨어뜨려 놓아 주세요."
                candidates = candidates[:1]
            else:
                self.pending_label = label
                return f"{self._name(label)}이 여러 개 보입니다. 왼쪽 또는 오른쪽을 말씀해 주세요."
        self.target_id, self.target_label = candidates[0].id, label
        return f"{self._name(label)}을 찾아 안내하겠습니다."

    def _guidance(self, track: Track) -> tuple[tuple, str, str]:
        assert self.observation is not None and self.observation.hand is not None
        hx, hy = self.observation.hand.point
        x1, y1, x2, y2 = track.detection.box
        aspect = self.config["image_aspect"]
        dx, dy = (((x1 + x2) / 2 - hx) * aspect, (y1 + y2) / 2 - hy)
        # Distance to box in image-height units, not centimetres.
        distance = math.hypot(max(x1 - hx, 0, hx - x2) * aspect, max(y1 - hy, 0, hy - y2))
        near, middle = self.config["near_distance"], self.config["middle_distance"]
        proximity = "near" if distance <= near else "middle" if distance <= middle else "far"
        strength = self.config["strengths"][proximity]
        if proximity == "near":
            direction = "near"
        else:
            # Rotate camera axes to user-facing axes, then optional horizontal flip.
            turns = self.config["rotation_deg"] // 90
            for _ in range(turns):
                dx, dy = -dy, dx
            if self.config["flip_x"]:
                dx = -dx
            direction = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("back" if dy > 0 else "forward")
        levels = [0] * 5
        levels[self.config["channel_map"][direction]] = strength
        return tuple(levels), direction, proximity

    def step(self, now: float, observation: Observation | None = None,
             command: str | None = None, audio_busy: bool = False) -> Decision:
        self._ingest(observation, now)
        speech = self._request(command, now) if command is not None else None
        state, reason = "scanning", "idle"
        levels, direction, proximity = (0, 0, 0, 0, 0), None, None
        if self.user_stopped:
            state, reason = "stopped", "user_stop"
        elif not self._fresh(now):
            state, reason = "paused", "stale_frame"
            if self.target_id is not None:
                self.requires_reselect = True
        elif self.target_id is not None:
            if self.requires_reselect:
                state, reason = "paused", "target_lost"
            elif self.observation is None or self.observation.hand is None or self.observation.hand.score < self.config["hand_score"]:
                state, reason = "paused", "hand_lost"
            else:
                track = next(t for t in self.tracker.tracks if t.id == self.target_id)
                levels, direction, proximity = self._guidance(track)
                state, reason = "guiding", "near" if proximity == "near" else "moving"
        if not speech and self.target_id is not None and reason != self.last_reason:
            speech = {"target_lost": "목표를 놓쳤습니다. 물체를 다시 선택해 주세요.",
                      "hand_lost": "손이 보이지 않아 안내를 잠시 멈춥니다.",
                      "stale_frame": "영상이 갱신되지 않아 안내를 멈춥니다.",
                      "near": "목표 가까이에 있습니다."}.get(reason)
        if state == "scanning" and not self.pending_label and not speech and not audio_busy:
            if now - self.last_announcement >= self.config["announce_interval_s"]:
                labels = sorted({t.detection.label for t in self.tracker.tracks
                                 if t.count >= self.config["stable_frames"]})
                if labels:
                    speech = ", ".join(self._name(label) for label in labels) + "이 보입니다."
                    self.last_announcement = now
        self.last_reason = reason
        return Decision(levels, state, reason, self.target_label, speech, direction, proximity)
