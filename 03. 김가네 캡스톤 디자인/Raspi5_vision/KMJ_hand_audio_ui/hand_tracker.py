"""Hand landmark point and handedness. Owned by KMJ per the 2026-10-12 plan.

Takes the shared RGB frame and returns models.Hand or None. The camera is
opened once by the common pipeline; nothing here opens a capture device.

    frame -> HandTracker.detect() -> Hand | None -> controller

Two layers:

  hand_from_result  MediaPipe landmarks to a representative point and a
                    handedness decision. Ambiguous frames return None.
  HandStabilizer    smooths the two failures measured on 2026-10-07.

Measured with a laptop webcam, hand_landmarker.task, cotton gloves, right hand:

    back of hand / palm   100% for every glove colour
    hand held upright     bare 99.6%, pink 32.8%, white 36.2%, black 0.0%
    handedness            flips between Left and Right while a gloved hand is
                          fully open; a slight curl reports Right correctly

vision.hand is Right, so a flipped frame is dropped and guidance stops even
though the landmarks were fine. Voting over recent frames keeps one answer.

Reaching for an object tilts the hand upright, which is where gloves fail.
Repeating the last measured point for a short gap keeps the controller in
guiding instead of flapping into hand_lost. The hold is bounded and never
invents a hand: it only repeats a point that was measured.

MediaPipe's handedness score is a left/right classification score. It is not
landmark accuracy and not a detection probability.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

# Wrist plus the four finger base joints. Their mean is the point the
# controller uses for direction and proximity.
POINT_LANDMARKS = (0, 5, 9, 13, 17)
LANDMARK_COUNT = 21

DEFAULTS = {
    "vote_frames": 5,         # frames kept for the handedness vote
    "vote_margin": 2,         # how far ahead the winner must be to switch
    "hold_ms": 400,           # how long a measured point may be repeated
    "min_score": 0.0,         # extra gate on vision.hand_score_threshold
}


def stability_config(values: dict | None = None) -> dict:
    """Validate before the camera starts, like the other config helpers."""
    config = {**DEFAULTS, **(values or {})}
    for key in ("vote_frames", "vote_margin"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"hand_stability.{key} must be a positive integer")
    if config["vote_margin"] > config["vote_frames"]:
        raise ValueError("vote_margin cannot exceed vote_frames")
    for key in ("hold_ms", "min_score"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"hand_stability.{key} must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"hand_stability.{key} must be zero or more")
    if config["hold_ms"] > 2000:
        raise ValueError("hand_stability.hold_ms must not exceed 2000")
    return config


def hand_from_result(result: Any, selected: str, threshold: float):
    """Reject ambiguous hands; handedness score is not landmark confidence."""
    from haptic_mvp.models import Hand

    candidates = []
    if len(result.hand_landmarks) != len(result.handedness):
        raise ValueError("MediaPipe landmark/handedness counts differ")
    for landmarks, categories in zip(result.hand_landmarks, result.handedness):
        if len(landmarks) != LANDMARK_COUNT or not categories:
            continue
        category = categories[0]
        handedness = category.category_name
        score = float(category.score)
        if (handedness not in {"Left", "Right"} or not math.isfinite(score)
                or not 0 <= score <= 1 or score < threshold):
            continue
        if selected != "Any" and selected != handedness:
            continue
        points = [(float(landmarks[i].x), float(landmarks[i].y))
                  for i in POINT_LANDMARKS]
        if any(not math.isfinite(value) or not 0 <= value <= 1
               for point in points for value in point):
            continue
        point = (sum(p[0] for p in points) / len(points),
                 sum(p[1] for p in points) / len(points))
        candidates.append(Hand(point=point, score=score, handedness=handedness))
    return candidates[0] if len(candidates) == 1 else None


@dataclass(frozen=True)
class Stats:
    """What the stabilizer did, for the inspection panel and measurements."""

    frames: int = 0
    tracked: int = 0
    held: int = 0
    flips_absorbed: int = 0
    switches: int = 0

    @property
    def hold_ratio(self) -> float:
        return self.held / self.frames if self.frames else 0.0


class HandStabilizer:
    """
    Feed every frame's hand, take back a steadier one.

    Call update() once per frame, including frames where the tracker returned
    None, so the hold can expire on time.
    """

    def __init__(self, config: dict | None = None, selected: str = "Right"):
        self.config = stability_config(config)
        self.selected = selected
        self.votes: deque[str] = deque(maxlen=self.config["vote_frames"])
        self.handedness: str | None = None
        self.last_hand = None
        self.last_seen: float | None = None
        self.frames = 0
        self.tracked = 0
        self.held = 0
        self.flips_absorbed = 0
        self.switches = 0

    def _vote(self, reading: str) -> str:
        """Decide from recent frames, not just this one."""
        self.votes.append(reading)
        if self.handedness is None:
            self.handedness = reading
            return self.handedness
        current = sum(1 for vote in self.votes if vote == self.handedness)
        if len(self.votes) - current - current >= self.config["vote_margin"]:
            self.handedness = reading
            self.switches += 1
            self.votes.clear()
            self.votes.append(reading)
        elif reading != self.handedness:
            self.flips_absorbed += 1
        return self.handedness

    def update(self, hand, now: float):
        """Return the hand to guide with, or None. Keeps the input type."""
        self.frames += 1
        if hand is not None and hand.score >= self.config["min_score"]:
            voted = self._vote(hand.handedness)
            if self.selected != "Any" and voted != self.selected:
                return self._maybe_hold(now)
            steady = (hand if voted == hand.handedness
                      else replace(hand, handedness=voted))
            self.last_hand = steady
            self.last_seen = now
            self.tracked += 1
            return steady
        return self._maybe_hold(now)

    def _maybe_hold(self, now: float):
        """Repeat the last measured point while the gap is short enough."""
        if self.last_hand is None or self.last_seen is None:
            return None
        if (now - self.last_seen) * 1000 > self.config["hold_ms"]:
            self.last_hand = None
            self.last_seen = None
            self.votes.clear()
            return None
        self.held += 1
        return self.last_hand

    def reset(self) -> None:
        """Forget the current hand. Call on stop or target change."""
        self.votes.clear()
        self.handedness = None
        self.last_hand = None
        self.last_seen = None

    def stats(self) -> Stats:
        return Stats(self.frames, self.tracked, self.held,
                     self.flips_absorbed, self.switches)


class HandTracker:
    """
    MediaPipe Tasks Hand Landmarker on the shared RGB frame.

    Replaces vision.MediaPipeHandTracker. Same config keys, same contract,
    with the measured smoothing applied before the hand leaves this module.

        with HandTracker(config) as tracker:
            hand = tracker.detect(rgb, captured_at)
    """

    def __init__(self, config: dict):
        model = Path(str(config.get("hand_model", "")))
        if model.suffix.lower() != ".task" or not model.is_file():
            raise ValueError(
                f"Local .task model is missing: {model}; see VISION_SETUP.md")
        self.model_path = str(model.resolve())
        self.selected = config.get("hand", "Right")
        self.threshold = float(config.get("hand_score_threshold", 0.6))
        self.stabilizer = HandStabilizer(config.get("hand_stability"),
                                         self.selected)
        self.last_timestamp = -1
        self.last_frame_id: int | None = None
        self.last_result = None

    def __enter__(self):
        import mediapipe as mp

        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self.threshold,
            min_hand_presence_confidence=self.threshold,
            min_tracking_confidence=self.threshold,
        )
        self.tracker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        return self

    def detect(self, rgb: Any, captured_at: float, frame_id: int | None = None):
        """
        One frame in, one Hand or None out.

        frame_id identifies which frame this result belongs to. It is optional
        so the existing two-argument call keeps working, but passing it lets
        the module refuse a frame it has already answered. Parallel inference
        must not mix results from different frames.
        """
        if frame_id is not None:
            if self.last_frame_id is not None and frame_id <= self.last_frame_id:
                raise ValueError(
                    f"frame_id must increase: got {frame_id} after "
                    f"{self.last_frame_id}")
            self.last_frame_id = frame_id

        # VIDEO timestamps must strictly increase even with sub-ms captures.
        timestamp = max(self.last_timestamp + 1, int(captured_at * 1000))
        self.last_timestamp = timestamp
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.tracker.detect_for_video(image, timestamp)
        raw = hand_from_result(result, self.selected, self.threshold)
        self.last_result = self.stabilizer.update(raw, captured_at)
        return self.last_result

    def last(self):
        """The Hand this tracker last returned, and for which frame."""
        return self.last_frame_id, self.last_result

    def stats(self) -> Stats:
        return self.stabilizer.stats()

    def reset(self) -> None:
        self.stabilizer.reset()
        self.last_result = None

    def __exit__(self, *args):
        self.tracker.close()
        return False
