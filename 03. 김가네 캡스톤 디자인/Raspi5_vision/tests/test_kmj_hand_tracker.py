"""Hand point, handedness and smoothing, without a camera or MediaPipe.

Selection cases keep the behaviour the shared pipeline already relied on.
Smoothing cases come from the 2026-10-07 glove measurement: handedness flips on
a fully open gloved hand, and short dropouts when the hand tilts upright.
"""

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

from KMJ_hand_audio_ui.hand_tracker import (
    HandStabilizer, hand_from_result, stability_config,
)


def mediapipe_result(*names, score=0.9, landmarks=21, x=0.4, y=0.6):
    """Minimal stand-in for a HandLandmarkerResult."""
    return SimpleNamespace(
        hand_landmarks=[[SimpleNamespace(x=x, y=y) for _ in range(landmarks)]
                        for _ in names],
        handedness=[[SimpleNamespace(category_name=name, score=score)]
                    for name in names])


class SelectionTests(unittest.TestCase):
    """Which frames may guide at all."""

    def test_the_requested_hand_is_selected(self):
        hand = hand_from_result(mediapipe_result("Left", "Right"), "Right", 0.6)
        self.assertEqual(hand.point, (0.4, 0.6))
        self.assertEqual(hand.handedness, "Right")

    def test_two_hands_are_ambiguous(self):
        for names, selected in ((("Left", "Right"), "Any"),
                                (("Right", "Right"), "Right")):
            with self.subTest(names=names, selected=selected):
                self.assertIsNone(
                    hand_from_result(mediapipe_result(*names), selected, 0.6))

    def test_wrong_hand_or_no_hand_does_not_guide(self):
        self.assertIsNone(hand_from_result(mediapipe_result("Left"), "Right", 0.6))
        self.assertIsNone(hand_from_result(mediapipe_result(), "Any", 0.6))

    def test_low_handedness_score_does_not_guide(self):
        result = mediapipe_result("Right", score=0.4)
        self.assertIsNone(hand_from_result(result, "Right", 0.6))

    def test_invalid_landmarks_do_not_guide(self):
        result = mediapipe_result("Right")
        result.hand_landmarks[0][5].x = float("nan")
        self.assertIsNone(hand_from_result(result, "Right", 0.6))
        result = mediapipe_result("Right")
        result.hand_landmarks[0][9].y = 1.5
        self.assertIsNone(hand_from_result(result, "Right", 0.6))

    def test_incomplete_landmark_set_is_skipped(self):
        self.assertIsNone(
            hand_from_result(mediapipe_result("Right", landmarks=20), "Right", 0.6))

    def test_mismatched_counts_are_an_explicit_failure(self):
        result = mediapipe_result("Right")
        result.handedness = []
        with self.assertRaisesRegex(ValueError, "counts differ"):
            hand_from_result(result, "Right", 0.6)

    def test_point_is_the_mean_of_the_wrist_and_finger_bases(self):
        result = mediapipe_result("Right")
        for index, value in zip((0, 5, 9, 13, 17), (0.1, 0.2, 0.3, 0.4, 0.5)):
            result.hand_landmarks[0][index] = SimpleNamespace(x=value, y=value)
        hand = hand_from_result(result, "Right", 0.6)
        self.assertAlmostEqual(hand.point[0], 0.3)
        self.assertAlmostEqual(hand.point[1], 0.3)


@dataclass(frozen=True)
class FakeHand:
    """Same fields as models.Hand."""

    point: tuple = (0.5, 0.5)
    score: float = 0.9
    handedness: str = "Right"


def feed(stabilizer, readings, start=0.0, step=0.05):
    """Play a sequence of per-frame readings and collect the outputs."""
    return [stabilizer.update(hand, start + i * step)
            for i, hand in enumerate(readings)]


class ConfigurationTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        self.assertEqual(stability_config()["vote_frames"], 5)

    def test_invalid_settings_are_rejected_before_the_camera_opens(self):
        for settings in ({"vote_frames": 0}, {"vote_frames": 2.5},
                         {"vote_margin": 99}, {"hold_ms": -1},
                         {"hold_ms": 5000}, {"vote_frames": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                stability_config(settings)


class HandednessTests(unittest.TestCase):
    def test_a_single_flipped_frame_does_not_drop_the_hand(self):
        stabilizer = HandStabilizer(selected="Right")
        readings = [FakeHand()] * 4 + [FakeHand(handedness="Left")] + [FakeHand()] * 3
        results = feed(stabilizer, readings)
        self.assertTrue(all(result is not None for result in results))
        self.assertEqual(stabilizer.stats().flips_absorbed, 1)
        self.assertEqual(stabilizer.stats().switches, 0)

    def test_alternating_frames_still_report_one_hand(self):
        # The gloved open hand alternated Left/Right between frames.
        stabilizer = HandStabilizer(selected="Right")
        readings = [FakeHand(handedness="Right" if i % 2 == 0 else "Left")
                    for i in range(12)]
        results = feed(stabilizer, readings)
        self.assertTrue(all(result is not None for result in results))
        self.assertTrue(all(r.handedness == "Right" for r in results))

    def test_a_sustained_change_is_followed(self):
        stabilizer = HandStabilizer(selected="Any")
        feed(stabilizer, [FakeHand()] * 5)
        self.assertEqual(stabilizer.handedness, "Right")
        feed(stabilizer, [FakeHand(handedness="Left")] * 6, start=1.0)
        self.assertEqual(stabilizer.handedness, "Left")
        self.assertEqual(stabilizer.stats().switches, 1)

    def test_the_other_hand_is_still_refused(self):
        stabilizer = HandStabilizer(selected="Right")
        results = feed(stabilizer, [FakeHand(handedness="Left")] * 6)
        self.assertTrue(all(result is None for result in results))

    def test_the_reported_handedness_is_the_voted_one(self):
        stabilizer = HandStabilizer(selected="Right")
        feed(stabilizer, [FakeHand()] * 4)
        result = stabilizer.update(FakeHand(handedness="Left"), 1.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.handedness, "Right")


class DropoutTests(unittest.TestCase):
    def test_a_short_gap_keeps_the_last_measured_point(self):
        stabilizer = HandStabilizer({"hold_ms": 400}, selected="Right")
        hand = FakeHand(point=(0.3, 0.7))
        stabilizer.update(hand, 0.0)
        for offset in (0.1, 0.2, 0.3):
            with self.subTest(offset=offset):
                held = stabilizer.update(None, offset)
                self.assertIsNotNone(held)
                self.assertEqual(held.point, (0.3, 0.7))

    def test_a_long_gap_gives_up(self):
        stabilizer = HandStabilizer({"hold_ms": 400}, selected="Right")
        stabilizer.update(FakeHand(), 0.0)
        self.assertIsNone(stabilizer.update(None, 0.5))
        # Once dropped, the hand is not resurrected by a later empty frame.
        self.assertIsNone(stabilizer.update(None, 0.55))

    def test_holding_never_invents_a_hand(self):
        stabilizer = HandStabilizer(selected="Right")
        self.assertIsNone(stabilizer.update(None, 0.0))
        self.assertIsNone(stabilizer.update(None, 0.1))
        self.assertEqual(stabilizer.stats().held, 0)

    def test_a_hold_is_counted_separately_from_a_tracked_frame(self):
        stabilizer = HandStabilizer({"hold_ms": 400}, selected="Right")
        stabilizer.update(FakeHand(), 0.0)
        stabilizer.update(None, 0.1)
        stats = stabilizer.stats()
        self.assertEqual((stats.frames, stats.tracked, stats.held), (2, 1, 1))
        self.assertAlmostEqual(stats.hold_ratio, 0.5)

    def test_zero_hold_disables_smoothing(self):
        stabilizer = HandStabilizer({"hold_ms": 0}, selected="Right")
        stabilizer.update(FakeHand(), 0.0)
        self.assertIsNone(stabilizer.update(None, 0.01))

    def test_reset_clears_the_held_hand(self):
        stabilizer = HandStabilizer({"hold_ms": 400}, selected="Right")
        stabilizer.update(FakeHand(), 0.0)
        stabilizer.reset()
        self.assertIsNone(stabilizer.update(None, 0.05))


class ScoreTests(unittest.TestCase):
    def test_low_score_frames_are_treated_as_missing(self):
        stabilizer = HandStabilizer({"min_score": 0.8, "hold_ms": 0},
                                    selected="Right")
        self.assertIsNone(stabilizer.update(FakeHand(score=0.5), 0.0))
        self.assertIsNotNone(stabilizer.update(FakeHand(score=0.9), 0.1))


class FrameIdTests(unittest.TestCase):
    """A result must belong to one frame, so parallel inference cannot mix."""

    class Tracker:
        """HandTracker's frame bookkeeping, without MediaPipe."""

        def __init__(self):
            from KMJ_hand_audio_ui.hand_tracker import HandTracker
            self.check = HandTracker.detect.__get__(self, type(self))
            self.last_frame_id = None
            self.last_result = None

    def test_repeated_or_older_frame_id_is_refused(self):
        from KMJ_hand_audio_ui.hand_tracker import HandTracker

        tracker = HandTracker.__new__(HandTracker)
        tracker.last_frame_id = 7
        for frame_id in (7, 3):
            with self.subTest(frame_id=frame_id):
                with self.assertRaisesRegex(ValueError, "must increase"):
                    HandTracker.detect(tracker, None, 1.0, frame_id)

    def test_frame_id_is_optional(self):
        from KMJ_hand_audio_ui.hand_tracker import HandTracker

        tracker = HandTracker.__new__(HandTracker)
        tracker.last_frame_id = 7
        # Without a frame_id the check is skipped; the call fails later, on
        # MediaPipe, not on frame bookkeeping.
        with self.assertRaises(AttributeError):
            HandTracker.detect(tracker, None, 1.0)


if __name__ == "__main__":
    unittest.main()
