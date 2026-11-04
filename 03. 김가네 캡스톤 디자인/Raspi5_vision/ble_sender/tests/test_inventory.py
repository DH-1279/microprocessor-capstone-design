"""Waiting-state inventory survives missed detections without inventing control data."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from KHS_KMJ_vision.inventory import ObjectInventory
from ble_sender.integration import detector_arguments
from haptic_mvp.models import Detection
from test_integration import CUP, replay


def labels(items):
    return tuple(item.detection.label for item in items)


class InventoryTests(unittest.TestCase):
    def test_three_of_five_then_three_second_hold_and_expiry(self):
        inventory = ObjectInventory()
        for i, detections in enumerate(((CUP,), (), (CUP,), (), (CUP,))):
            now = 10 + i / 10
            items = inventory.update(detections, now, now)
            self.assertEqual(labels(items), ("cup",) if i == 4 else ())
        self.assertEqual(labels(inventory.update((), 10.5, 10.5)), ("cup",))
        self.assertEqual(inventory.snapshot(13.39)[0].seen_at, 10.4)
        self.assertEqual(inventory.snapshot(13.41), ())

    def test_duplicate_boxes_and_duplicate_stale_future_frames_cannot_win_votes(self):
        inventory = ObjectInventory()
        inventory.update((CUP, CUP, CUP), 1, 1)
        self.assertEqual(inventory.update((CUP,), 1, 1.01), ())
        self.assertEqual(inventory.update((CUP,), 0.9, 1.3), ())
        self.assertEqual(inventory.update((CUP,), 2, 1.1), ())
        self.assertEqual(inventory.update((CUP,), 1.1, 1.1), ())
        self.assertEqual(labels(inventory.update((CUP,), 1.2, 1.2)), ("cup",))

    def test_long_gap_clears_unconfirmed_votes_and_order_is_stable(self):
        inventory = ObjectInventory()
        mouse = Detection("mouse", (0.5, 0.5, 0.7, 0.7), 0.8)
        for now in (1, 1.1):
            inventory.update((mouse, CUP), now, now)
        self.assertEqual(inventory.update((CUP, mouse), 2, 2), ())
        inventory.update((mouse, CUP), 2.1, 2.1)
        self.assertEqual(labels(inventory.update((CUP, mouse), 2.2, 2.2)), ("cup", "mouse"))
        self.assertEqual(labels(inventory.update((mouse,), 2.3, 2.3)), ("cup", "mouse"))

    def test_invalid_cli_settings_are_rejected(self):
        with redirect_stderr(io.StringIO()):
            for args in (["--list-hits", "6"], ["--list-window", "0"],
                         ["--list-hold", "nan"], ["--selection-wait", "0"]):
                with self.assertRaises(SystemExit):
                    detector_arguments(args)


class InventoryLoopTests(unittest.TestCase):
    def test_waiting_list_survives_hand_occlusion_then_selects_only_fresh_box(self):
        moved = Detection("cup", (0.6, 0.2, 0.7, 0.3), 0.9)
        rows = [{}, {}, {},
                {"text": "캡스톤", "objects": (), "valid": False},
                {"text": "컵", "objects": (), "valid": False},
                {"objects": (moved,)}]
        result = replay(rows, list_hits=3)
        self.assertEqual(result.code, 0, result.log)
        self.assertEqual(result.states, ["WAIT_WAKE"] * 3 + ["WAIT_TARGET"] * 2 + ["SEARCHING"])
        self.assertTrue(any("현재 책상위엔 컵이 있습니다" in text for text in result.spoken))
        self.assertTrue(any("[LIST HOLD] cup" in text for text in result.overlays))
        active = [item for item in result.bridge.sent if item is not None]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["frame_id"], 6)
        self.assertEqual(active[0]["target_box"], moved.box)
        self.assertEqual(active[0]["target_seen_at"], active[0]["captured_at"])

    def test_expired_list_not_announced_and_one_frame_false_positive_not_listed(self):
        expired = replay([{}, {}, {}, {"text": "캡스톤", "objects": (), "at": 4.3}], list_hits=3)
        self.assertEqual(expired.code, 0, expired.log)
        self.assertTrue(any("인식된 물건이 없습니다" in text for text in expired.spoken))
        false_positive = replay([{}, {"text": "캡스톤", "objects": ()}], list_hits=3)
        self.assertTrue(any("인식된 물건이 없습니다" in text for text in false_positive.spoken))

    def test_selection_timeout_or_cancel_never_starts_old_box_guidance(self):
        initial = [{}, {}, {}, {"text": "캡스톤", "objects": ()}, {"text": "컵", "objects": ()}]
        expired = replay(initial + [{"objects": (), "at": 3.5}, {"at": 3.6}], list_hits=3)
        self.assertEqual(expired.code, 0, expired.log)
        self.assertEqual(expired.states[-1], "WAIT_TARGET")
        self.assertFalse(any(item is not None for item in expired.bridge.sent))
        self.assertEqual(sum("위치를 확인하지 못했습니다" in s for s in expired.spoken), 1)
        cancelled = replay(initial + [{"text": "취소", "objects": ()}, {}], list_hits=3)
        self.assertEqual(cancelled.states[-1], "WAIT_WAKE")
        self.assertFalse(any(item is not None for item in cancelled.bridge.sent))

    def test_visible_selection_waits_for_confirmation_without_changing_acquisition(self):
        result = replay([{"text": "캡스톤"}, {"text": "컵"}, {}], list_hits=3)
        self.assertEqual(result.code, 0, result.log)
        self.assertEqual(result.states, ["WAIT_TARGET", "WAIT_TARGET", "SEARCHING"])
        active = [item for item in result.bridge.sent if item is not None]
        self.assertEqual([item["frame_id"] for item in active], [3])


if __name__ == "__main__":
    unittest.main()
