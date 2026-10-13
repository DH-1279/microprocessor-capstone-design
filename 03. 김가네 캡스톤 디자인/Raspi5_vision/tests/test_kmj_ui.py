"""Status page contracts tested without a browser, web server or camera."""

from dataclasses import dataclass, field
import time
import unittest

from KMJ_hand_audio_ui.ui.server import Bridge, allowed_actions

TARGETS = {
    "cup": {"name": "컵", "aliases": ["컵", "머그", "cup"]},
    "cell phone": {"name": "휴대전화", "aliases": ["휴대폰", "핸드폰"]},
}


@dataclass
class FakeDecision:
    """Same fields the controller returns, without importing models."""

    levels: tuple = (0, 0, 0, 0, 0)
    state: str = "scanning"
    reason: str = "idle"
    target: str | None = None
    speech: str | None = None
    direction: str | None = None
    proximity: str | None = None


class Harness:
    def __init__(self, decision=None):
        self.decision = decision or FakeDecision()
        self.sent: list = []
        self.bridge = Bridge(lambda: (self.decision, time.monotonic()),
                             self.sent.append, TARGETS)

    def set(self, **changes):
        self.decision = FakeDecision(**{**self.decision.__dict__, **changes})


class ActionTests(unittest.TestCase):
    def test_stop_is_offered_in_every_state(self):
        for state, reason in (("scanning", "idle"), ("guiding", "moving"),
                              ("guiding", "near"), ("paused", "hand_lost"),
                              ("paused", "target_lost"), ("stopped", "user_stop"),
                              ("guiding", "an-unlisted-reason")):
            with self.subTest(state=state, reason=reason):
                self.assertIn("stop", allowed_actions(state, reason))

    def test_target_selection_is_hidden_while_guiding(self):
        self.assertNotIn("select_target", allowed_actions("guiding", "moving"))
        self.assertIn("select_target", allowed_actions("scanning", "idle"))
        self.assertIn("select_target", allowed_actions("paused", "target_lost"))

    def test_done_is_offered_only_when_the_image_says_near(self):
        self.assertIn("done", allowed_actions("guiding", "near"))
        self.assertNotIn("done", allowed_actions("guiding", "moving"))


class SnapshotTests(unittest.TestCase):
    def test_snapshot_mirrors_the_decision_the_glove_receives(self):
        harness = Harness()
        harness.set(state="guiding", reason="moving", target="cup",
                    levels=(0, 120, 0, 0, 0), direction="right",
                    proximity="middle")
        snapshot = harness.bridge.snapshot()
        self.assertEqual(snapshot["state"], "guiding")
        self.assertEqual(snapshot["target_text"], "컵")
        self.assertEqual(snapshot["output_requested"], [0, 120, 0, 0, 0])
        self.assertEqual(snapshot["direction"], "right")

    def test_revision_changes_only_when_the_decision_changes(self):
        harness = Harness()
        first = harness.bridge.snapshot()["revision"]
        self.assertEqual(harness.bridge.snapshot()["revision"], first)
        harness.set(state="guiding", reason="moving", levels=(90, 0, 0, 0, 0))
        self.assertNotEqual(harness.bridge.snapshot()["revision"], first)

    def test_unknown_reason_still_renders_without_inventing_text(self):
        harness = Harness()
        harness.set(reason="a-new-reason")
        snapshot = harness.bridge.snapshot()
        self.assertEqual(snapshot["reason"], "a-new-reason")
        self.assertEqual(snapshot["message"], "")

    def test_objects_come_from_the_configured_targets(self):
        names = [item["name"] for item in Harness().bridge.snapshot()["objects"]]
        self.assertEqual(names, ["컵", "휴대전화"])


class CommandTests(unittest.TestCase):
    def test_button_reaches_the_controller_as_a_spoken_alias(self):
        harness = Harness()
        revision = harness.bridge.snapshot()["revision"]
        result = harness.bridge.handle({"intent": "select_cup",
                                        "revision": revision,
                                        "command_id": "a"})
        self.assertTrue(result["accepted"])
        self.assertEqual(harness.sent, ["컵"])

    def test_label_with_a_space_is_sent_as_its_alias(self):
        harness = Harness()
        harness.bridge.handle({"intent": "select_cell_phone",
                               "revision": harness.bridge.snapshot()["revision"],
                               "command_id": "b"})
        self.assertEqual(harness.sent, ["휴대폰"])

    def test_repeated_command_id_is_sent_once(self):
        harness = Harness()
        revision = harness.bridge.snapshot()["revision"]
        for _ in range(3):
            harness.bridge.handle({"intent": "stop", "revision": revision,
                                   "command_id": "same"})
        self.assertEqual(harness.sent, ["정지"])

    def test_a_late_start_cannot_restart_guidance(self):
        harness = Harness()
        stale = harness.bridge.snapshot()["revision"]
        harness.set(state="stopped", reason="user_stop")
        result = harness.bridge.handle({"intent": "select_cup",
                                        "revision": stale, "command_id": "c"})
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "stale_revision")
        self.assertEqual(harness.sent, [])

    def test_stop_is_never_refused_for_arriving_late(self):
        harness = Harness()
        harness.set(state="guiding", reason="moving", target="cup")
        result = harness.bridge.handle({"intent": "stop", "revision": -1,
                                        "command_id": "d"})
        self.assertTrue(result["accepted"])
        self.assertEqual(harness.sent, ["정지"])

    def test_unsupported_intent_is_refused_with_a_reason(self):
        harness = Harness()
        for intent, reason in (("", "empty_intent"),
                               ("select_laptop", "unknown_button"),
                               ("책상 정리했어", "unknown_word")):
            with self.subTest(intent=intent):
                result = harness.bridge.handle({"intent": intent})
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], reason)
        self.assertEqual(harness.sent, [])

    def test_page_never_writes_into_the_decision(self):
        harness = Harness()
        before = harness.decision
        harness.bridge.handle({"intent": "stop"})
        self.assertIs(harness.decision, before)


if __name__ == "__main__":
    unittest.main()
