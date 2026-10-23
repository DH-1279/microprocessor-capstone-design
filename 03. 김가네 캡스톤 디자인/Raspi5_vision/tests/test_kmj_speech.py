"""Command grammar tested without a microphone, Whisper model or audio device."""

import unittest

from KMJ_hand_audio_ui.speech import (
    Intent, button_id, parse, parse_input, refusal_speech, word_distance,
)

TARGETS = {
    "cup": {"name": "컵", "aliases": ["컵", "머그", "cup"]},
    "bottle": {"name": "물병", "aliases": ["물병", "생수", "보틀", "bottle"]},
    "cell phone": {"name": "휴대전화",
                   "aliases": ["휴대폰", "휴대전화", "스마트폰", "핸드폰", "phone"]},
    "book": {"name": "책", "aliases": ["책", "book"]},
    "mouse": {"name": "마우스", "aliases": ["마우스", "mouse"]},
}


class TargetRequestTests(unittest.TestCase):
    def test_aliases_and_names_resolve_to_the_detector_label(self):
        for text, label in (("컵 찾아줘", "cup"), ("머그 찾아줘", "cup"),
                            ("생수 찾아줘", "bottle"), ("핸드폰", "cell phone"),
                            ("휴대전화 어디야", "cell phone"),
                            ("책 찾아", "book"), ("마우스 알려줘", "mouse")):
            with self.subTest(text=text):
                self.assertEqual(parse(text, TARGETS).name, label)

    def test_particles_and_filler_words_do_not_block_a_request(self):
        for text in ("컵을 찾아줘", "책 좀 찾아줘", "그 컵 찾아줘"):
            with self.subTest(text=text):
                self.assertTrue(parse(text, TARGETS))

    def test_missing_spaces_still_resolve_a_short_command(self):
        # whisper-cli returns "컵찾아줘" as often as "컵 찾아줘".
        for text, label in (("컵찾아줘", "cup"), ("물병찾아줘", "bottle"),
                            ("마우스찾아줘", "mouse")):
            with self.subTest(text=text):
                self.assertEqual(parse(text, TARGETS).name, label)

    def test_split_short_command_is_rejoined(self):
        # Measured STT output: "오른쪽" came back as "모른 쪽".
        self.assertEqual(parse("모른 쪽", TARGETS).name, "right")


class MisrecognitionTests(unittest.TestCase):
    """Recorded whisper-cli output from the 2026-10-12 microphone session."""

    def test_single_word_noise_is_recovered(self):
        for text, label in (("컴 찾아줘", "cup"), ("겁 찾아줘", "cup"),
                            ("눈병 찾아줘", "bottle"), ("물농 찾아줘", "bottle"),
                            ("물경 찾아줘", "bottle"), ("휴데폰", "cell phone"),
                            ("마우즈", "mouse"), ("마우스 타자줘", "mouse"),
                            ("전지", "stop"), ("웬쪽", "left")):
            with self.subTest(text=text):
                self.assertEqual(parse(text, TARGETS).name, label)

    def test_recovery_never_crosses_a_word_boundary(self):
        # Substring matching accepted these; whole-word distance must not.
        for text in ("물컵 어디", "마우스패드", "책상", "전화 왔어"):
            with self.subTest(text=text):
                self.assertFalse(parse(text, TARGETS))


class RefusalTests(unittest.TestCase):
    """Spec UX-08: an unsupported utterance must not start guidance."""

    def test_unrelated_speech_is_refused(self):
        for text in ("책상 정리했어", "책상 위에 뭐 있어", "오늘 날씨 어때",
                     "안녕하세요", "내일 수업 몇 시야"):
            with self.subTest(text=text):
                intent = parse(text, TARGETS)
                self.assertFalse(intent)
                self.assertEqual(intent.reason, "unknown_word")

    def test_negation_is_refused_rather_than_guessed(self):
        for text in ("컵 말고 책", "컵 아니라 물병", "책 빼고 컵"):
            with self.subTest(text=text):
                self.assertEqual(parse(text, TARGETS).reason, "negation")

    def test_several_targets_require_one_request(self):
        intent = parse("컵이랑 책 찾아줘", TARGETS)
        self.assertEqual(intent.reason, "multiple_targets")
        self.assertIn("cup", intent.detail)
        self.assertIn("book", intent.detail)

    def test_two_commands_or_a_mixed_command_is_refused(self):
        self.assertEqual(parse("정지 목록", TARGETS).reason, "multiple_commands")
        self.assertEqual(parse("오른쪽 왼쪽", TARGETS).reason, "multiple_directions")
        self.assertEqual(parse("컵 정지", TARGETS).reason, "mixed_command")

    def test_empty_and_request_only_input(self):
        self.assertEqual(parse("", TARGETS).reason, "empty")
        self.assertEqual(parse("   ", TARGETS).reason, "empty")
        self.assertEqual(parse("찾아줘", TARGETS).reason, "no_command")

    def test_every_refusal_has_speech_or_stays_silent(self):
        for text in ("책상 정리했어", "컵 말고 책", "컵이랑 책", "찾아줘", ""):
            with self.subTest(text=text):
                self.assertIsInstance(refusal_speech(parse(text, TARGETS)), str)


class ControlCommandTests(unittest.TestCase):
    def test_control_words(self):
        for text, name in (("정지", "stop"), ("멈춰", "stop"), ("그만", "stop"),
                           ("중지", "stop"), ("stop", "stop"),
                           ("목록", "list"), ("리스트", "list"),
                           ("시작", "start"), ("네", "start"),
                           ("계속", "resume"), ("찾았어", "done"),
                           ("다시", "repeat")):
            with self.subTest(text=text):
                intent = parse(text, TARGETS)
                self.assertEqual((intent.kind, intent.name), ("control", name))

    def test_request_word_is_not_mistaken_for_the_done_command(self):
        # "찾아줘" and "찾았어" differ by two jamo; exact matching must win.
        self.assertEqual(parse("컵 찾아줘", TARGETS).name, "cup")
        self.assertEqual(parse("찾았어", TARGETS).name, "done")

    def test_directions(self):
        self.assertEqual(parse("왼쪽", TARGETS).name, "left")
        self.assertEqual(parse("오른쪽", TARGETS).name, "right")


class StateTests(unittest.TestCase):
    """Only the commands that make sense in a state are accepted."""

    def test_guidance_does_not_accept_a_new_target(self):
        self.assertFalse(parse("컵 찾아줘", TARGETS, "guiding"))
        self.assertEqual(parse("컵 찾아줘", TARGETS, "guiding").reason,
                         "not_allowed_in_state")

    def test_stop_is_accepted_in_every_state(self):
        for state in ("scanning", "selecting", "confirming", "guiding",
                      "paused", "waiting", "near", "stopped"):
            with self.subTest(state=state):
                self.assertEqual(parse("정지", TARGETS, state).name, "stop")

    def test_resume_only_after_the_hand_returns(self):
        self.assertEqual(parse("계속", TARGETS, "waiting").name, "resume")
        self.assertFalse(parse("계속", TARGETS, "scanning"))
        self.assertFalse(parse("시작", TARGETS, "waiting"))

    def test_done_only_when_already_near(self):
        self.assertEqual(parse("찾았어", TARGETS, "near").name, "done")
        self.assertFalse(parse("찾았어", TARGETS, "guiding"))

    def test_side_choice_only_while_selecting(self):
        self.assertEqual(parse("왼쪽", TARGETS, "selecting").name, "left")
        self.assertFalse(parse("왼쪽", TARGETS, "scanning"))

    def test_unknown_state_falls_back_to_accepting_everything(self):
        self.assertTrue(parse("컵 찾아줘", TARGETS, "no-such-state"))


class ButtonTests(unittest.TestCase):
    """UI buttons reach the same parser as speech, so one path is tested."""

    def test_control_buttons_match_their_spoken_command(self):
        for button, name in (("stop", "stop"), ("start", "start"),
                             ("resume", "resume"), ("done", "done"),
                             ("list", "list"), ("repeat", "repeat"),
                             ("choose_left", "left"), ("choose_right", "right")):
            with self.subTest(button=button):
                self.assertEqual(parse_input(button, TARGETS).name, name)

    def test_object_buttons_use_the_configured_labels(self):
        for label in TARGETS:
            with self.subTest(label=label):
                intent = parse_input(button_id(label), TARGETS)
                self.assertEqual((intent.kind, intent.name), ("target", label))

    def test_button_id_survives_labels_containing_spaces(self):
        self.assertEqual(button_id("cell phone"), "select_cell_phone")

    def test_unknown_button_is_refused_not_guessed(self):
        intent = parse_input("select_laptop", TARGETS)
        self.assertFalse(intent)
        self.assertEqual(intent.reason, "unknown_button")

    def test_object_buttons_obey_the_same_state_rules(self):
        self.assertEqual(parse_input("select_cup", TARGETS, "guiding").reason,
                         "not_allowed_in_state")


class ConfigurationTests(unittest.TestCase):
    def test_targets_come_from_configuration_not_a_fixed_table(self):
        targets = {"keyboard": {"name": "키보드", "aliases": ["키보드"]}}
        self.assertEqual(parse("키보드 찾아줘", targets).name, "keyboard")
        self.assertFalse(parse("컵 찾아줘", targets))

    def test_a_label_without_aliases_still_matches_itself(self):
        self.assertEqual(parse("spoon", {"spoon": {}}).name, "spoon")

    def test_result_is_a_frozen_intent_carrying_the_raw_sentence(self):
        intent = parse("컵 찾아줘", TARGETS)
        self.assertIsInstance(intent, Intent)
        self.assertEqual(intent.raw, "컵 찾아줘")
        with self.assertRaises(Exception):
            intent.name = "book"


class DistanceTests(unittest.TestCase):
    def test_identical_and_unrelated_words(self):
        self.assertEqual(word_distance("컵", "컵"), 0)
        self.assertEqual(word_distance("", "컵"), 1)
        self.assertGreater(word_distance("책상", "책"), 0.4)

    def test_one_jamo_difference_stays_close(self):
        self.assertLess(word_distance("컴", "컵"), 0.4)
        self.assertLess(word_distance("전지", "정지"), 0.4)


if __name__ == "__main__":
    unittest.main()
