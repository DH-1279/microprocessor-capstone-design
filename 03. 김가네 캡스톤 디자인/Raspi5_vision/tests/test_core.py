import json
import math
import struct
import tempfile
from pathlib import Path

import pytest

from haptic_mvp.config import load_config, validate
from haptic_mvp.controller import Controller
from haptic_mvp.models import Detection, Hand, Observation
from haptic_mvp.protocol import Command, SimulatedReceiver, Status
from haptic_mvp.runtime import CommandInbox, Latest
from haptic_mvp.simulation import CUP, HAND, SCENARIOS, Harness


@pytest.fixture
def config():
    return load_config(resolve_paths=False)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_closed_loop(config, name):
    assert SCENARIOS[name](config)


def test_single_frame_does_not_become_stable(config):
    controller = Controller(config["controller"])
    frame = Observation(1, 0, (CUP,), HAND)
    for _ in range(20):
        controller.step(0.1, frame)
    assert controller.step(0.1, frame, "컵").stopped
    assert controller.tracker.tracks[0].count == 1


def test_old_frame_cannot_replace_newer(config):
    h = Harness(config)
    h.warm()
    h.step(command="컵")
    before = h.controller.observation
    h.controller.step(h.now, Observation(0, 0, (), None))
    assert h.controller.observation is before


@pytest.mark.parametrize("coordinate", [math.nan, math.inf, -0.1, 1.1])
def test_invalid_hand_stops(config, coordinate):
    h = Harness(config)
    h.warm()
    h.step(command="컵")
    decision = h.step(hand=Hand((coordinate, 0.5), 0.9, "Right"))
    assert decision.stopped


def test_future_frame_stops(config):
    h = Harness(config)
    h.warm()
    h.step(command="컵")
    decision = h.controller.step(h.now, Observation(90, h.now + 100, (CUP,), HAND))
    assert decision.stopped


def test_other_target_request_stops_previous(config):
    h = Harness(config)
    h.warm()
    h.step(command="컵")
    assert h.step(command="물병 찾아줘").stopped
    assert h.controller.target_id is None


def test_desktop_is_not_book(config):
    h = Harness(config)
    book = Detection("book", CUP.box, 0.9)
    h.warm((book,))
    assert h.step((book,), command="책상").stopped


def test_no_cross_class_target_switch(config):
    h = Harness(config)
    h.warm()
    h.step(command="컵")
    bottle = Detection("bottle", CUP.box, 0.95)
    assert h.step((bottle,)).reason == "target_lost"


def test_ambiguous_overlapping_instances_drop_id(config):
    h = Harness(config)
    h.warm()
    h.step(command="컵")
    duplicate = Detection("cup", (0.64, 0.3, 0.84, 0.55), 0.94)
    assert h.step((CUP, duplicate)).reason == "target_lost"


def test_scene_list_does_not_interrupt_speech(config):
    h = Harness(config)
    for n in range(3):
        decision = h.controller.step(n * 0.1, Observation(n, n * 0.1, (CUP,), HAND), audio_busy=True)
        assert decision.speech is None
    decision = h.controller.step(0.3, Observation(3, 0.3, (CUP,), HAND))
    assert "컵" in decision.speech
    assert h.controller.step(0.4, Observation(4, 0.4, (CUP,), HAND)).speech is None


@pytest.mark.parametrize("rotation,flip,expected", [(0, False, "right"), (90, False, "back"),
                                                  (180, False, "left"), (270, False, "forward"),
                                                  (0, True, "left")])
def test_camera_axis_transform(config, rotation, flip, expected):
    config["controller"]["rotation_deg"] = rotation
    config["controller"]["flip_x"] = flip
    h = Harness(config)
    h.warm()
    assert h.step(hand=Hand((0.4, 0.425), 1, "Right"), command="컵").direction == expected


def test_protocol_known_firmware_vector():
    command = Command(4660, 300, (0, 1, 127, 254, 255))
    assert command.encode().hex() == "010034122c0100017ffeff"
    assert Command.decode(command.encode()) == command


@pytest.mark.parametrize("length", [n for n in range(0, 513) if n != 11])
def test_bad_wire_lengths(length):
    with pytest.raises(ValueError):
        Command.decode(bytes(length))


@pytest.mark.parametrize("flag", range(2, 256))
def test_reserved_command_flags(flag):
    with pytest.raises(ValueError):
        Command.decode(struct.pack("<BBHH5B", 1, flag, 0, 100, 0, 0, 0, 0, 0))


def test_replay_stops_and_does_not_refresh_watchdog():
    receiver = SimulatedReceiver()
    receiver.connect()
    packet = Command(65535, 500, (1, 2, 3, 4, 5)).encode()
    assert receiver.receive(packet, 0)
    assert not receiver.receive(packet, 0.2)
    assert not any(receiver.levels) and receiver.invalid
    assert receiver.receive(Command(0, 500, (1, 2, 3, 4, 5)).encode(), 0.3)
    receiver.tick(0.8)
    assert receiver.expired and not any(receiver.levels)


def test_stale_stop_always_stops_without_sequence_rollback():
    receiver = SimulatedReceiver()
    receiver.connect()
    receiver.receive(Command(9, 500, (100, 0, 0, 0, 0)).encode(), 0)
    assert receiver.receive(Command(2, 500, (0, 0, 0, 0, 0), True).encode(), 0.1)
    assert receiver.sequence == 9 and not any(receiver.levels)


@pytest.mark.parametrize("ttl", [0, 99, 1001, 65535])
def test_invalid_ttl(ttl):
    with pytest.raises(ValueError):
        Command.decode(struct.pack("<BBHH5B", 1, 0, 1, ttl, 1, 0, 0, 0, 0))


def test_zero_command_not_expired_and_disconnect_resets():
    receiver = SimulatedReceiver()
    receiver.connect()
    receiver.receive(Command(0, 100, (0, 0, 0, 0, 0)).encode(), 0)
    receiver.tick(0.2)
    assert not receiver.expired
    receiver.receive(b"bad", 0.3)
    receiver.disconnect()
    assert not receiver.invalid and receiver.sequence is None


def test_status_rejects_old_led_firmware():
    with pytest.raises(ValueError):
        Status.decode(b"0")


def test_exact_millisecond_expiry_matches_firmware():
    receiver = SimulatedReceiver()
    receiver.connect()
    receiver.receive(Command(17112, 300, (224, 99, 8, 5, 176)).encode(), 13.220)
    receiver.tick(13.519)
    assert any(receiver.levels)
    receiver.tick(13.520)
    assert receiver.expired and not any(receiver.levels)


@pytest.mark.parametrize("bad", [None, 7, "wrong"])
def test_invalid_message_container_is_rejected(bad):
    assert not Detection("cup", bad, 0.9).valid()
    assert not Hand(bad, 0.9, "Right").valid()
    assert not Observation(0, 0, bad, None).valid()


def test_command_mailbox_prioritizes_stop():
    inbox = CommandInbox()
    inbox.put("컵")
    inbox.put("정지")
    inbox.put("물병")
    assert inbox.take() == "정지"
    assert inbox.take() is None
    inbox.put("물병")
    assert inbox.take() == "물병"


def test_latest_take_and_get():
    mailbox = Latest()
    mailbox.put("a")
    assert mailbox.get() == "a"
    assert mailbox.take() == "a"
    assert mailbox.get() is None


@pytest.mark.parametrize("section,key,value", [
    ("ble", "ttl_ms", 99), ("ble", "ttl_ms", True), ("ble", "send_interval_s", 0.3),
    ("controller", "max_frame_age_s", float("nan")), ("controller", "rotation_deg", 45),
    ("controller", "image_aspect", 1), ("controller", "stable_frames", 0),
    ("controller", "flip_x", "false"), ("vision", "source", "network"),
    ("vision", "mirror", "false"), ("vision", "score_threshold", float("nan")),
    ("vision", "device", None), ("vision", "max_objects", 0),
    ("audio", "whisper_model", None),
])
def test_invalid_settings(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate(config)


def test_config_partial_override_and_unknown_key(config):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps({"controller": {"announce_interval_s": 8}}))
        assert load_config(path)["controller"]["announce_interval_s"] == 8
        path.write_text('{"ble":{"tttl_ms":500}}')
        with pytest.raises(ValueError):
            load_config(path)
