"""Async application lifecycle tests with synthetic vision and local audio fakes."""

import asyncio
import threading
import time

import pytest

from haptic_mvp import audio, runtime, vision
from haptic_mvp.config import load_config
from haptic_mvp.models import Decision, Detection, Hand, Observation
from haptic_mvp.runtime import PendingSpeech


class FakeSpeaker:
    speaking = False
    listening = False

    def __init__(self):
        self.messages = []
        self.accept = True

    def say(self, text):
        if self.accept:
            self.messages.append(text)
        return self.accept


def selected(reason="moving", speech="컵을 찾아 안내하겠습니다."):
    return Decision((70, 0, 0, 0, 0), "guiding", reason, "cup", speech)


@pytest.mark.parametrize("reason", ["near", "hand_lost", "moving"])
def test_ack_survives_valid_same_target_state_changes(reason):
    pending, speaker = PendingSpeech(), FakeSpeaker()
    speaker.listening = True
    pending.update(selected(), 0, 1, "컵 찾아줘")
    assert not pending.deliver(speaker)
    pending.update(selected(reason, "상태 안내"), 0.1, 1)
    speaker.listening = False
    assert pending.deliver(speaker)
    assert speaker.messages == ["컵을 찾아 안내하겠습니다."]
    assert pending.deliver(speaker)
    assert speaker.messages[-1] == "상태 안내"


def test_ack_is_not_expired_while_target_remains_valid_and_audio_is_busy():
    pending, speaker = PendingSpeech(), FakeSpeaker()
    speaker.speaking = True
    pending.update(selected(), 0, 1, "컵")
    pending.update(selected("near", None), 20, 1)
    assert not pending.deliver(speaker)
    speaker.speaking = False
    assert pending.deliver(speaker)


@pytest.mark.parametrize("reason", ["target_lost", "stale_frame", "ble_disconnected", "link_change"])
def test_ack_does_not_survive_invalid_guidance(reason):
    pending, speaker = PendingSpeech(), FakeSpeaker()
    pending.update(selected(), 0, 1, "컵")
    pending.update(Decision(state="paused", reason=reason, target="cup"), 0.1, 1)
    assert not pending.deliver(speaker)


def test_new_command_or_link_epoch_cancels_even_a_same_label_ack():
    pending, speaker = PendingSpeech(), FakeSpeaker()
    pending.update(selected(), 0, 1, "컵")
    pending.update(selected(speech=None), 0.1, 2)
    assert not pending.deliver(speaker)
    pending.update(selected(), 0.2, 2, "컵")
    pending.update(selected(speech=None), 0.3, 2, "다른 명령")
    assert not pending.deliver(speaker)


def test_stop_replaces_pending_ack_and_say_rejection_keeps_stop_response():
    pending, speaker = PendingSpeech(), FakeSpeaker()
    pending.update(selected(), 0, 1, "컵")
    pending.update(Decision(state="stopped", reason="user_stop", speech="중지했습니다"), 0.1, 1, "정지")
    speaker.accept = False
    assert not pending.deliver(speaker)
    speaker.accept = True
    assert pending.deliver(speaker)
    assert speaker.messages == ["중지했습니다"]


def test_old_status_is_discarded_and_queue_never_accumulates_summaries():
    pending, speaker = PendingSpeech(), FakeSpeaker()
    for number in range(100):
        pending.update(Decision(speech=f"목록 {number}"), number / 100, 1)
    assert pending.ack is None
    pending.update(Decision(), 5, 1)
    assert not pending.deliver(speaker)


def setup_runtime(monkeypatch, *, inject_old_command=False, close_error=False, fail_vision=False):
    config = load_config(resolve_paths=False)
    config["audio"]["enabled"] = True
    config["ble"]["dry_run"] = False
    events, holder = [], {}
    vision_warm = threading.Event()
    vision_closed = threading.Event()

    class FakeAudio(FakeSpeaker):
        def __init__(self, settings, on_text):
            super().__init__()
            self.on_text = on_text
            self.error = None
            holder["audio"] = self

        def start(self):
            events.append("audio_start")

        def get_error(self):
            error, self.error = self.error, None
            return error

        def discard_input(self):
            events.append("discard_input")
            if inject_old_command:
                # Mimics a completed STT callback right before discard obtains its lock.
                self.on_text("컵 찾아줘")

        def close(self):
            events.append("audio_close")
            if close_error:
                raise RuntimeError("speaker close failure")

    def run_vision(config, publish, stop):
        try:
            if fail_vision:
                raise RuntimeError("camera disconnected")
            for frame_id in range(10000):
                publish(Observation(frame_id, time.monotonic(),
                                    (Detection("cup", (0.65, 0.3, 0.85, 0.55), 0.95),),
                                    Hand((0.2, 0.6), 0.9, "Right")))
                if frame_id >= 4:
                    vision_warm.set()
                if stop.wait(0.01):
                    break
        finally:
            events.append("vision_close")
            vision_closed.set()

    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(audio, "AudioService", FakeAudio)
    monkeypatch.setattr(vision, "run_vision", run_vision)
    return config, events, holder, vision_warm, vision_closed


async def until(predicate, timeout=1.5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Async runtime condition timed out")
        await asyncio.sleep(0.005)


def test_real_controller_reconnect_requires_fresh_command_and_discards_old_stt(monkeypatch):
    config, events, holder, warm, closed = setup_runtime(monkeypatch, inject_old_command=True)

    async def fake_transport(settings, get_latest, stop, on_connection):
        try:
            on_connection(True)
            await until(warm.is_set)
            await asyncio.sleep(0.05)  # Let controller ingest distinct stability frames.
            holder["audio"].on_text("컵 찾아줘")
            await until(lambda: any(get_latest()[0].levels))
            on_connection(False)
            assert get_latest()[0].stopped
            await asyncio.sleep(0.04)
            on_connection(True)
            await asyncio.sleep(0.06)
            assert get_latest()[0].stopped
            assert get_latest()[0].target is None
            holder["audio"].on_text("컵 찾아줘")
            await until(lambda: any(get_latest()[0].levels))
            stop.set()
        finally:
            events.append("transport_close")

    monkeypatch.setattr(runtime, "transmit_loop", fake_transport)
    asyncio.run(runtime.run(config, duration_s=3))
    assert events.count("discard_input") == 3
    assert "audio_close" in events
    assert closed.is_set()


def test_explicit_stop_clears_callback_racing_with_discard(monkeypatch):
    config, events, holder, warm, closed = setup_runtime(monkeypatch, inject_old_command=True)

    async def fake_transport(settings, get_latest, stop, on_connection):
        on_connection(True)
        await until(warm.is_set)
        await asyncio.sleep(0.05)
        holder["audio"].on_text("컵 찾아줘")
        await until(lambda: any(get_latest()[0].levels))
        holder["audio"].on_text("정지")
        await until(lambda: get_latest()[0].reason == "user_stop")
        await asyncio.sleep(0.06)
        assert get_latest()[0].reason == "user_stop"
        assert get_latest()[0].stopped
        stop.set()

    monkeypatch.setattr(runtime, "transmit_loop", fake_transport)
    asyncio.run(runtime.run(config, duration_s=3))
    assert events.count("discard_input") == 2
    assert closed.is_set()


def test_failed_transport_does_not_skip_audio_and_camera_cleanup(monkeypatch):
    config, events, holder, warm, closed = setup_runtime(monkeypatch)

    async def failed_transport(settings, get_latest, stop, on_connection):
        await until(warm.is_set)
        raise RuntimeError("transport exploded")

    monkeypatch.setattr(runtime, "transmit_loop", failed_transport)
    with pytest.raises(RuntimeError, match="transport exploded"):
        asyncio.run(runtime.run(config, duration_s=3))
    assert "audio_close" in events
    assert closed.is_set()


def test_audio_error_stops_ble_before_audio_shutdown_and_keeps_original_error(monkeypatch):
    config, events, holder, warm, closed = setup_runtime(monkeypatch, close_error=True)

    async def fake_transport(settings, get_latest, stop, on_connection):
        try:
            on_connection(True)
            await until(warm.is_set)
            holder["audio"].error = "microphone unplugged"
            await asyncio.Event().wait()
        finally:
            assert get_latest()[0].stopped
            assert get_latest()[0].reason == "shutdown"
            events.append("transport_close")

    monkeypatch.setattr(runtime, "transmit_loop", fake_transport)
    with pytest.raises(RuntimeError, match="microphone unplugged"):
        asyncio.run(runtime.run(config, duration_s=3))
    assert events.index("transport_close") < events.index("audio_close")
    assert closed.is_set()


def test_vision_failure_closes_audio_and_requests_transport_stop(monkeypatch):
    config, events, holder, warm, closed = setup_runtime(monkeypatch, fail_vision=True)

    async def fake_transport(settings, get_latest, stop, on_connection):
        await stop.wait()

    monkeypatch.setattr(runtime, "transmit_loop", fake_transport)
    with pytest.raises(RuntimeError, match="camera disconnected"):
        asyncio.run(runtime.run(config, duration_s=3))
    assert "audio_close" in events
    assert closed.is_set()


def test_normal_shutdown_reports_close_error_but_still_joins_vision(monkeypatch):
    config, events, holder, warm, closed = setup_runtime(monkeypatch, close_error=True)

    async def fake_transport(settings, get_latest, stop, on_connection):
        await until(warm.is_set)
        stop.set()

    monkeypatch.setattr(runtime, "transmit_loop", fake_transport)
    with pytest.raises(RuntimeError, match="speaker close failure"):
        asyncio.run(runtime.run(config, duration_s=3))
    assert closed.is_set()
