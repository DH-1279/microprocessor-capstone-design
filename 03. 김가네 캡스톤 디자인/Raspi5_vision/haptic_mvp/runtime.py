"""Concurrent hardware runtime. Main loop never waits for inference, audio or BLE."""

import asyncio
from dataclasses import dataclass
import json
import queue
import sys
import threading
import time

from .ble_link import transmit_loop
from .controller import Controller
from .models import Decision


class Latest:
    def __init__(self):
        self._value = None
        self._lock = threading.Lock()

    def put(self, value):
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value

    def take(self):
        with self._lock:
            value, self._value = self._value, None
            return value


def is_stop(text):
    return isinstance(text, str) and (text.strip().lower() in ("stop", "off")
                                      or any(x in text for x in ("정지", "멈춰", "그만", "중지", "종료")))


class CommandInbox(Latest):
    """Do not overwrite an unprocessed STOP with a later target request."""

    def put(self, value):
        with self._lock:
            if value is None or not is_stop(self._value):
                self._value = value


class ConsoleAudio:
    speaking = False
    listening = False

    def start(self):
        pass

    def say(self, text):
        print(f"안내: {text}", flush=True)
        return True

    def get_error(self):
        return None

    def close(self):
        pass

    def discard_input(self):
        pass


@dataclass
class SpeechMessage:
    text: str
    requested_at: float
    target: str | None
    reason: str
    epoch: int


class PendingSpeech:
    """At most one target acknowledgement and one current status announcement.

    A still-valid target acknowledgement survives changes in hand visibility and
    proximity. Summaries/status expire instead of building a delayed speech queue.
    """

    INVALID_TARGET_REASONS = {"target_lost", "stale_frame", "ble_disconnected", "link_change"}

    def __init__(self):
        self.ack: SpeechMessage | None = None
        self.status: SpeechMessage | None = None

    def update(self, decision: Decision, now: float, epoch: int, command: str | None = None) -> None:
        if command is not None:
            self.ack = self.status = None
        if self.ack and (self.ack.target != decision.target or self.ack.epoch != epoch
                         or decision.reason in self.INVALID_TARGET_REASONS or decision.state == "stopped"):
            self.ack = None
        if self.status and (self.status.target != decision.target or self.status.epoch != epoch
                            or self.status.reason != decision.reason or now - self.status.requested_at > 3):
            self.status = None
        if decision.speech:
            message = SpeechMessage(decision.speech, now, decision.target, decision.reason, epoch)
            if command is not None and decision.target is not None and decision.reason not in self.INVALID_TARGET_REASONS:
                self.ack = message
            else:
                self.status = message

    def deliver(self, speaker) -> bool:
        if speaker.speaking or speaker.listening:
            return False
        message = self.ack or self.status
        if message is None or not speaker.say(message.text):
            return False
        if self.ack is message:
            self.ack = None
        else:
            self.status = None
        return True


async def run(config: dict, duration_s: float | None = None) -> None:
    if sys.platform != "linux":
        raise RuntimeError("Hardware run requires Raspberry Pi/Linux. Use simulate on this PC.")
    from .vision import run_vision

    loop = asyncio.get_running_loop()
    observations, commands = Latest(), CommandInbox()
    errors = queue.Queue()
    thread_stop = threading.Event()
    stop = asyncio.Event()
    controller = Controller(config["controller"])
    latest = [Decision(), time.monotonic()]
    dry = config["ble"]["dry_run"]
    link_ready = dry
    link_epoch = 0

    def on_text(text):
        commands.put(text)

    if config["audio"]["enabled"]:
        from .audio import AudioService
        speaker = AudioService(config["audio"], on_text)
    else:
        speaker = ConsoleAudio()

    def vision_worker():
        try:
            run_vision(config["vision"], observations.put, thread_stop)
            if not thread_stop.is_set():
                errors.put("Vision input ended")
        except Exception as error:
            errors.put(f"Vision: {type(error).__name__}: {error}")

    def on_connection(connected):
        nonlocal link_ready, link_epoch
        link_ready = connected
        link_epoch += 1
        speaker.discard_input()
        # Clear synchronously before the transport obtains its next decision.
        controller._clear_target()
        controller.user_stopped = True
        latest[:] = [Decision(state="stopped", reason="link_change"), time.monotonic()]
        commands.put(None)
        print("BLE 연결 상태 변경: 목표를 다시 선택해 주세요.", flush=True)

    def keyboard_ready():
        text = sys.stdin.readline()
        if not text or text.strip().lower() in ("quit", "exit", "q"):
            stop.set()
        elif text.strip():
            on_text(text.strip())

    thread = threading.Thread(target=vision_worker, name="vision", daemon=True)
    transport = None
    keyboard_added = False
    started = time.monotonic()
    failure = None
    try:
        speaker.start()
        thread.start()
        if not dry:
            transport = asyncio.create_task(transmit_loop(
                config["ble"], lambda: tuple(latest), stop, on_connection))
        if sys.stdin.isatty():
            loop.add_reader(sys.stdin.fileno(), keyboard_ready)
            keyboard_added = True
        print(f"MVP started | BLE={'LOG ONLY' if dry else 'REAL'} | type '컵 찾아줘', '정지', 'quit'", flush=True)
        print("진동 채널은 임시 매핑입니다. 수신 펌웨어 기본값은 모터 출력 없는 DRY_RUN입니다.", flush=True)
        previous = None
        pending_speech = PendingSpeech()
        while not stop.is_set():
            now = time.monotonic()
            if duration_s is not None and now - started >= duration_s:
                break
            if not errors.empty():
                raise RuntimeError(errors.get_nowait())
            audio_error = speaker.get_error()
            if audio_error:
                raise RuntimeError(f"Audio: {audio_error}")
            if transport is not None and transport.done():
                transport.result()
                raise RuntimeError("BLE task stopped unexpectedly")
            command = commands.take()
            if is_stop(command):
                speaker.discard_input()
                commands.put(None)
            busy = speaker.speaking or speaker.listening
            decision = controller.step(now, observations.get(), command, busy)
            if not link_ready and not dry:
                decision = Decision(state="paused", reason="ble_disconnected", speech=decision.speech)
            latest[:] = [decision, now]
            pending_speech.update(decision, now, link_epoch, command)
            pending_speech.deliver(speaker)
            signature = (decision.state, decision.reason, decision.target, decision.levels)
            if signature != previous:
                print(json.dumps({"state": decision.state, "reason": decision.reason,
                                  "target": decision.target, "direction": decision.direction,
                                  "proximity": decision.proximity, "levels": decision.levels}, ensure_ascii=False), flush=True)
                previous = signature
            await asyncio.sleep(0.02)
    except BaseException as error:
        failure = error
    finally:
        latest[:] = [Decision(state="stopped", reason="shutdown"), time.monotonic()]
        stop.set()
        thread_stop.set()
        if keyboard_added:
            try:
                loop.remove_reader(sys.stdin.fileno())
            except Exception as error:
                if failure is None:
                    failure = error
        # Signal/cancel BLE before waiting for audio or camera shutdown.
        if transport is not None:
            transport.cancel()
            try:
                await asyncio.wait_for(transport, 5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as error:
                if failure is None:
                    failure = error
        try:
            await asyncio.to_thread(speaker.close)
        except Exception as error:
            if failure is None:
                failure = error
        if thread.ident is not None:
            await asyncio.to_thread(thread.join, 3)
        if thread.is_alive():
            print("Vision worker did not stop within 3 seconds; receiver watchdog remains active.", file=sys.stderr)
        print("MVP stopped | OFF requested", flush=True)
    if failure is not None:
        raise failure
