import asyncio
import socket
import threading
import time
import unittest
from unittest.mock import patch

from ble_sender.cli import Inbox, run
from ble_sender.integration import detector_arguments
from ble_sender.publisher import VisionPublisher
from test_integration import replay
from test_sender import hand, box_at


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class AutoStartupTests(unittest.TestCase):
    def test_detector_video_voice_start_while_esp_is_off(self):
        class Missing:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                raise RuntimeError("ESP not powered")
            async def __aexit__(self, *a):
                pass
        with patch("ble_sender.cli.Link", Missing):
            result = replay([{}, {"text": "캡스톤"}, {"text": "컵"}, {}, {}], real_runtime=True)
        self.assertEqual(result.code, 0, result.log)
        self.assertEqual(len(result.states), 5)
        self.assertEqual(result.states[-1], "SEARCHING")
        self.assertTrue(any("장갑이 연결되지 않았습니다" in text for text in result.spoken))
        self.assertFalse(result.bridge.thread.is_alive())


class AutoRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_power_loss_is_detected_without_a_motion_write(self):
        states, sessions = [], []
        recovered = asyncio.Event()
        stop = threading.Event()
        args = detector_arguments(["--no-console", "--port", str(free_port()), "--retry-delay", "0.01"])
        class IdleLink:
            def __init__(self, *a, **kw):
                self.index = len(sessions)
                sessions.append(self)
            async def __aenter__(self):
                if self.index == 1:
                    recovered.set()
                return self
            async def status(self):
                if self.index == 0:
                    raise ConnectionError("ESP powered off while idle")
                return "idle"
            async def send(self, cue):
                self_test.assertIsNone(cue)
            async def __aexit__(self, *a):
                pass
        self_test = self
        with patch("ble_sender.cli.Link", IdleLink), patch("ble_sender.auto_link.HEALTH_INTERVAL", 0.01), patch("builtins.print"):
            task = asyncio.create_task(run(args, stop_requested=stop, on_state=states.append))
            try:
                await asyncio.wait_for(recovered.wait(), 1)
                self.assertIn("DISCONNECTED", [e["state"] for e in states])
            finally:
                stop.set()
                await asyncio.wait_for(task, 1)

    async def test_off_then_on_then_disconnect_reconnect_and_fresh_resume(self):
        sessions, sent, controllers, states = [], [], [], []
        disconnected, connected_again, resumed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        ready, stop = asyncio.Event(), threading.Event()
        args = detector_arguments(["--no-console", "--port", str(free_port()), "--retry-delay", "0.03",
                                   "--stable-frames", "1"])
        def inbox(c, changed):
            controllers.append(c)
            return Inbox(c, changed)
        class Link:
            def __init__(self, *a, **kw):
                self.index = len(sessions)
                sessions.append(self)
            async def __aenter__(self):
                if self.index == 0:
                    raise RuntimeError("ESP off")
                if self.index == 2:
                    connected_again.set()
                return self
            async def send(self, cue):
                sent.append((self.index, cue))
                if self.index == 1 and cue is not None:
                    disconnected.set()
                    raise TimeoutError("mock write delay")
                if self.index == 2 and cue is not None:
                    resumed.set()
            async def __aexit__(self, *a):
                pass
        with patch("ble_sender.cli.Link", Link), patch("ble_sender.cli.Inbox", side_effect=inbox), patch("builtins.print"):
            task = asyncio.create_task(run(args, ready=ready, stop_requested=stop, on_state=states.append))
            publisher = VisionPublisher(args.port)
            frame = 0
            async def stream_until(event):
                nonlocal frame
                for _ in range(60):
                    if event.is_set():
                        return
                    frame += 1
                    publisher.publish(target_id="target", frame_id=frame, captured_at=time.monotonic(),
                                      size=(640, 640), landmarks=hand(), target_box=box_at(0.15, 0.55))
                    await asyncio.sleep(0.02)
                self.fail(f"event missing: {task.done()}, {controllers[0].reason}")
            try:
                await asyncio.wait_for(ready.wait(), 1)
                await stream_until(disconnected)
                await asyncio.wait_for(connected_again.wait(), 1)
                await asyncio.sleep(0.05)
                self.assertFalse(resumed.is_set())  # Cached pre-disconnect cue was discarded.
                await stream_until(resumed)
                self.assertEqual([e["state"] for e in states], ["WAITING", "CONNECTED", "DISCONNECTED", "CONNECTED"])
                self.assertFalse(controllers[0].paused)
            finally:
                publisher.close()
                stop.set()
                await asyncio.wait_for(task, 1)

    async def test_quit_interrupts_pending_scan(self):
        entered, terminal = asyncio.Event(), []
        args = detector_arguments(["--port", str(free_port())])
        class Pending:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                entered.set()
                await asyncio.sleep(30)
            async def __aexit__(self, *a):
                pass
        def console(loop, commands, changed):
            terminal.append(commands)
            return lambda: None
        with patch("ble_sender.cli.Link", Pending), patch("ble_sender.cli.start_console", console), patch("builtins.print"):
            task = asyncio.create_task(run(args))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                terminal[0].put_nowait("quit")
                await asyncio.wait_for(task, 1)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
