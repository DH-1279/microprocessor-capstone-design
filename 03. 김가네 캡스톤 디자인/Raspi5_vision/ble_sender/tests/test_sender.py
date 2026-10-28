import asyncio
import ctypes
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ble_sender.cli import Inbox, transmit
from ble_sender.controller import Controller
from ble_sender.guidance import Mapper, fresh_cue
from ble_sender.link import Link
from ble_sender.protocol import PACKET, Channel, Cue, Status, Zone, encode
from ble_sender.publisher import VisionPublisher


def hand(flip=False, angle=0, aspect=1):
    points = [(0.5, 0.5)] * 21
    for index, xy in {
        0: (0.5, 0.7),
        5: (0.4, 0.5),
        9: (0.5, 0.48),
        13: (0.55, 0.5),
        17: (0.6, 0.5),
        4: (0.33, 0.5),
        8: (0.4, 0.3),
        12: (0.5, 0.28),
        16: (0.55, 0.3),
        20: (0.65, 0.35),
    }.items():
        points[index] = xy
    result = []
    for x, y in points:
        if flip:
            x = 1 - x
        x, y = x - 0.5, y - 0.5
        x, y = (x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle))
        result.append((0.5 + x / aspect, 0.5 + y))
    return result


def box_at(x, y, half=0.01):
    return x - half, y - half, x + half, y + half


def packet(seq=1, captured=1.0, **fields):
    value = dict(
        version=1,
        session="test-process",
        sequence=seq,
        active=True,
        frame_id=seq,
        captured_at=captured,
        target_id="cup-lock-1",
        size=[640, 640],
        landmarks=hand(),
        target_box=box_at(0.15, 0.55),
    )
    value.update(fields)
    return json.dumps(value).encode()


class GeometryTests(unittest.TestCase):
    def result(self, points=None, box=None, **options):
        size = options.pop("size", (640, 640))
        mapper = Mapper(mode=options.pop("mode", 3), stable_frames=1, **options)
        return mapper.update(points or hand(), box or box_at(0.15, 0.55), 1.0, 1.01, size)

    def test_flip_changes_thumb_to_little_without_handedness_label(self):
        a, b = self.result(), self.result(hand(flip=True))
        self.assertEqual((a.facing, a.cue.channel), ("palm", Channel.TH))
        self.assertEqual((b.facing, b.cue.channel), ("back", Channel.PI))

    def test_rotation_and_aspect_preserve_anatomical_direction(self):
        for angle in (0, 0.4, math.pi / 2, math.pi, -math.pi / 2):
            for aspect in (1, 4 / 3, 16 / 9, 0.75):
                x = 0.5 + (-0.35 * math.cos(angle) - 0.05 * math.sin(angle)) / aspect
                y = 0.5 - 0.35 * math.sin(angle) + 0.05 * math.cos(angle)
                result = self.result(hand(angle=angle, aspect=aspect), box_at(x, y), size=(640 * aspect, 640))
                self.assertEqual(result.cue.channel, Channel.TH)

    def test_modes_and_direction_channels(self):
        self.assertEqual(self.result(box=box_at(0.5, 0.15)).cue.channel, Channel.IN)
        self.assertEqual(self.result(box=box_at(0.5, 0.95)).cue.channel, Channel.RI)
        near = self.result(box=box_at(0.51, 0.536, 0.02))
        self.assertEqual((near.cue.zone, near.cue.channel), (Zone.NEAR, Channel.MI))
        for flip in (False, True):
            self.assertEqual(self.result(hand(flip=flip), mode=2).cue.channel, Channel.IN)

    def test_invalid_stale_edge_and_held_frames_stop(self):
        m = Mapper(stable_frames=1)
        edge = hand()
        edge[0] = (0.5, 0.5)
        for points, box, capture, now, observed in (
            (None, box_at(0.1, 0.5), 1, 1.1, True),
            (edge, box_at(0.1, 0.5), 2, 2.1, True),
            (hand(), None, 3, 3.1, True),
            (hand(), box_at(0.1, 0.5), 4, 4.3, True),
            (hand(), box_at(0.1, 0.5), 5, 5.1, False),
        ):
            self.assertIsNone(m.update(points, box, capture, now, (640, 640), observed=observed))
        for bad in (math.nan, math.inf, -1, 2):
            points = hand()
            points[8] = (bad, 0.3)
            self.assertIsNone(self.result(points))

    def test_duplicate_frames_and_direction_switch_need_fresh_votes(self):
        m = Mapper(mode=3, stable_frames=3)
        for t in (1, 1.1):
            self.assertIsNone(m.update(hand(), box_at(0.15, 0.55), t, t, (640, 640)))
        self.assertIsNotNone(m.update(hand(), box_at(0.15, 0.55), 1.2, 1.2, (640, 640)))
        self.assertIsNone(m.update(hand(), box_at(0.15, 0.55), 1.2, 1.21, (640, 640)))
        self.assertIsNone(m.update(hand(), box_at(0.85, 0.55), 1.3, 1.3, (640, 640)))

    def test_hysteresis_and_freshness(self):
        m = Mapper()
        m.zone = Zone.NEAR
        self.assertEqual(m._zone(0.4), Zone.NEAR)
        self.assertEqual(m._zone(0.46), Zone.MIDDLE)
        m.zone = Zone.MIDDLE
        self.assertEqual(m._zone(0.30), Zone.MIDDLE)
        self.assertEqual(m._zone(0.24), Zone.NEAR)
        record = self.result()
        self.assertIsNotNone(fresh_cue(record, 1.25))
        self.assertIsNone(fresh_cue(record, 1.251))
        self.assertIsNone(fresh_cue(record, 0.99))

    def test_mirrored_inference_and_target_transform_together(self):
        r = self.result(hand(flip=True), box_at(0.85, 0.55), mirrored=True)
        self.assertEqual((r.facing, r.cue.channel), ("palm", Channel.TH))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.c = Controller(mode=3, clock=lambda: self.now, stable_frames=1)

    def feed(self, seq=1, **fields):
        self.now = fields.get("captured", seq / 10)
        return self.c.receive(packet(seq, **dict(captured=self.now, **fields)))

    def test_wait_lock_cancel_idle_and_restart(self):
        self.assertIsNone(self.c.cue())
        self.feed()
        self.assertIsNotNone(self.c.cue())
        self.assertTrue(self.c.receive(packet(2, active=False)))
        self.assertIsNone(self.c.cue())
        self.feed(3, target_id="new-lock")
        self.assertIsNotNone(self.c.cue())

    def test_expiry_does_not_relabel_capture_time(self):
        self.feed()
        self.now += 0.36
        self.assertIsNone(self.c.cue())
        self.assertFalse(self.c.receive(packet(2, captured=0.1)))
        self.assertIsNone(self.c.cue())

    def test_old_or_future_or_malformed_messages_cannot_extend_output(self):
        self.feed()
        self.assertFalse(self.c.receive(packet(1)))  # repeated IPC sequence
        self.assertFalse(self.c.receive(packet(2, frame_id=1)))
        self.assertIsNone(self.c.cue())
        self.assertFalse(self.c.receive(packet(3, captured=100)))
        self.assertFalse(self.c.receive(b"invalid-json"))
        self.assertFalse(self.c.receive(b"[1,2]"))
        self.assertFalse(self.c.receive(b"x" * 12289))

    def test_terminal_stop_latches_until_explicit_resume_and_new_frame(self):
        self.feed()
        self.assertEqual(self.c.command("stop")[1:], (True, False))
        self.feed(2)
        self.assertIsNone(self.c.cue())
        self.c.command("resume")
        self.feed(3)
        self.assertIsNotNone(self.c.cue())

    def test_switch_mode_clears_old_observation_and_uses_new_frames(self):
        self.feed()
        self.assertEqual(self.c.cue().channel, Channel.TH)
        self.assertTrue(self.c.command("mode 2")[1])
        self.assertIsNone(self.c.cue())
        self.feed(2)
        self.assertEqual((self.c.cue().mode, self.c.cue().channel), (2, Channel.IN))
        self.assertFalse(self.c.command("mode 2")[1])
        self.assertIsNotNone(self.c.cue())

    def test_target_change_resets_confirmation(self):
        self.c = Controller(mode=3, clock=lambda: self.now, stable_frames=3)
        for seq in (1, 2, 3):
            self.feed(seq)
        self.assertIsNotNone(self.c.cue())
        self.feed(4, target_id="cup-lock-2")
        self.assertIsNone(self.c.cue())

    def test_restart_producer_only_after_previous_input_is_silent(self):
        self.feed()
        self.assertFalse(self.c.receive(packet(1, captured=0.1, session="second")))
        self.now = 1
        self.assertTrue(self.c.receive(packet(1, captured=1, session="second")))
        self.assertIsNotNone(self.c.cue())

    def test_manual_test_is_opt_in_bounded_and_respects_mode(self):
        self.c.command("test middle th 2")
        self.assertIsNone(self.c.cue())
        self.c.allow_test = True
        self.c.command("test middle th 2")
        self.assertEqual(self.c.cue().channel, Channel.TH)
        self.now = 2
        self.assertIsNone(self.c.cue())
        for command in ("test near th 2", "test far mi 2", "test middle th 3", "test far th nan", "test far th -1"):
            self.c.command(command)
            self.assertIsNone(self.c.cue())
        self.c.command("mode 2")
        self.c.command("test middle th 2")
        self.assertIsNone(self.c.cue())
        self.c.command("test near in 1")
        self.assertEqual(self.c.cue().channel, Channel.IN)


class ProtocolTests(unittest.TestCase):
    def test_known_hex_vectors_and_stop_layout(self):
        self.assertEqual(encode(1, Cue(2, 0, 1)).hex(), "02000100f40102000100b4b4b4b4b400")
        self.assertEqual(encode(4, None).hex(), "02010400f40100000000000000000000")
        self.assertEqual(
            encode(0x1234, Cue(3, 1, 4, (150, 160, 170, 180, 190))),
            bytes.fromhex("02 00 34 12 f4 01 03 01 04 00 96 a0 aa b4 be 00"),
        )

    def test_invalid_packets_and_status(self):
        for cue in (
            Cue(3, 2, 1),
            Cue(3, 0, 2),
            Cue(2, 0, 1, (0,) * 5),
            Cue(2.0, 0, 1),
            Cue(2, False, 1),
            Cue(2, 0, True),
        ):
            with self.assertRaises(ValueError):
                encode(0, cue)
        for sequence, ttl in ((-1, 500), (65536, 500), (0, 99), (0, 1001)):
            with self.assertRaises(ValueError):
                encode(sequence, None, ttl)
        valid = PACKET.pack(2, 19, 5, 400, 3, 0, 4, 1, 0, 0, 0, 0, 180, 0)
        self.assertEqual(Status.decode(valid).duties, (0, 0, 0, 0, 180))
        with self.assertRaises(ValueError):
            Status.decode(b"1" * 11)


class FakeClient:
    def __init__(self, fail=False):
        self.is_connected, self.writes, self.fail = True, [], fail

    async def write_gatt_char(self, uuid, data, response):
        self.writes.append((uuid, data, response))
        if self.fail:
            raise OSError("mock BLE error")

    async def disconnect(self):
        self.is_connected = False


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_connect_initial_stop_and_dry_status(self):
        from ble_sender.protocol import NAME, SERVICE

        client = FakeClient()
        client.connect = AsyncMock()
        client.read_gatt_char = AsyncMock(return_value=PACKET.pack(2, 17, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0))

        async def find(predicate, timeout):
            self.assertFalse(
                predicate(SimpleNamespace(name=NAME), SimpleNamespace(local_name=None, service_uuids=None))
            )
            device = SimpleNamespace(name=NAME)
            self.assertTrue(predicate(device, SimpleNamespace(local_name=NAME, service_uuids=[SERVICE])))
            return device

        module = SimpleNamespace(BleakClient=lambda _: client, BleakScanner=SimpleNamespace(find_device_by_filter=find))
        with patch.dict(sys.modules, bleak=module), patch("builtins.print"):
            async with Link():
                self.assertEqual(PACKET.unpack(client.writes[0][1])[1], 1)
        self.assertEqual(len(client.writes), 2)
        self.assertFalse(client.is_connected)

    async def test_ble_sequences_wrap_and_cleanup_stop(self):
        link = Link()
        link.client = client = FakeClient()
        link.sequence = 65535
        await link.send(Cue(2, 0, 1))
        await link.send(Cue(2, 0, 1))
        await link.close()
        packets = [PACKET.unpack(row[1]) for row in client.writes]
        self.assertEqual([p[2] for p in packets], [65535, 0, 1])
        self.assertEqual(packets[-1][1], 1)
        self.assertTrue(all(row[2] for row in client.writes))
        self.assertFalse(client.is_connected)

    async def test_write_failure_still_disconnects(self):
        link = Link()
        link.client = client = FakeClient(fail=True)
        with self.assertLogs("ble_sender.link", level="ERROR"):
            await link.close()
        self.assertFalse(client.is_connected)

    async def test_udp_publisher_to_real_async_inbox(self):
        loop = asyncio.get_running_loop()
        clock = [time.monotonic()]
        c, changed = Controller(stable_frames=1, clock=lambda: clock[0]), asyncio.Event()
        transport, _ = await loop.create_datagram_endpoint(lambda: Inbox(c, changed), local_addr=("127.0.0.1", 0))
        port = transport.get_extra_info("sockname")[1]
        try:
            with VisionPublisher(port) as publisher:
                clock[0] += 1
                publisher.publish(
                    target_id="test",
                    frame_id=1,
                    captured_at=clock[0],
                    size=(640, 640),
                    landmarks=hand(),
                    target_box=box_at(0.15, 0.55),
                )
                await asyncio.wait_for(changed.wait(), 1)
                self.assertIsNotNone(c.cue(), c.reason)
                changed.clear()
                publisher.stop()
                await asyncio.wait_for(changed.wait(), 1)
                self.assertIsNone(c.cue())
        finally:
            transport.close()

    async def test_publisher_stops_for_unobserved_or_missing_data(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(("127.0.0.1", 0))
            receiver.setblocking(False)
            with VisionPublisher(receiver.getsockname()[1]) as publisher:
                publisher.publish(
                    target_id="test",
                    frame_id=1,
                    captured_at=time.monotonic(),
                    size=(640, 640),
                    landmarks=hand(),
                    target_box=None,
                )
                raw = await asyncio.wait_for(asyncio.get_running_loop().sock_recv(receiver, 12288), 1)
                self.assertFalse(json.loads(raw)["active"])

    async def test_transmit_heartbeats_stop_after_stall_and_accept_mode_command(self):
        clock = [1.0]
        c = Controller(mode=3, stable_frames=1, clock=lambda: clock[0])
        clock[0] = 1.01
        c.receive(packet(1, captured=clock[0]))
        commands, changed, finished = asyncio.Queue(), asyncio.Event(), asyncio.Event()
        sent = []
        stopped = asyncio.Event()

        class FakeLink:
            async def send(self, cue):
                sent.append(cue)
                if len(sent) == 3:
                    clock[0] += 1  # Advance source age without a flaky wall-clock test.
                    changed.set()
                if cue is None:
                    stopped.set()

        async def steering():
            await asyncio.wait_for(stopped.wait(), 2)
            commands.put_nowait("mode 2")
            commands.put_nowait("quit")
            changed.set()

        task = asyncio.create_task(steering())
        await transmit(c, FakeLink(), commands, changed, finished, interval=0.01, reply=lambda _: None)
        await task
        self.assertIsNotNone(sent[0])
        self.assertTrue(all(value is not None for value in sent[:3]))
        self.assertTrue(all(value is None for value in sent[3:]))
        self.assertEqual(c.mode, 2)

    async def test_force_stop_is_sent_before_new_mode_or_quit(self):
        c = Controller(mode=3, stable_frames=1)
        commands, changed, finished = asyncio.Queue(), asyncio.Event(), asyncio.Event()
        sent = []
        commands.put_nowait("mode 2")
        commands.put_nowait("quit")

        class FakeLink:
            async def send(self, cue):
                sent.append(cue)

        await transmit(c, FakeLink(), commands, changed, finished, reply=lambda _: None)
        self.assertEqual(sent, [None, None])


@unittest.skipUnless(os.environ.get("HAPTIC_V2_LIB"), "native ESP test library not supplied")
class FirmwareParityTests(unittest.TestCase):
    def test_python_packets_drive_actual_cpp_firmware_patterns_and_expiry(self):
        lib = ctypes.CDLL(os.environ["HAPTIC_V2_LIB"])
        ptr = ctypes.POINTER(ctypes.c_uint8)
        lib.v2_receive.argtypes = (ptr, ctypes.c_size_t, ctypes.c_uint32)
        lib.v2_receive.restype = ctypes.c_int
        lib.v2_status.argtypes = (ctypes.c_uint32, ptr)
        for mode in (2, 3):
            for zone in Zone:
                for channel in Channel:
                    if mode == 3 and ((zone == Zone.NEAR) != (channel == Channel.MI)):
                        continue
                    lib.v2_reset()
                    cue = Cue(mode, zone, channel, (150, 160, 170, 180, 190))
                    for n in range(80):
                        now = n * 50
                        buf = (ctypes.c_uint8 * 16).from_buffer_copy(encode(n, cue))
                        self.assertEqual(lib.v2_receive(buf, 16, now), 1)
                        out = (ctypes.c_uint8 * 16)()
                        lib.v2_status(now, out)
                        actual = Status.decode(bytes(out))
                        on = (
                            (now < 120 or 240 <= now < 360)
                            if mode == 3 and zone == Zone.NEAR
                            else now % (1200, 700, 400)[zone] < 200
                        )
                        self.assertEqual(
                            actual.duties, tuple(cue.duties[j] if on and j == channel else 0 for j in range(5))
                        )
                    lib.v2_status(now + 500, out)
                    final = Status.decode(bytes(out))
                    self.assertTrue(final.flags & 4)
                    self.assertEqual(final.duties, (0,) * 5)


if __name__ == "__main__":
    unittest.main()
