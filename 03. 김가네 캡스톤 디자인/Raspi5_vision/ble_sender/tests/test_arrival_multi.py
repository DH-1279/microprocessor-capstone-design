import ctypes
import os
import unittest

from test_sender import hand, packet
from ble_sender.arrival import ArrivalGate
from ble_sender.controller import Controller
from ble_sender.guidance import Mapper
from ble_sender.protocol import Cue, Status, encode


BOX = (0.35, 0.25, 0.45, 0.35)  # Index fingertip is inside.


class ArrivalTests(unittest.TestCase):
    def test_time_same_finger_inner_box_and_once_per_lock(self):
        gate = ArrivalGate()
        for t in (1, 1.1, 1.2, 1.3):
            self.assertFalse(gate.update("a", hand(), BOX, t, t, (640, 640)))
        self.assertTrue(gate.update("a", hand(), BOX, 1.41, 1.41, (640, 640)))
        self.assertFalse(gate.update("a", hand(), BOX, 1.5, 1.5, (640, 640)))
        self.assertFalse(gate.update("b", hand(), BOX, 1.6, 1.6, (640, 640)))

    def test_loss_stale_and_brief_border_overlap_never_complete(self):
        for invalid in (None, "stale", "border"):
            gate = ArrivalGate()
            gate.update("a", hand(), BOX, 1, 1, (640, 640))
            for t in (1.1, 1.2, 1.3, 1.41, 1.5):
                points = hand()
                if invalid == "border":
                    points[8] = (0.35, 0.3)
                box = None if invalid is None else BOX
                self.assertFalse(gate.update("a", points, box, t, t + (1 if invalid == "stale" else 0), (640, 640)))

    def test_alternating_fingers_cannot_accumulate_dwell(self):
        gate = ArrivalGate()
        for n in range(20):
            points = hand()
            points[8] = (0.4, 0.3) if n % 2 else (0.7, 0.7)
            points[12] = (0.7, 0.7) if n % 2 else (0.4, 0.3)
            t = 1 + n / 10
            self.assertFalse(gate.update("a", points, BOX, t, t, (640, 640)))

    def test_controller_event_survives_idle_but_explicit_stop_cancels(self):
        clock = [0.0]
        c = Controller(clock=lambda: clock[0], stable_frames=1, multi_rate=True)
        clock[0] = 1
        self.assertTrue(c.receive(packet(1, captured=1, target_box=BOX, event="arrival")))
        event = c.cue()
        self.assertEqual(event.duties, (255,) * 5)
        self.assertIsNotNone(event.arrival_id)
        c.receive(packet(2, active=False, idle=True))
        clock[0] = 1.5
        self.assertEqual(c.cue(), event)
        c.receive(packet(3, captured=1.5, event="arrival", target_box=BOX))
        self.assertEqual(c.arrival_until, 2.5)  # Duplicate did not extend deadline.
        c.command("stop")
        self.assertIsNone(c.cue())
        c.command("resume")
        clock[0] = 1.6
        c.receive(packet(4, captured=1.6, event="arrival", target_box=BOX))
        self.assertIsNone(c.cue())  # Same completed lock cannot replay.

    def test_controller_event_expires_and_stale_event_is_rejected(self):
        c = Controller(clock=lambda: 0, stable_frames=1)
        self.assertFalse(c.receive(packet(1, captured=1, event="arrival"), now=2))
        self.assertIsNone(c.cue(now=2))
        c.receive(packet(2, captured=2, event="arrival"), now=2)
        self.assertIsNotNone(c.cue(now=2.1))
        self.assertIsNone(c.cue(now=3.51))

    def test_multi_mapper_encodes_different_distances_and_overlap_keeps_near(self):
        mapper = Mapper(mode=2, multi_rate=True, stable_frames=1)
        cue = mapper.update(hand(), BOX, 1, 1, (640, 640)).cue
        self.assertEqual(cue.zones[1], 2)
        self.assertGreater(len(set(cue.zones)), 1)
        wire = encode(1, cue)
        self.assertEqual(wire[1], 2)
        packed = wire[7] | wire[8] << 8
        self.assertEqual(tuple((packed >> (2*i)) & 3 for i in range(5)), cue.zones)


@unittest.skipUnless(os.environ.get("HAPTIC_V2_LIB"), "native bridge required")
class NativeMultiTests(unittest.TestCase):
    def setUp(self):
        self.lib = ctypes.CDLL(os.environ["HAPTIC_V2_LIB"])
        self.lib.v2_reset()
        self.seq = 0

    def send(self, cue, t, ttl=500):
        data = encode(self.seq, cue, ttl)
        self.seq += 1
        raw = (ctypes.c_uint8 * 16).from_buffer_copy(data)
        self.assertEqual(self.lib.v2_receive(raw, 16, t), 1)

    def status(self, t):
        raw = (ctypes.c_uint8 * 16)()
        self.lib.v2_status(t, raw)
        return Status.decode(bytes(raw))

    def test_five_independent_rates_and_refresh_phase(self):
        cue = Cue(2, 0, 1, zones=(0, 1, 2, 0, 2))
        self.send(cue, 0)
        self.assertEqual(self.status(0).duties, (180,) * 5)
        self.assertEqual(self.status(200).duties, (0,) * 5)
        self.send(cue, 300)
        self.assertEqual(self.status(400).duties, (0, 0, 180, 0, 180))
        self.send(cue, 600)
        self.assertEqual(self.status(700).duties, (0, 180, 0, 0, 0))
        self.assertEqual(self.status(1100).duties, (0,) * 5)  # TTL expiry.

    def test_maximum_double_tap_once_even_after_refresh_stop_and_expiry(self):
        cue = Cue(2, 2, 1, (255,) * 5, arrival_id=10)
        self.send(cue, 0)
        for t, duty in ((0, 255), (149, 255), (150, 0), (299, 0), (300, 255), (449, 255), (450, 0)):
            self.assertEqual(self.status(t).duties, (duty,) * 5)
        self.send(cue, 460)
        self.assertEqual(self.status(460).duties, (0,) * 5)
        self.send(None, 470)
        self.send(cue, 480)
        self.assertEqual(self.status(480).duties, (0,) * 5)
        self.send(Cue(2, 2, 1, (255,) * 5, arrival_id=11), 500)
        self.assertEqual(self.status(500).duties, (255,) * 5)
        self.send(None, 510)
        self.assertEqual(self.status(510).duties, (0,) * 5)

    def test_arrival_ttl_cannot_reenergize_on_late_same_event(self):
        cue = Cue(3, 2, 2, (255,) * 5, arrival_id=1)
        self.send(cue, 0, ttl=100)
        self.assertEqual(self.status(100).duties, (0,) * 5)
        self.send(cue, 110)
        self.assertEqual(self.status(110).duties, (0,) * 5)
        self.assertEqual(self.status(110).capabilities, 3)
