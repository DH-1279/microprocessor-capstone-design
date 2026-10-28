"""Nonblocking, loopback-only IPC from the existing vision/voice process.

No models, cameras, BLE or audio are imported here. Times are CLOCK_MONOTONIC
on the SAME Pi, never wall clock and never replaced with the send time.
"""

import json
import socket
import uuid

HOST = "127.0.0.1"
PORT = 8766
MAX_DATAGRAM = 12288


class VisionPublisher:
    def __init__(self, port=PORT):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("port must be 1..65535")
        self.destination = (HOST, port)
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)

    def _send(self, **fields):
        self.sequence += 1
        data = json.dumps(
            dict(version=1, session=self.session, sequence=self.sequence, **fields),
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(data) > MAX_DATAGRAM:
            raise ValueError("vision datagram too large")
        try:
            self.socket.sendto(data, self.destination)
            return True
        except OSError:
            # Do not delay camera/audio if the separate sender is stopped.
            # Missing updates expire at the receiver; there is no local backlog.
            return False

    def publish(self, *, target_id, frame_id, captured_at, size, landmarks, target_box, observed=True, event=None):
        """Call only in active guidance, with matched hand+target in ONE frame.

        target_id: unique per lock, not just a class name. frame_id must increase
        within this publisher lifetime. landmarks: 21 normalized (x,y) pairs or
        MediaPipe points. target_box: normalized (x1,y1,x2,y2).
        observed=False or missing points/box sends STOP, never a held result.
        """
        if not observed or landmarks is None or target_box is None:
            return self.stop()
        points = [(float(p.x), float(p.y)) if hasattr(p, "x") else (float(p[0]), float(p[1])) for p in landmarks]
        return self._send(
            active=True,
            target_id=target_id,
            frame_id=frame_id,
            captured_at=captured_at,
            size=tuple(size),
            landmarks=points,
            target_box=tuple(target_box),
            event=event,
        )

    def idle(self):
        return self._send(active=False, idle=True)

    def stop(self):
        """Call immediately on cancel, reach completion, loss or idle states."""
        return self._send(active=False)

    def close(self):
        if self.socket.fileno() != -1:
            try:
                self.stop()
            finally:
                self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
