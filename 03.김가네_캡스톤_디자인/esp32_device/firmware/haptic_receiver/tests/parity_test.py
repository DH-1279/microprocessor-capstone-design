"""Compare the real portable C++ receiver with the Pi simulator via ctypes."""

import argparse
import ctypes
import importlib
from pathlib import Path
import random
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True, help="compiled bridge .dll or .so")
    parser.add_argument(
        "--python-root", type=Path,
        default=Path(__file__).resolve().parents[4] / "Raspi5_vision",
        help="directory containing the haptic_mvp Python package",
    )
    parser.add_argument("--operations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1279)
    args = parser.parse_args()
    if args.operations <= 0:
        parser.error("--operations must be positive")
    if not args.library.is_file():
        parser.error("--library must refer to the compiled host bridge")
    sys.path.insert(0, str(args.python_root.resolve()))
    protocol = importlib.import_module("haptic_mvp.protocol")
    library = ctypes.CDLL(str(args.library.resolve()))
    for name in ("reset", "connect_receiver", "disconnect_receiver"):
        function = getattr(library, name)
        function.argtypes = []
        function.restype = None
    library.receive_packet.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t, ctypes.c_uint32,
    ]
    library.receive_packet.restype = ctypes.c_int
    library.snapshot_receiver.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_ubyte)]
    library.snapshot_receiver.restype = None
    library.reset()
    model = protocol.SimulatedReceiver()
    rng = random.Random(args.seed)
    now_ms = 0
    history: list[tuple[int, int, str]] = []

    for index in range(args.operations):
        now_ms += rng.randrange(0, 81)
        operation = rng.randrange(12)
        if operation == 0:
            library.connect_receiver()
            model.connect()
            event = "connect"
        elif operation == 1:
            library.disconnect_receiver()
            model.disconnect()
            event = "disconnect"
        elif operation in (2, 3):
            if model.deadline:
                # Deliberately hit exact millisecond expiry, not just a later tick.
                now_ms = max(now_ms, model.deadline)
            event = "tick"
        else:
            previous = model.sequence if model.sequence is not None else rng.randrange(65536)
            sequence = (previous + rng.choice([-2, 0, 1, 1, 1, 2, 32768])) % 65536
            stop = rng.randrange(6) == 0
            levels = (
                (0,) * 5 if stop or rng.randrange(6) == 0
                else tuple(rng.randrange(256) for _ in range(5))
            )
            packet = protocol.Command(
                sequence, rng.choice([100, 300, 500, 1000]), levels, stop,
            ).encode()
            if rng.randrange(5) == 0:
                packet = rng.choice([
                    packet[:-1], packet + b"\x00", b"\x02" + packet[1:],
                    packet[:1] + b"\x80" + packet[2:],
                ])
            buffer = (ctypes.c_ubyte * len(packet)).from_buffer_copy(packet)
            actual_accept = bool(library.receive_packet(buffer, len(packet), now_ms))
            expected_accept = model.receive(packet, now_ms / 1000)
            if actual_accept != expected_accept:
                raise AssertionError(
                    (index, "accept mismatch", actual_accept, expected_accept, packet.hex())
                )
            event = "receive " + packet.hex()
        history.append((index, now_ms, event))
        history = history[-8:]
        model.tick(now_ms / 1000)
        result = (ctypes.c_ubyte * 11)()
        library.snapshot_receiver(now_ms, result)
        flags = (
            16 | int(model.connected) | (2 if any(model.levels) else 0)
            | (4 if model.expired else 0) | (8 if model.invalid else 0)
        )
        remaining = max(0, model.deadline - now_ms) if any(model.levels) else 0
        got = (
            result[0], result[1], result[2] + (result[3] << 8),
            result[4] + (result[5] << 8), tuple(result[6:]),
        )
        wanted = (1, flags, model.sequence or 0, remaining, model.levels)
        if got != wanted:
            raise AssertionError(("state mismatch", got, wanted, history))
    print(
        f"PASS: {args.operations} Python/C++ receiver parity operations (seed={args.seed}); "
        "malformed packets, replay, exact timeout, disconnect, reconnect, remaining TTL"
    )


if __name__ == "__main__":
    main()
