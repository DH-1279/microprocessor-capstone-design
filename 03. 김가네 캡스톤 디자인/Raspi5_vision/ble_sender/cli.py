"""Run from Raspi5_vision: python -m ble_sender --mode 2."""

import argparse
import asyncio
import logging
import signal
import sys
import threading
from time import monotonic

from .controller import Controller
from .link import Link
from .protocol import Channel, encode
from .publisher import HOST, PORT


class Inbox(asyncio.DatagramProtocol):
    def __init__(self, controller, changed):
        self.controller, self.changed = controller, changed

    def datagram_received(self, data, addr):
        if addr[0] == HOST:
            self.controller.receive(data)
            self.changed.set()


class ConsoleLink:
    """Host dry run: still exercises the real packet encoder, no BLE import."""

    def __init__(self):
        self.sequence = 0

    async def __aenter__(self):
        print("송신부 DRY_RUN: BLE 연결과 실제 모터 출력 없음", flush=True)
        return self

    async def send(self, cue):
        encode(self.sequence, cue)
        self.sequence = (self.sequence + 1) & 65535

    async def status(self):
        return f"송신부 DRY_RUN | 전송 횟수={self.sequence} (sequence 순환)"

    async def __aexit__(self, *_):
        await self.send(None)


def start_console(loop, commands, changed):
    def enqueue(line):
        commands.put_nowait(line)
        changed.set()

    # Linux/PuTTY: avoid a blocking reader thread holding stdin during exit.
    try:
        fd = sys.stdin.fileno()

        def readable():
            line = sys.stdin.readline()
            enqueue(line if line else "quit")
            if not line:
                loop.remove_reader(fd)

        loop.add_reader(fd, readable)
        return lambda: loop.remove_reader(fd)
    except (NotImplementedError, OSError, ValueError):
        pass  # Windows event loops / regular redirected files.

    def read():
        while True:
            line = sys.stdin.readline()
            try:
                loop.call_soon_threadsafe(enqueue, line if line else "quit")
            except RuntimeError:
                return  # Event loop already closed after Ctrl+C or a link error.
            if not line:
                return

    threading.Thread(target=read, name="ble-console", daemon=True).start()
    return lambda: None


async def transmit(controller, link, commands, changed, finished, *, interval=0.1, reply=print):
    """Only this coroutine writes BLE; callbacks update desired state only."""
    last_cue, last_write = object(), -float("inf")
    while not finished.is_set():
        changed.clear()
        while not commands.empty():
            line = commands.get_nowait()
            message, force_stop, quit_requested = controller.command(line)
            if force_stop:
                await link.send(None)
                last_cue, last_write = None, monotonic()
            if message:
                reply(message)
            if line.lower().split() == ["status"]:
                reply(str(await link.status()))
            if quit_requested:
                finished.set()
                break
        if finished.is_set():
            break
        # Re-evaluate capture age AFTER every BLE await; a queued old cue must
        # never acquire a new freshness timestamp because a write was slow.
        now = monotonic()
        cue = controller.cue()
        if cue != last_cue or now - last_write >= interval:
            await link.send(cue)
            if cue != last_cue:
                text = (
                    "STOP"
                    if cue is None
                    else (f"mode={cue.mode} {cue.zone.name}/{cue.channel.name} duty={cue.duties[cue.channel]}")
                )
                reply(f"송신: {text} | {controller.reason}")
            last_cue, last_write = cue, now
        wait_for = max(0.001, interval - (monotonic() - last_write))
        try:
            await asyncio.wait_for(changed.wait(), wait_for)
        except TimeoutError:
            pass


async def run(args, *, ready=None, stop_requested=None):
    loop = asyncio.get_running_loop()
    changed, finished = asyncio.Event(), asyncio.Event()
    controller = Controller(
        mode=args.mode,
        duties=tuple(args.duties),
        rate_channel=Channel[args.rate_channel],
        max_age=args.max_age,
        stable_frames=args.stable_frames,
        near=args.near,
        middle=args.middle,
        mirrored=args.mirrored,
        allow_test=args.allow_test,
    )
    transport, _ = await loop.create_datagram_endpoint(
        lambda: Inbox(controller, changed),
        local_addr=(HOST, args.port),
    )
    commands = asyncio.Queue()

    def close_console():
        pass

    def request_exit():
        finished.set()
        changed.set()

    installed_signals = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_exit)
            installed_signals.append(signum)
        except (NotImplementedError, RuntimeError):
            pass

    owner = asyncio.current_task()

    async def watch_owner():
        while True:
            if stop_requested.is_set():
                owner.cancel()  # Also cancel a scan/connect during startup.
                return
            await asyncio.sleep(0.05)

    watcher = asyncio.create_task(watch_owner()) if stop_requested is not None else None
    try:
        async with ConsoleLink() if args.dry_run else Link(args.address) as link:
            print(f"영상 입력: {HOST}:{args.port} | mode={args.mode} | 왼손 장갑", flush=True)
            print("명령: mode 2 / mode 3 / status / stop / resume / quit", flush=True)
            if args.allow_test:
                print("독립 시험 허용: test far in 2 (2안), test middle th 2 (3안)", flush=True)
            if not args.no_console:
                close_console = start_console(loop, commands, changed)
            if ready is not None:
                ready.set()  # Only after connection, initial STOP and status validation.
            await transmit(controller, link, commands, changed, finished, reply=lambda s: print(s, flush=True))
    finally:
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        close_console()
        transport.close()
        for signum in installed_signals:
            loop.remove_signal_handler(signum)


def parser():
    p = argparse.ArgumentParser(description="김가네 왼손 장갑 BLE v2 송신부")
    p.add_argument("--mode", type=int, choices=(2, 3), default=2)
    p.add_argument("--address", help="ESP BLE 주소. 생략하면 이름/서비스로 검색")
    p.add_argument("--port", type=int, default=PORT, help="Pi 내부 UDP 포트")
    p.add_argument("--dry-run", action="store_true", help="BLE 없이 매핑/패킷만 검사")
    p.add_argument("--allow-test", action="store_true", help="최대 2초 독립 모터 시험 허용")
    p.add_argument("--no-console", action="store_true", help="키보드 입력 없이 실행")
    p.add_argument("--rate-channel", choices=tuple(Channel.__members__), default="IN")
    p.add_argument(
        "--duties",
        nargs=5,
        type=int,
        default=[180] * 5,
        metavar=("TH", "IN", "MI", "RI", "PI"),
        help="채널별 duty 1..255",
    )
    p.add_argument("--max-age", type=float, default=0.35, help="원본 프레임 유효 시간(초)")
    p.add_argument("--stable-frames", type=int, default=3)
    p.add_argument("--near", type=float, default=0.35, help="손 폭 대비 near 경계")
    p.add_argument("--middle", type=float, default=1.5, help="손 폭 대비 middle 경계")
    p.add_argument("--mirrored", action="store_true", help="추론 입력 자체가 좌우 반전된 경우")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if not 1 <= args.port <= 65535:
        p.error("포트는 1..65535")
    logging.basicConfig(level=logging.WARNING)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass
    except (ValueError, RuntimeError, OSError, TimeoutError, ImportError) as error:
        p.exit(1, f"송신부 종료: {error}\n")
