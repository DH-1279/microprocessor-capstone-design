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
from .protocol import Channel, Zone, encode
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
    # Link.__aenter__ already confirmed STOP before this loop starts.
    last_cue, last_write = None, -float("inf")
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
        # STOP is immediate. Active changes are coalesced to at most 10Hz;
        # do not fill the adapter with a burst when direction/zone jitters.
        # An idle STOP needs no heartbeat because it cannot expire into ON.
        if (cue is None and last_cue is not None) or (cue is not None and now - last_write >= interval):
            await link.send(cue)
            if cue != last_cue:
                text = (
                    "STOP"
                    if cue is None
                    else (
                        f"mode={cue.mode} {Zone(cue.zone).name}/{Channel(cue.channel).name} duty={cue.duties[cue.channel]}"
                    )
                )
                reply(f"송신: {text} | {controller.reason}")
            last_cue, last_write = cue, now
        wait_for = interval if cue is None else max(0.001, interval - (monotonic() - last_write))
        try:
            await asyncio.wait_for(changed.wait(), wait_for)
        except TimeoutError:
            pass


async def wait_reconnect(controller, commands, changed, finished, detail, *, reply=print):
    """No link I/O here. Explicit reconnect, then explicit resume, are required."""
    controller.command("stop")
    reply(f"BLE 진동 안내 중단 | {detail}")
    reply("영상·음성은 계속 동작합니다. reconnect로 재연결한 뒤 resume으로 진동 안내를 재개하세요.")
    while not finished.is_set():
        changed.clear()
        while not commands.empty():
            line = commands.get_nowait()
            words = line.lower().split()
            if words == ["reconnect"]:
                return True
            if words and words[0] in ("resume", "test"):
                reply("BLE가 중단되어 있습니다. 먼저 reconnect를 입력하세요.")
                continue
            message, _, quit_requested = controller.command(line)
            if message:
                reply(message)
            if quit_requested:
                finished.set()
                return False
        try:
            await asyncio.wait_for(changed.wait(), 0.1)
        except TimeoutError:
            pass
    return False


async def run(args, *, ready=None, stop_requested=None, on_fault=None):
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
    connected_once = False
    console_started = False

    def reply(text):
        print(text, flush=True)

    try:
        while not finished.is_set():
            connecting = True
            try:
                async with (
                    ConsoleLink()
                    if args.dry_run
                    else Link(
                        args.address, scan_timeout=args.scan_timeout, connect_timeout=args.connect_timeout
                    ) as link
                ):
                    connecting = False
                    was_connected = connected_once
                    connected_once = True
                    print(f"영상 입력: {HOST}:{args.port} | mode={controller.mode} | 왼손 장갑", flush=True)
                    print("명령: mode 2 / mode 3 / status / stop / resume / reconnect / quit", flush=True)
                    if args.allow_test:
                        print("독립 시험 허용: test far in 2 (2안), test middle th 2 (3안)", flush=True)
                    if not args.no_console and not console_started:
                        close_console = start_console(loop, commands, changed)
                        console_started = True
                    if was_connected:
                        reply("BLE 재연결 완료 | 진동은 정지 상태입니다. resume 이후 새 프레임으로 재개합니다.")
                    if ready is not None:
                        ready.set()
                    await transmit(controller, link, commands, changed, finished, reply=reply)
            except Exception as error:
                if not connected_once or (not connecting and not isinstance(error, (TimeoutError, ConnectionError))):
                    raise
                detail = f"{type(error).__name__}: {error}"
                if on_fault is not None:
                    on_fault(detail)
                if not await wait_reconnect(controller, commands, changed, finished, detail, reply=reply):
                    break
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
    p.add_argument("--scan-timeout", type=float, default=10, help="BLE 검색 제한(초), 기본 10")
    p.add_argument("--connect-timeout", type=float, default=30, help="연결·서비스 확인 제한(초), 기본 30")
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
