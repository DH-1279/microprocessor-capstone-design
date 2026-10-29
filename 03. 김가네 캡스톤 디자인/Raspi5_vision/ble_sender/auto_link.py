"""BLE availability is independent of the camera/voice application's life."""

import asyncio
import math
import signal
from time import monotonic

from .controller import Controller
from .protocol import Channel
from .publisher import HOST


MESSAGES = {
    "WAITING": "장갑이 연결되지 않았습니다. 연결을 기다립니다.",
    "CONNECTED": "장갑이 연결되었습니다.",
    "DISCONNECTED": "장갑 연결이 끊겼습니다. 자동으로 다시 연결합니다.",
    "INCOMPATIBLE": "장갑 펌웨어를 업데이트해주세요. 연결을 다시 확인합니다.",
    "DRY_RUN": "모의 송신 준비가 완료되었습니다. 실제 장갑에는 연결하지 않습니다.",
}
HEALTH_INTERVAL = 2.0


async def run_auto(args, *, ready=None, stop_requested=None, on_fault=None, on_state=None):
    from . import cli  # Runtime import keeps the console/link injectable in tests.

    if not math.isfinite(args.retry_delay) or not 0 < args.retry_delay <= 10:
        raise ValueError("retry-delay는 0초 초과, 10초 이하입니다.")
    loop = asyncio.get_running_loop()
    owner = asyncio.current_task()
    changed, finished = asyncio.Event(), asyncio.Event()
    controller = Controller(mode=args.mode, duties=tuple(args.duties),
                            rate_channel=Channel[args.rate_channel], max_age=args.max_age,
                            stable_frames=args.stable_frames, near=args.near, middle=args.middle,
                            mirrored=args.mirrored, allow_test=args.allow_test,
                            multi_rate=not args.single_rate)
    controller.set_link_ready(False)

    def request_exit():
        finished.set()
        owner.cancel()  # Also interrupt pending scan/connect, not only retry sleep.

    class Commands(asyncio.Queue):
        def put_nowait(self, line):
            if line.lower().split() in (["quit"], ["exit"], ["q"]):
                request_exit()
            else:
                super().put_nowait(line)
                changed.set()

    commands = Commands()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: cli.Inbox(controller, changed), local_addr=(HOST, args.port))
    def close_console():
        pass
    watcher = None
    signals = []
    state = None

    def notify(value):
        nonlocal state
        if value == state:
            return
        state = value
        event = dict(state=value, message=MESSAGES[value])
        print(f"BLE {value} | {event['message']}", flush=True)
        if on_state is not None:
            on_state(event)

    async def watch_owner():
        while not stop_requested.is_set():
            await asyncio.sleep(0.05)
        owner.cancel()

    async def retry_wait(delay):
        deadline = monotonic() + delay
        while monotonic() < deadline:
            changed.clear()
            while not commands.empty():
                line = commands.get_nowait()
                if line.lower().split() == ["reconnect"]:
                    return
                message, _, _ = controller.command(line)
                if message:
                    print(message, flush=True)
            try:
                await asyncio.wait_for(changed.wait(), max(0.001, deadline - monotonic()))
            except TimeoutError:
                break

    try:
        if not args.no_console:
            close_console = cli.start_console(loop, commands, changed)
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, request_exit)
                signals.append(signum)
            except (NotImplementedError, RuntimeError):
                pass
        if stop_requested is not None:
            watcher = asyncio.create_task(watch_owner())
        notify("DRY_RUN" if args.dry_run else "WAITING")
        if ready is not None:
            ready.set()  # IPC is ready; no ESP connection is required to start vision.
        delay = args.retry_delay
        while not finished.is_set():
            was_ready = False
            try:
                async with (cli.ConsoleLink() if args.dry_run else
                            cli.Link(args.address, scan_timeout=args.scan_timeout,
                                     connect_timeout=args.connect_timeout)) as link:
                    controller.set_link_ready(True)  # Discard cached motion and require new frames.
                    was_ready = True
                    delay = args.retry_delay
                    notify("DRY_RUN" if args.dry_run else "CONNECTED")
                    await cli.transmit(controller, link, commands, changed, finished,
                                       reply=lambda s: print(s, flush=True), health_interval=HEALTH_INTERVAL)
            except (TimeoutError, ConnectionError, RuntimeError) as error:
                controller.set_link_ready(False)
                print(f"BLE 연결 재시도 | {type(error).__name__}: {error}", flush=True)
                if "펌웨어 업데이트 필요" in str(error):
                    notify("INCOMPATIBLE")
                elif was_ready:
                    notify("DISCONNECTED")
                    if on_fault is not None and on_state is None:
                        on_fault(str(error))
                await retry_wait(delay)
                delay = min(10.0, delay * 2)
    except asyncio.CancelledError:
        if not finished.is_set() and not (stop_requested is not None and stop_requested.is_set()):
            raise
    finally:
        controller.set_link_ready(False)
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        close_console()
        transport.close()
        for signum in signals:
            loop.remove_signal_handler(signum)
