"""Lifecycle owned by detector.py; the BLE loop never runs on the vision thread."""

import asyncio
import math
import queue
import threading

from .cli import parser as sender_parser, run
from .link import startup_budget
from .publisher import VisionPublisher


def detector_arguments(argv=None):
    parser = sender_parser()
    parser.description = "영상·음성 안내와 ESP32 BLE 자동 연결"
    parser.add_argument("--no-ble", action="store_true", help="ESP 없이 영상·음성만 실행")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("포트는 1..65535")
    if any(not math.isfinite(value) or value <= 0 for value in (args.scan_timeout, args.connect_timeout)):
        parser.error("BLE 검색·연결 제한은 0보다 큰 유한한 초 값이어야 합니다.")
    return args


class DetectorBLE:
    """Start once, publish current observations, close before audio teardown.

    No BLE connection is silently replaced by dry run. A connection error
    fails startup; --no-ble and --dry-run are explicit operator choices.
    """

    def __init__(self, args):
        self.args = args
        self.ready = threading.Event()
        self.stop_requested = threading.Event()
        self.thread = None
        self.publisher = None
        self.error = None
        self.notices = queue.SimpleQueue()

    def _worker(self):
        try:
            asyncio.run(run(self.args, ready=self.ready, stop_requested=self.stop_requested, on_fault=self.notices.put))
        except asyncio.CancelledError as error:
            if not self.stop_requested.is_set():
                self.error = error
        except BaseException as error:
            self.error = error
        finally:
            self.ready.set()  # Wake start() on failure as well as success.

    def start(self):
        if self.args.no_ble:
            print("BLE 사용 안 함: 영상·음성만 실행합니다.", flush=True)
            return
        if self.thread is not None:
            raise RuntimeError("BLE sender already started")
        self.thread = threading.Thread(target=self._worker, name="detector-ble", daemon=True)
        self.thread.start()
        # Budget includes scanning, connection, initial I/O and cleanup. It
        # must not cancel a legitimate 30s connection after only 30s total.
        startup_timeout = startup_budget(self.args.scan_timeout, self.args.connect_timeout)
        if not self.ready.wait(startup_timeout):
            self.close()
            raise TimeoutError("BLE 초기화 시간 초과: ESP 전원과 Bluetooth를 확인하세요.")
        if not self.check():
            raise RuntimeError("BLE 송신부가 초기화 중 종료되었습니다.")
        self.publisher = VisionPublisher(self.args.port)

    def check(self):
        if self.error is not None:
            detail = str(self.error).strip() or type(self.error).__name__
            raise RuntimeError(f"BLE 송신부 오류: {detail}") from self.error
        return self.args.no_ble or (self.thread is not None and self.thread.is_alive())

    def stop(self):
        if self.publisher is not None:
            self.publisher.stop()

    def pop_notice(self):
        try:
            return self.notices.get_nowait()
        except queue.Empty:
            return None

    def publish(self, *, active, **observation):
        if self.publisher is None:
            return
        if active:
            self.publisher.publish(**observation)
        else:
            self.publisher.idle()

    def arrive(self, **observation):
        if self.publisher is not None:
            return self.publisher.publish(event="arrival", **observation)

    def close(self):
        try:
            if self.publisher is not None:
                self.publisher.close()
                self.publisher = None
        finally:
            self.stop_requested.set()
            if self.thread is not None:
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    raise RuntimeError("BLE 작업이 종료되지 않았습니다. ESP TTL 정지를 확인하세요.")
