"""Linux ALSA capture with stable card ID and bounded USB reconnect attempts."""

import os
import selectors
import subprocess
import threading
from time import monotonic


class AlsaMicrophone:
    def __init__(self, executable, device, block_bytes, on_pcm, on_state,
                 *, retry_s=2.0, stall_s=2.0):
        self.command = [executable, "-q", "-N", "-D", device, "-t", "raw",
                        "-f", "S16_LE", "-r", "16000", "-c", "1"]
        self.block_bytes = block_bytes
        self.on_pcm, self.on_state = on_pcm, on_state
        self.retry_s, self.stall_s = retry_s, stall_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._process = None
        self._thread = None
        self._online = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="alsa-microphone", daemon=True)
        self._thread.start()

    def _notify(self, online, reason=""):
        if self._online != online:
            self._online = online
            self.on_state(online, reason)

    def _session(self):
        process = subprocess.Popen(self.command, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        with self._lock:
            self._process = process
        try:
            # ALSA is Linux-only: selectors can wait on this subprocess pipe.
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                last_data = monotonic()
                pending = bytearray()
                while not self._stop.is_set():
                    code = process.poll()
                    if code is not None:
                        raise ConnectionError(f"arecord 종료 코드 {code}; 장치·USB·다른 녹음 앱 점유 확인")
                    if not selector.select(timeout=0.1):
                        if monotonic() - last_data >= self.stall_s:
                            raise ConnectionError("마이크 PCM 수신 중단")
                        continue
                    data = os.read(process.stdout.fileno(), self.block_bytes)
                    if not data:
                        raise ConnectionError("마이크 입력 스트림 종료")
                    last_data = monotonic()
                    pending.extend(data)
                    while len(pending) >= self.block_bytes and not self._stop.is_set():
                        block = bytes(pending[:self.block_bytes])
                        del pending[:self.block_bytes]
                        self._notify(True)
                        self.on_pcm(block)  # Silent PCM is healthy: do not confuse mute with USB loss.
        finally:
            try:
                self._dispose(process)
            finally:
                with self._lock:
                    self._process = None

    @staticmethod
    def _dispose(process):
        try:
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        finally:
            if process.stdout is not None:
                process.stdout.close()

    def _run(self):
        self._notify(False, "마이크 연결 대기")
        while not self._stop.is_set():
            try:
                self._session()
            except Exception as exc:
                if not self._stop.is_set():
                    self._notify(False, str(exc))
            if self._stop.wait(self.retry_s):
                break

    def abort(self):
        self._stop.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass

    def close(self):
        self.abort()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                raise RuntimeError("마이크 복구 작업 종료 시간 초과")
