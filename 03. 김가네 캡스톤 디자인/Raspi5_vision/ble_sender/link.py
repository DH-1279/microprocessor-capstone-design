"""One serialized BLE writer; no vision or model imports."""

import asyncio
import logging
import math
from time import monotonic

from .protocol import COMMAND, NAME, SERVICE, STATUS, Status, encode

logger = logging.getLogger(__name__)
CONNECT_GRACE = 2
STARTUP_IO_TIMEOUT = 5
WRITE_ACK_TIMEOUT = 1.0
DISCONNECT_TIMEOUT = 3


def startup_budget(scan_timeout, connect_timeout):
    return scan_timeout + connect_timeout + CONNECT_GRACE + 2 * STARTUP_IO_TIMEOUT + DISCONNECT_TIMEOUT + 2


class Link:
    def __init__(self, address=None, *, scan_timeout=10.0, connect_timeout=30.0):
        if any(not math.isfinite(value) or value <= 0 for value in (scan_timeout, connect_timeout)):
            raise ValueError("BLE 제한 시간은 0보다 큰 유한한 초 값이어야 합니다.")
        self.address, self.client, self.sequence = address, None, 0
        self.scan_timeout, self.connect_timeout = scan_timeout, connect_timeout
        self.last_operation = "초기화"
        self.last_ack_s = None
        self.last_slow_notice = -math.inf

    async def __aenter__(self):
        from bleak import BleakClient, BleakScanner

        print(f"BLE 검색 중 | 제한={self.scan_timeout:g}초", flush=True)
        try:
            if self.address:
                device = await BleakScanner.find_device_by_address(self.address, timeout=self.scan_timeout)
            else:
                device = await BleakScanner.find_device_by_filter(
                    lambda d, a: (a.local_name or d.name) == NAME and SERVICE in (a.service_uuids or []),
                    timeout=self.scan_timeout,
                )
        except (ValueError, ImportError):
            raise
        except Exception as error:
            raise ConnectionError(f"BLE 검색 실패: {type(error).__name__}: {error}") from error
        if device is None:
            raise RuntimeError(f"{NAME}를 찾지 못했습니다. BLE v2 수신기를 확인하세요.")
        print(f"BLE 장치 발견 | {getattr(device, 'address', self.address) or NAME}", flush=True)
        self.client = BleakClient(device, timeout=self.connect_timeout)
        stage = "연결·GATT 서비스 확인"
        try:
            print(f"BLE {stage} 중 | 제한={self.connect_timeout:g}초", flush=True)
            # Allow BlueZ/Bleak's own timeout to clean up first. The outer
            # deadline only bounds a backend that does not return on time.
            await asyncio.wait_for(self.client.connect(), self.connect_timeout + CONNECT_GRACE)
            print("BLE 연결·서비스 확인 완료", flush=True)
            stage = "초기 STOP 전송"
            print(f"BLE {stage} 중 | 제한={STARTUP_IO_TIMEOUT}초", flush=True)
            # Initial STOP cannot energize a motor, so allow first-ATT setup
            # latency independently of the live-write acknowledgement limit.
            await self._write(None, STARTUP_IO_TIMEOUT)
            stage = "초기 상태 읽기"
            print(f"BLE {stage} 중 | 제한={STARTUP_IO_TIMEOUT}초", flush=True)
            status = await self.status(timeout=STARTUP_IO_TIMEOUT)
            if status.capabilities != 7:
                raise RuntimeError(f"ESP 펌웨어 업데이트 필요: capability={status.capabilities:#04x}, 필요=0x07. ble_final_receiver를 다시 업로드하세요 (350ms 획득 알림/반복 근접 안내).")
            print(
                "BLE 연결 | " + ("ESP DRY_RUN: 실제 모터 출력 없음" if status.flags & 16 else "실제 모터 출력 가능"),
                flush=True,
            )
            return self
        except BaseException as error:
            try:
                # Startup sent only STOP. After an ATT timeout, disconnect
                # instead of queueing another write on the stalled transaction.
                await self.close(send_stop=False)
            except Exception:
                logger.exception("BLE cleanup failed after startup error")
            if isinstance(error, TimeoutError):
                hint = (
                    "ESP에서 flags=11이었다면 연결 뒤 서비스 확인이 지연됐을 수 있습니다."
                    if stage == "연결·GATT 서비스 확인"
                    else "ESP 로그와 GATT 특성 응답을 확인하세요."
                )
                raise RuntimeError(
                    f"BLE {stage} 시간 초과 (주소={getattr(device, 'address', self.address) or NAME}). " + hint
                ) from error
            if isinstance(error, Exception) and not isinstance(error, (RuntimeError, ValueError, ImportError)):
                raise ConnectionError(f"BLE {stage} 실패: {type(error).__name__}: {error}") from error
            raise

    async def send(self, cue):
        await self._write(cue, WRITE_ACK_TIMEOUT)

    async def _write(self, cue, timeout):
        if self.client is None or not self.client.is_connected:
            raise ConnectionError("BLE 연결이 끊어졌습니다.")
        data = encode(self.sequence, cue, ttl_ms=1000 if cue is not None and cue.arrival_id is not None else 500)
        self.last_operation = (
            f"명령 쓰기 seq={self.sequence} STOP"
            if cue is None
            else f"명령 쓰기 seq={self.sequence} mode={int(cue.mode)} zone={int(cue.zone)} "
            f"ch={int(cue.channel)} duty={cue.duties[cue.channel]}"
        )
        if cue is not None and (cue.zones is not None or cue.arrival_id is not None):
            self.last_operation = f"명령 쓰기 seq={self.sequence} {cue.describe()}"
        self.sequence = (self.sequence + 1) & 65535
        started = monotonic()
        try:
            await asyncio.wait_for(self.client.write_gatt_char(COMMAND, data, response=True), timeout)
        except TimeoutError as error:
            raise TimeoutError(
                f"{self.last_operation} | 응답 {monotonic() - started:.3f}초 초과 (제한 {timeout:g}초)"
            ) from error
        except Exception as error:
            raise ConnectionError(f"{self.last_operation} 실패: {type(error).__name__}: {error}") from error
        self.last_ack_s = monotonic() - started
        if cue is not None and self.last_ack_s > 0.3 and monotonic() - self.last_slow_notice >= 5:
            self.last_slow_notice = monotonic()
            logger.warning("BLE 응답 지연 %.3f초 (연결 유지) | %s", self.last_ack_s, self.last_operation)

    async def status(self, *, timeout=2):
        self.last_operation = "상태 읽기"
        try:
            data = await asyncio.wait_for(self.client.read_gatt_char(STATUS), timeout)
        except TimeoutError as error:
            raise TimeoutError(f"BLE 상태 읽기 응답 시간 초과 (제한 {timeout:g}초)") from error
        except Exception as error:
            raise ConnectionError(f"BLE 상태 읽기 실패: {type(error).__name__}: {error}") from error
        result = Status.decode(data)
        if not result.flags & 1 or result.flags & (8 | 32):
            raise RuntimeError(f"ESP 연결/명령/하드웨어 오류: flags={result.flags:#x}")
        return result

    async def close(self, *, send_stop=True):
        if self.client is not None:
            try:
                if send_stop and self.client.is_connected:
                    await self.send(None)
            except Exception:
                logger.exception("STOP write failed; ESP TTL is the fallback")
            finally:
                try:
                    await asyncio.wait_for(self.client.disconnect(), DISCONNECT_TIMEOUT)
                finally:
                    self.client = None

    async def __aexit__(self, exc_type, exc, _):
        try:
            await self.close(send_stop=not isinstance(exc, (TimeoutError, ConnectionError)))
        except Exception:
            if exc is None:
                raise
            logger.exception("BLE 종료 중 오류: 원래 통신 오류를 유지합니다.")
