"""One serialized BLE writer; no vision or model imports."""

import asyncio
import logging
import math

from .protocol import COMMAND, NAME, SERVICE, STATUS, Status, encode

logger = logging.getLogger(__name__)
CONNECT_GRACE = 2
STARTUP_IO_TIMEOUT = 5
ACTIVE_WRITE_TIMEOUT = 0.3
DISCONNECT_TIMEOUT = 3


def startup_budget(scan_timeout, connect_timeout):
    return scan_timeout + connect_timeout + CONNECT_GRACE + 2 * STARTUP_IO_TIMEOUT + DISCONNECT_TIMEOUT + 2


class Link:
    def __init__(self, address=None, *, scan_timeout=10.0, connect_timeout=30.0):
        if any(not math.isfinite(value) or value <= 0 for value in (scan_timeout, connect_timeout)):
            raise ValueError("BLE 제한 시간은 0보다 큰 유한한 초 값이어야 합니다.")
        self.address, self.client, self.sequence = address, None, 0
        self.scan_timeout, self.connect_timeout = scan_timeout, connect_timeout

    async def __aenter__(self):
        from bleak import BleakClient, BleakScanner

        print(f"BLE 검색 중 | 제한={self.scan_timeout:g}초", flush=True)
        if self.address:
            device = await BleakScanner.find_device_by_address(self.address, timeout=self.scan_timeout)
        else:
            device = await BleakScanner.find_device_by_filter(
                lambda d, a: (a.local_name or d.name) == NAME and SERVICE in (a.service_uuids or []),
                timeout=self.scan_timeout,
            )
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
            # latency without relaxing the deadline for live guidance writes.
            await self._write(None, STARTUP_IO_TIMEOUT)
            stage = "초기 상태 읽기"
            print(f"BLE {stage} 중 | 제한={STARTUP_IO_TIMEOUT}초", flush=True)
            status = await self.status(timeout=STARTUP_IO_TIMEOUT)
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
            raise

    async def send(self, cue):
        await self._write(cue, ACTIVE_WRITE_TIMEOUT)

    async def _write(self, cue, timeout):
        if self.client is None or not self.client.is_connected:
            raise RuntimeError("BLE disconnected")
        data = encode(self.sequence, cue)
        self.sequence = (self.sequence + 1) & 65535
        await asyncio.wait_for(self.client.write_gatt_char(COMMAND, data, response=True), timeout)

    async def status(self, *, timeout=2):
        result = Status.decode(await asyncio.wait_for(self.client.read_gatt_char(STATUS), timeout))
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

    async def __aexit__(self, *_):
        await self.close()
