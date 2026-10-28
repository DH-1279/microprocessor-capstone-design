"""One serialized BLE writer; no vision or model imports."""

import asyncio
import logging

from .protocol import COMMAND, NAME, SERVICE, STATUS, Status, encode

logger = logging.getLogger(__name__)


class Link:
    def __init__(self, address=None):
        self.address, self.client, self.sequence = address, None, 0

    async def __aenter__(self):
        from bleak import BleakClient, BleakScanner

        if self.address:
            device = await BleakScanner.find_device_by_address(self.address, timeout=10)
        else:
            device = await BleakScanner.find_device_by_filter(
                lambda d, a: (a.local_name or d.name) == NAME and SERVICE in (a.service_uuids or []),
                timeout=10,
            )
        if device is None:
            raise RuntimeError(f"{NAME}를 찾지 못했습니다. BLE v2 수신기를 확인하세요.")
        self.client = BleakClient(device)
        try:
            await asyncio.wait_for(self.client.connect(), 15)
            await self.send(None)
            status = await self.status()
            print(
                "BLE 연결 | " + ("ESP DRY_RUN: 실제 모터 출력 없음" if status.flags & 16 else "실제 모터 출력 가능"),
                flush=True,
            )
            return self
        except BaseException:
            await self.close()
            raise

    async def send(self, cue):
        if self.client is None or not self.client.is_connected:
            raise RuntimeError("BLE disconnected")
        data = encode(self.sequence, cue)
        self.sequence = (self.sequence + 1) & 65535
        await asyncio.wait_for(self.client.write_gatt_char(COMMAND, data, response=True), 0.3)

    async def status(self):
        result = Status.decode(await asyncio.wait_for(self.client.read_gatt_char(STATUS), 2))
        if not result.flags & 1 or result.flags & (8 | 32):
            raise RuntimeError(f"ESP 연결/명령/하드웨어 오류: flags={result.flags:#x}")
        return result

    async def close(self):
        if self.client is not None:
            try:
                if self.client.is_connected:
                    await self.send(None)
            except Exception:
                logger.exception("STOP write failed; ESP TTL is the fallback")
            finally:
                try:
                    await asyncio.wait_for(self.client.disconnect(), 3)
                finally:
                    self.client = None

    async def __aexit__(self, *_):
        await self.close()
