"""One outstanding BLE write, reconnect with a new target-selection requirement."""

import asyncio
import math
import time
from typing import Callable

from .models import Decision
from .protocol import COMMAND_UUID, SERVICE_UUID, STATUS_UUID, Command, Status


class BleLink:
    def __init__(self, config: dict):
        self.config = config
        self.client = None
        self.sequence = 0
        self.connected = False
        self.connect_count = 0

    async def connect(self) -> None:
        from bleak import BleakClient, BleakScanner

        found = await BleakScanner.discover(timeout=self.config["scan_timeout_s"], return_adv=True)
        devices = [device for device, adv in found.values()
                   if SERVICE_UUID in {uuid.lower() for uuid in (adv.service_uuids or [])}
                   and (self.config["address"] is None
                        or device.address.lower() == self.config["address"].lower())]
        if len(devices) != 1:
            raise RuntimeError(f"Haptic receiver count={len(devices)}; check firmware/power or configure address")
        def disconnected(client):
            # A late callback from the previous connection must not clear a new one.
            if self.client is client:
                self.connected = False

        client = BleakClient(devices[0], disconnected_callback=disconnected)
        self.client = client
        try:
            await asyncio.wait_for(client.connect(), self.config["operation_timeout_s"] * 4)
            handshake_sequence = self.sequence
            await asyncio.wait_for(client.write_gatt_char(
                COMMAND_UUID, Command(self.sequence, self.config["ttl_ms"], (0, 0, 0, 0, 0), True).encode(),
                response=True), self.config["operation_timeout_s"])
            self.sequence = (self.sequence + 1) & 65535
            status = Status.decode(bytes(await asyncio.wait_for(
                client.read_gatt_char(STATUS_UUID), self.config["operation_timeout_s"])))
            if (not status.flags & 1 or status.flags & 14 or any(status.levels)
                    or status.remaining_ms != 0 or status.sequence != handshake_sequence):
                raise RuntimeError("Receiver did not confirm connected STOP state")
            self.connected = True
            self.connect_count += 1
            print(f"BLE connected: {devices[0].address}; receiver dry_run={bool(status.flags & 16)}", flush=True)
        except BaseException:
            self.connected = False
            try:
                await asyncio.wait_for(client.disconnect(), 2)
            except Exception:
                pass
            self.client = None
            raise

    async def send(self, decision: Decision) -> None:
        if not self.connected or self.client is None or not self.client.is_connected:
            self.connected = False
            raise ConnectionError("BLE disconnected")
        packet = Command(self.sequence, self.config["ttl_ms"], decision.levels, decision.stopped).encode()
        self.sequence = (self.sequence + 1) & 65535
        try:
            await asyncio.wait_for(self.client.write_gatt_char(COMMAND_UUID, packet, response=True),
                                   self.config["operation_timeout_s"])
        except BaseException:
            self.connected = False
            raise

    async def close(self) -> None:
        client = self.client
        if client is None:
            return
        try:
            if client.is_connected:
                await asyncio.wait_for(client.write_gatt_char(
                    COMMAND_UUID, Command(self.sequence, self.config["ttl_ms"], (0, 0, 0, 0, 0), True).encode(),
                    response=True), 1)
        except Exception:
            pass  # Receiver watchdog is the independent fallback.
        finally:
            self.connected = False
            self.client = None
            try:
                await asyncio.wait_for(client.disconnect(), 2)
            except Exception:
                pass


async def transmit_loop(config: dict, get_latest: Callable[[], tuple[Decision, float]],
                        stop: asyncio.Event, on_connection: Callable[[bool], None],
                        link_factory=BleLink) -> None:
    link = link_factory(config)
    previously_connected = False
    try:
        while not stop.is_set():
            try:
                if not link.connected:
                    if previously_connected:
                        on_connection(False)
                        previously_connected = False
                        await link.close()
                    await link.connect()
                    on_connection(True)
                    previously_connected = True
                decision, generated = get_latest()
                # Never refresh an actuator watchdog with a stale application decision.
                age = time.monotonic() - generated
                if not math.isfinite(age) or age < -0.05 or age > config["ttl_ms"] / 1000:
                    decision = Decision(state="paused", reason="stale_decision")
                await link.send(decision)
                try:
                    await asyncio.wait_for(stop.wait(), config["send_interval_s"])
                except asyncio.TimeoutError:
                    pass
            except (Exception,) as error:
                if previously_connected:
                    on_connection(False)
                    previously_connected = False
                print(f"BLE retry: {error}", flush=True)
                await link.close()
                try:
                    await asyncio.wait_for(stop.wait(), config["reconnect_delay_s"])
                except asyncio.TimeoutError:
                    pass
    finally:
        await link.close()
