"""BLE session/transport tests use asynchronous fakes, not an adapter or radio."""

import asyncio
import sys
import time
from types import SimpleNamespace

import pytest

from haptic_mvp.ble_link import BleLink, transmit_loop
from haptic_mvp.models import Decision
from haptic_mvp.protocol import COMMAND_UUID, PACKET, SERVICE_UUID, STATUS_UUID, Command


@pytest.fixture
def config():
    return {"address": None, "scan_timeout_s": 0.01, "operation_timeout_s": 0.03,
            "ttl_ms": 500, "send_interval_s": 0.001, "reconnect_delay_s": 0.001}


def fake_bleak(monkeypatch, *, devices=1, status=None):
    instances = []

    class Scanner:
        @staticmethod
        async def discover(**kwargs):
            return {str(number): (SimpleNamespace(address=f"AA:BB:CC:00:00:{number:02X}"),
                                  SimpleNamespace(service_uuids=[SERVICE_UUID.upper()]))
                    for number in range(devices)}

    class Client:
        def __init__(self, device, disconnected_callback):
            self.device = device
            self.callback = disconnected_callback
            self.is_connected = False
            self.writes = []
            self.disconnects = 0
            self.block_writes = False
            self.write_cancelled = False
            instances.append(self)

        async def connect(self):
            self.is_connected = True

        async def write_gatt_char(self, uuid, data, response):
            assert uuid == COMMAND_UUID and response is True
            if self.block_writes:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.write_cancelled = True
            self.writes.append(Command.decode(data))

        async def read_gatt_char(self, uuid):
            assert uuid == STATUS_UUID
            sent = self.writes[-1]
            if status:
                return status(sent)
            return PACKET.pack(1, 17, sent.sequence, 0, 0, 0, 0, 0, 0)

        async def disconnect(self):
            self.disconnects += 1
            self.is_connected = False
            self.callback(self)

    monkeypatch.setitem(sys.modules, "bleak", SimpleNamespace(BleakScanner=Scanner, BleakClient=Client))
    return instances


def test_connect_confirms_stop_then_send_and_close_stop(config, monkeypatch):
    clients = fake_bleak(monkeypatch)
    async def scenario():
        link = BleLink(config)
        await link.connect()
        assert link.connected
        assert clients[0].writes[0].stop
        await link.send(Decision((0, 80, 0, 0, 0), "guiding"))
        await link.close()
        assert not link.connected and link.client is None
        assert clients[0].writes[-1].stop
        assert [command.sequence for command in clients[0].writes] == [0, 1, 2]
        assert clients[0].disconnects == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("count", [0, 2])
def test_scan_rejects_missing_or_ambiguous_receiver(config, monkeypatch, count):
    clients = fake_bleak(monkeypatch, devices=count)
    with pytest.raises(RuntimeError, match=f"receiver count={count}"):
        asyncio.run(BleLink(config).connect())
    assert not clients


def test_configured_address_resolves_multiple_receivers(config, monkeypatch):
    clients = fake_bleak(monkeypatch, devices=2)
    config["address"] = "aa:bb:cc:00:00:01"
    async def scenario():
        link = BleLink(config)
        await link.connect()
        assert clients[0].device.address == "AA:BB:CC:00:00:01"
        await link.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("flags,sequence_offset,remaining,levels", [
    (16, 0, 0, (0, 0, 0, 0, 0)),  # Not connected.
    (19, 0, 0, (0, 0, 0, 0, 0)),  # Still active despite STOP.
    (21, 0, 0, (0, 0, 0, 0, 0)),  # Expired flag.
    (25, 0, 0, (0, 0, 0, 0, 0)),  # Invalid command flag.
    (17, 1, 0, (0, 0, 0, 0, 0)),  # Unrelated sequence acknowledgement.
    (17, 0, 100, (0, 0, 0, 0, 0)),
    (17, 0, 0, (10, 0, 0, 0, 0)),
])
def test_handshake_rejects_unconfirmed_stop_and_disconnects(config, monkeypatch, flags, sequence_offset, remaining, levels):
    clients = fake_bleak(monkeypatch, status=lambda sent: PACKET.pack(
        1, flags, (sent.sequence + sequence_offset) & 65535, remaining, *levels))
    link = BleLink(config)
    with pytest.raises(RuntimeError, match="STOP state"):
        asyncio.run(link.connect())
    assert link.client is None and not link.connected
    assert clients[0].disconnects == 1


def test_late_disconnect_from_old_client_does_not_clear_new_connection(config, monkeypatch):
    clients = fake_bleak(monkeypatch)
    async def scenario():
        link = BleLink(config)
        await link.connect()
        old = clients[0]
        await link.close()
        await link.connect()
        old.callback(old)
        assert link.connected
        clients[1].callback(clients[1])
        assert not link.connected
        await link.close()
    asyncio.run(scenario())


def test_write_timeout_cancels_write_marks_disconnected_and_closes(config, monkeypatch):
    clients = fake_bleak(monkeypatch)
    async def scenario():
        link = BleLink(config)
        await link.connect()
        clients[0].block_writes = True
        with pytest.raises(asyncio.TimeoutError):
            await link.send(Decision((30, 0, 0, 0, 0), "guiding"))
        assert clients[0].write_cancelled
        assert not link.connected
        clients[0].block_writes = False
        await link.close()
        assert clients[0].writes[-1].stop
    asyncio.run(scenario())


@pytest.mark.parametrize("timestamp", [0.0, float("nan"), float("inf"), float("-inf"), "future"])
def test_transport_never_refreshes_watchdog_with_invalid_or_stale_decision(config, timestamp):
    async def scenario():
        stop, sent, changes = asyncio.Event(), [], []
        class Link:
            connected = False
            async def connect(self):
                self.connected = True
            async def send(self, decision):
                sent.append(decision)
                stop.set()
            async def close(self):
                self.connected = False
        generated = time.monotonic() + 10 if timestamp == "future" else timestamp
        await transmit_loop(config, lambda: (Decision((90, 0, 0, 0, 0), "guiding"), generated),
                            stop, changes.append, lambda settings: Link())
        assert changes == [True]
        assert len(sent) == 1 and sent[0].stopped and sent[0].reason == "stale_decision"
    asyncio.run(scenario())


def test_disconnect_notification_precedes_reconnect_and_old_target_is_not_resent(config):
    async def scenario():
        stop, events, sent = asyncio.Event(), [], []
        latest = [Decision((90, 0, 0, 0, 0), "guiding"), time.monotonic()]
        class Link:
            connected = False
            async def connect(self):
                events.append("connect")
                self.connected = True
            async def send(self, decision):
                sent.append(decision)
                if len(sent) == 1:
                    self.connected = False  # Simulated Bleak disconnect callback.
                else:
                    stop.set()
            async def close(self):
                events.append("close")
                self.connected = False
        def changed(connected):
            events.append(connected)
            latest[:] = [Decision(state="stopped", reason="link_change"), time.monotonic()]
        await transmit_loop(config, lambda: tuple(latest), stop, changed, lambda settings: Link())
        assert events[:6] == ["connect", True, False, "close", "connect", True]
        assert len(sent) == 2 and all(decision.stopped for decision in sent)
        assert events[-1] == "close"
    asyncio.run(scenario())


def test_transport_send_failure_closes_and_reconnects_before_retry(config):
    async def scenario():
        stop, changes, events = asyncio.Event(), [], []
        class Link:
            connected = False
            sends = 0
            async def connect(self):
                events.append("connect")
                self.connected = True
            async def send(self, decision):
                self.sends += 1
                if self.sends == 1:
                    raise ConnectionError("radio dropped")
                stop.set()
            async def close(self):
                events.append("close")
                self.connected = False
        await transmit_loop(config, lambda: (Decision(), time.monotonic()), stop,
                            changes.append, lambda settings: Link())
        assert changes == [True, False, True]
        assert events == ["connect", "close", "connect", "close"]
    asyncio.run(scenario())


def test_cancel_inflight_transport_write_still_closes_link(config):
    async def scenario():
        stop, writing = asyncio.Event(), asyncio.Event()
        closed = []
        class Link:
            connected = False
            async def connect(self):
                self.connected = True
            async def send(self, decision):
                writing.set()
                await asyncio.Event().wait()
            async def close(self):
                closed.append(True)
                self.connected = False
        task = asyncio.create_task(transmit_loop(config, lambda: (Decision(), time.monotonic()),
                                                 stop, lambda connected: None, lambda settings: Link()))
        await asyncio.wait_for(writing.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed == [True]
    asyncio.run(scenario())
