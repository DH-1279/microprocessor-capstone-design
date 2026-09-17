"""Interactive Raspberry Pi/Linux terminal -> ESP32-C3 BLE LED test."""

import argparse
import asyncio
import sys
import time

from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

SERVICE_UUID = "328e0001-8c2a-4e58-9a48-512fc6ab1279"
LED_UUID = "328e0002-8c2a-4e58-9a48-512fc6ab1279"
COMMANDS = {"on": b"1", "1": b"1", "off": b"0", "0": b"0"}


async def find_device(address):
    print("KIMGANE-LED 검색 중 (8초)...")
    results = await BleakScanner.discover(timeout=8.0, return_adv=True)
    devices = [
        device
        for device, adv in results.values()
        if SERVICE_UUID in {uuid.lower() for uuid in (adv.service_uuids or [])}
        and (address is None or device.address.lower() == address.lower())
    ]
    if not devices:
        raise RuntimeError("장치를 찾지 못했습니다. ESP32 전원/펌웨어와 Pi Bluetooth를 확인하세요.")
    if len(devices) > 1:
        addresses = ", ".join(device.address for device in devices)
        raise RuntimeError(f"장치가 여러 개입니다: {addresses}\n--address 주소 로 선택하세요.")
    return devices[0]


async def read_state(client):
    state = bytes(await client.read_gatt_char(LED_UUID))
    if state not in (b"0", b"1"):
        raise RuntimeError(f"예상하지 못한 LED 상태: {state!r}")
    return state


async def set_led(client, value):
    started = time.perf_counter()
    await client.write_gatt_char(LED_UUID, value, response=True)
    state = await read_state(client)
    if state != value:
        raise RuntimeError(f"명령/상태 불일치: 요청={value!r}, 응답={state!r}")
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"LED {'ON' if state == b'1' else 'OFF'} 확인 | 쓰기+읽기 {elapsed_ms:.1f} ms")


async def read_command(disconnected):
    # Linux terminal/SSH: keep the BLE event loop running while waiting for input.
    loop = asyncio.get_running_loop()
    line = loop.create_future()

    def ready():
        if not line.done():
            line.set_result(sys.stdin.readline())

    print("명령 [on/off/status/quit]> ", end="", flush=True)
    loop.add_reader(sys.stdin.fileno(), ready)
    lost = asyncio.create_task(disconnected.wait())
    try:
        await asyncio.wait((line, lost), return_when=asyncio.FIRST_COMPLETED)
        if disconnected.is_set():
            raise RuntimeError("BLE 연결이 끊겼습니다. ESP32를 확인하고 다시 실행하세요.")
        return line.result().strip().lower() if line.result() else "quit"
    finally:
        loop.remove_reader(sys.stdin.fileno())
        lost.cancel()
        await asyncio.gather(lost, return_exceptions=True)
        if not line.done():
            line.cancel()


async def run(address):
    device = await find_device(address)
    disconnected = asyncio.Event()
    async with BleakClient(
        device, disconnected_callback=lambda _: disconnected.set()
    ) as client:
        print(f"연결됨: {device.name} ({device.address})")
        try:
            await set_led(client, b"0")
            while True:
                command = await read_command(disconnected)
                if command in ("quit", "q", "exit"):
                    break
                if command in COMMANDS:
                    await set_led(client, COMMANDS[command])
                elif command == "status":
                    state = await read_state(client)
                    print(f"LED {'ON' if state == b'1' else 'OFF'}")
                else:
                    print("on, off, status, quit 중 하나를 입력하세요.")
        finally:
            if client.is_connected:
                try:
                    await asyncio.wait_for(set_led(client, b"0"), timeout=2.0)
                except (BleakError, RuntimeError, OSError, asyncio.TimeoutError) as error:
                    print(f"종료 전 OFF 전송 실패: {error}", file=sys.stderr)
    print("연결 종료")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", help="여러 보드가 검색될 때 대상 BLE 주소")
    args = parser.parse_args()
    if sys.platform != "linux" or not sys.stdin.isatty():
        parser.error("라즈베리파이의 터미널 또는 대화형 SSH 터미널에서 실행하세요.")
    try:
        asyncio.run(run(args.address))
    except KeyboardInterrupt:
        print("\n사용자가 종료했습니다.")
    except (BleakError, RuntimeError, OSError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
