"""Standalone read/STOP check. Never sends a motor-on command or opens a camera."""

import argparse
import asyncio
from time import monotonic

from .link import Link, STARTUP_IO_TIMEOUT


async def run(args):
    async with Link(args.address, scan_timeout=args.scan_timeout, connect_timeout=args.connect_timeout) as link:
        for index in range(args.samples):
            started = monotonic()
            status = await link.status(timeout=STARTUP_IO_TIMEOUT)
            print(
                f"상태 읽기 {index + 1}/{args.samples} | {monotonic() - started:.3f}초 | "
                f"flags=0x{status.flags:02X} seq={status.sequence} duty={status.duties}",
                flush=True,
            )
            if index + 1 < args.samples:
                await asyncio.sleep(0.25)
    print("BLE 통신 검사 종료 | 전송은 STOP만 사용했습니다.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="카메라 없이 BLE 연결·STOP·상태 읽기 검사")
    parser.add_argument("--address")
    parser.add_argument("--scan-timeout", type=float, default=10)
    parser.add_argument("--connect-timeout", type=float, default=30)
    parser.add_argument("--samples", type=int, choices=range(1, 11), default=3)
    args = parser.parse_args(argv)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass
    except Exception as error:
        parser.exit(1, f"BLE 통신 검사 실패: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
