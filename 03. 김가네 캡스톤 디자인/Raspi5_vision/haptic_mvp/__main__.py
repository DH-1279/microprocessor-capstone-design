import argparse
import asyncio
import sys

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="김가네 온디바이스 햅틱 MVP")
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "doctor", "simulate"):
        command = sub.add_parser(name)
        command.add_argument("--config", help="JSON override file; unspecified settings use config/mvp.json")
        if name in ("run", "doctor"):
            command.add_argument("--ble", action="store_true", help="Connect real ESP32 haptic_receiver (default log only)")
            command.add_argument("--console", action="store_true", help="Use keyboard and printed speech instead of microphone/TTS")
        if name == "run":
            command.add_argument("--duration", type=float, help="Stop after specified seconds")
        if name == "doctor":
            command.add_argument("--deep", action="store_true", help="Import selected SDKs and check APIs in isolated 15-second subprocesses")
        if name == "simulate":
            command.add_argument("--report", help="Write scenario result JSON to this path")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if getattr(args, "ble", False):
            config["ble"]["dry_run"] = False
        if getattr(args, "console", False):
            config["audio"]["enabled"] = False
        if args.action == "doctor":
            from .doctor import doctor
            return doctor(config, deep=args.deep)
        if args.action == "simulate":
            from .simulation import simulate
            return simulate(config, args.report)
        if args.duration is not None and (args.duration <= 0 or not __import__("math").isfinite(args.duration)):
            parser.error("--duration must be a positive finite number")
        from .runtime import run
        asyncio.run(run(config, args.duration))
    except KeyboardInterrupt:
        print("\n중지했습니다.")
    except (Exception,) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
