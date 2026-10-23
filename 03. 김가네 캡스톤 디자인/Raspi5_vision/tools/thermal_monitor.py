#!/usr/bin/env python3
"""Pi CPU + Hailo-10H terminal dashboard; Python standard library only.

HailoRT 5.1.1 reference:
https://github.com/hailo-ai/hailort/blob/v5.1.1/hailort/libhailort/include/hailo/device.hpp
Health stats report die temperature, not the HAT PCB surface temperature.
The embedded helper uses the installed C++ SDK and never configures a network.
"""

import argparse
from contextlib import nullcontext
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


HELPER = r'''
#include <hailo/device.hpp>
#include <cmath>
#include <iostream>

int main() {
    auto created = hailort::Device::create();
    if (!created) {
        std::cerr << "Device open failed: status=" << created.status();
        return 1;
    }
    auto device = created.release();
    auto stats = device->query_health_stats();
    if (!stats) {
        std::cerr << "Health query failed: status=" << stats.status();
        return 2;
    }
    const float temperature = stats.value().on_die_temperature;
    // The SDK documents -1 as an unavailable-field sentinel.
    if (temperature == -1 || !std::isfinite(temperature)) {
        std::cerr << "Die temperature unavailable";
        return 3;
    }
    std::cout << temperature << std::endl;
    return 0;
}
'''


def temperature(value):
    result = float(value)
    if not math.isfinite(result) or not -40 <= result <= 150 or result == -1:
        raise ValueError("invalid or unavailable temperature")
    return result


def build_helper(directory):
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ missing: install with sudo apt install g++")
    source = directory / "hailo_temperature.cpp"
    binary = directory / "hailo_temperature"
    source.write_text(HELPER, encoding="utf-8")
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-lhailort", "-o", str(binary)],
        capture_output=True, text=True, timeout=45, check=False,
    )
    if result.returncode:
        raise RuntimeError("Hailo SDK build failed (headers/library required): " + result.stderr.strip()[-1200:])
    return binary


def read_cpu():
    # Prefer the named CPU zone; never treat an arbitrary device sensor as CPU.
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            kind = (zone / "type").read_text().strip()
            if kind in ("cpu-thermal", "cpu_thermal", "soc-thermal"):
                return temperature(float((zone / "temp").read_text()) / 1000)
        except (OSError, ValueError):
            continue
    result = subprocess.run(
        ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "CPU temperature query failed")
    return temperature(result.stdout.strip().split("=", 1)[1].split("'", 1)[0])


def read_hailo(binary):
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=4, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"Hailo query exited {result.returncode}")
    return temperature(result.stdout.strip())


def bar(value, width=26):
    count = round(min(100, max(0, value)) / 100 * width)
    return "█" * count + "░" * (width - count)


def render(values, peaks, errors, color=True, demo=False):
    def paint(text, code):
        return f"\033[{code}m{text}\033[0m" if color else text

    lines = [paint("╭── KIMGANE · THERMAL MONITOR ──────────────────────╮", "1;36"),
             "  Pi 5 CPU  /  AI HAT+ 2 · Hailo-10H",
             f"  {time.strftime('%Y-%m-%d %H:%M:%S')}   " + ("DEMO · SIMULATED" if demo else "LIVE · SENSOR READINGS"), ""]
    for label, value in zip(("CPU", "HAILO"), values):
        if value is None:
            lines.append(paint(f"  {label:<5}  N/A · query failed", "33"))
            continue
        # Colors are display bands, not manufacturer thermal limits.
        code = "32" if value < 60 else "33" if value < 80 else "31"
        lines.append(paint(f"  {label:<5} {value:5.1f} °C  {bar(value)}", code))
        lines.append(f"        session max {peaks[label]:.1f} °C")
    lines += ["", "  Bar scale: 0–100 °C · colors are display bands",
              "  HAILO: chip die temperature · Ctrl+C to exit"]
    for error in errors:
        # Do not allow subprocess text to inject terminal controls.
        clean = " ".join("".join(c if c.isprintable() else " " for c in error).split())
        lines.append("  " + clean[:140])
    lines.append(paint("╰──────────────────────────────────────────────────╯", "36"))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU + Hailo-10H temperature dashboard")
    parser.add_argument("--interval", type=float, default=2, help="Refresh seconds (minimum 1)")
    parser.add_argument("--once", action="store_true", help="Print one sample and exit")
    parser.add_argument("--demo", action="store_true", help="Show simulated values without hardware")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error("--interval must be a finite number >= 1")
    interactive = sys.stdout.isatty() and not args.once
    peaks = {"CPU": -math.inf, "HAILO": -math.inf}
    try:
        workspace = nullcontext(None) if args.demo else tempfile.TemporaryDirectory(prefix="kimgane-thermal-")
        with workspace as tmp:
            binary, setup_error = None, None
            if not args.demo:
                print("Preparing Hailo temperature reader...", flush=True)
                try:
                    binary = build_helper(Path(tmp))
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    setup_error = str(error)
            if interactive:
                print("\033[2J\033[H\033[?25l", end="", flush=True)
            while True:
                started = time.monotonic()
                values, errors = [], []
                for label in ("CPU", "HAILO"):
                    try:
                        if args.demo:
                            value = 46.5 if label == "CPU" else 54.2
                        elif label == "CPU":
                            value = read_cpu()
                        elif binary is None:
                            raise RuntimeError(setup_error)
                        else:
                            value = read_hailo(binary)
                        peaks[label] = max(peaks[label], value)
                        values.append(value)
                    except (OSError, RuntimeError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
                        values.append(None)
                        errors.append(f"{label}: {error}")
                frame = render(values, peaks, errors, color=interactive, demo=args.demo)
                print(("\033[H\033[J" if interactive else "") + frame, flush=True)
                if args.once:
                    return 1 if errors else 0
                time.sleep(max(0, args.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        return 0
    finally:
        if interactive:
            print("\033[0m\033[?25h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
