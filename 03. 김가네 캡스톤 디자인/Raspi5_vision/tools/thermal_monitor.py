#!/usr/bin/env python3
"""Pi CPU + Hailo-10H + active fan dashboard; standard library only.

HailoRT 5.1.1 reference:
https://github.com/hailo-ai/hailort/blob/v5.1.1/hailort/libhailort/include/hailo/device.hpp
Health stats report die temperature, not the HAT PCB surface temperature.
The embedded helper uses the installed C++ SDK and never configures a network.
"""

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata


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


@dataclass(frozen=True)
class FanReading:
    rpm: int
    pwm: int | None = None


def read_fan(root=Path("/sys/class/hwmon")):
    """Read the named Pi PWM fan, without changing kernel fan control."""
    devices = []
    for device in sorted(root.glob("hwmon*")):
        try:
            if (device / "name").read_text().strip() in ("pwmfan", "pwm-fan"):
                devices.append(device)
        except OSError:
            continue
    if len(devices) != 1:
        raise RuntimeError("PWM fan not found" if not devices else "Multiple PWM fans found")
    rpm = int((devices[0] / "fan1_input").read_text().strip())
    if rpm < 0:
        raise ValueError("Invalid fan RPM")
    # PWM is optional; RPM must still be available to claim rotation.
    pwm: int | None
    try:
        pwm = int((devices[0] / "pwm1").read_text().strip())
        if not 0 <= pwm <= 255:
            pwm = None
    except (OSError, ValueError):
        pwm = None
    return FanReading(rpm, pwm)


def fan_status(fan):
    if fan is None:
        return "FAN N/A", "33"
    if fan.rpm > 0:
        return f"FAN RUNNING {fan.rpm} RPM", "32"
    if fan.pwm:
        return "FAN CHECK 0 RPM (PWM ON)", "33"
    return "FAN STOPPED 0 RPM", "36"


# Only generated SGR color codes are preserved; error text is sanitized in render.
_SGR = re.compile(r"\033\[[0-9;]*m")


def fit_line(line, columns):
    """Clip by terminal cells, preserving colors without wrapping wide text."""
    result, used, index = [], 0, 0
    while index < len(line):
        match = _SGR.match(line, index)
        if match:
            result.append(match.group())
            index = match.end()
            continue
        char = line[index]
        width = 0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if used + width > columns:
            break
        if char.isprintable():
            result.append(char)
            used += width
        index += 1
    return "".join(result) + ("\033[0m" if "\033" in line else "")


class TerminalScreen:
    """Use the alternate screen, absolute row writes and no scrolling newlines."""

    def __init__(self, stream=None):
        self.stream = sys.stdout if stream is None else stream

    def __enter__(self):
        self.stream.write("\033[?1049h\033[?25l\033[2J\033[H")
        self.stream.flush()
        return self

    def draw(self, frame):
        size = shutil.get_terminal_size(fallback=(80, 24))
        # Leave the final column and row unused to avoid terminal autowrap/scroll.
        width, height = max(0, size.columns - 1), max(0, size.lines - 1)
        output = ["\033[H"]
        for row, line in enumerate(frame.splitlines()[:height], 1):
            output.append(f"\033[{row};1H\033[2K" + fit_line(line, width))
        output.append("\033[J")
        self.stream.write("".join(output))
        self.stream.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.write("\033[0m\033[?25h\033[?1049l")
        self.stream.flush()


def bar(value, width=26):
    count = round(min(100, max(0, value)) / 100 * width)
    return "█" * count + "░" * (width - count)


def render(values, peaks, errors, color=True, demo=False, fan=None, columns=80):
    def paint(text, code):
        return f"\033[{code}m{text}\033[0m" if color else text

    border_width = max(24, min(100, columns - 1))
    title = "── KIMGANE · THERMAL MONITOR "
    lines = [paint("╭" + title + "─" * max(0, border_width - len(title) - 2) + "╮", "1;36"),
             "  Pi 5 CPU  /  AI HAT+ 2 · Hailo-10H  /  Active Cooler",
             f"  {time.strftime('%Y-%m-%d %H:%M:%S')}   " + ("DEMO · SIMULATED" if demo else "LIVE · SENSOR READINGS"), ""]
    fan_text, fan_color = fan_status(fan)
    inline_fan = columns >= 76
    bar_width = 20 if inline_fan else max(4, min(26, columns - 23))
    for label, value in zip(("CPU", "HAILO"), values):
        if value is None:
            reading = paint(f"  {label:<5}  N/A · query failed", "33")
        else:
            # Colors are display bands, not manufacturer thermal limits.
            code = "32" if value < 60 else "33" if value < 80 else "31"
            reading = paint(f"  {label:<5} {value:5.1f} °C  {bar(value, bar_width)}", code)
        if label == "CPU" and inline_fan:
            reading += "  |  " + paint(fan_text, fan_color)
        lines.append(reading)
        if value is not None:
            lines.append(f"        session max {peaks[label]:.1f} °C")
        if label == "CPU" and not inline_fan:
            lines.append("  " + paint(fan_text, fan_color))
    lines += ["", "  Bar scale: 0–100 °C · colors are display bands",
              "  HAILO: chip die temperature · Ctrl+C to exit"]
    for error in errors:
        # Do not allow subprocess text to inject terminal controls.
        clean = " ".join("".join(c if c.isprintable() else " " for c in error).split())
        lines.append("  " + clean[:140])
    lines.append(paint("╰" + "─" * (border_width - 2) + "╯", "36"))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU + Hailo-10H temperature and active fan dashboard")
    parser.add_argument("--interval", type=float, default=2, help="Refresh seconds (minimum 1)")
    parser.add_argument("--once", action="store_true", help="Print one sample and exit")
    parser.add_argument("--demo", action="store_true", help="Show simulated values without hardware")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--screen", action="store_true", help="Force ANSI screen refresh (e.g. SSH terminal)")
    mode.add_argument("--log", action="store_true", help="Append samples instead of refreshing the screen")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error("--interval must be a finite number >= 1")
    interactive = (args.screen or sys.stdout.isatty()) and not (args.once or args.log)
    once = args.once or (not interactive and not args.log)
    peaks = {"CPU": -math.inf, "HAILO": -math.inf}
    try:
        workspace = nullcontext(None) if args.demo else tempfile.TemporaryDirectory(prefix="kimgane-thermal-")
        with workspace as tmp, (TerminalScreen() if interactive else nullcontext(None)) as screen:
            binary, setup_error = None, None
            if not args.demo:
                if screen is not None:
                    screen.draw("Preparing Hailo temperature reader...")
                else:
                    print("Preparing Hailo temperature reader...", flush=True)
                try:
                    assert tmp is not None
                    binary = build_helper(Path(tmp))
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    setup_error = str(error)
            while True:
                started = time.monotonic()
                values: list[float | None] = []
                errors: list[str] = []
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
                fan = None
                try:
                    fan = FanReading(2726, 75) if args.demo else read_fan()
                except (OSError, ValueError, RuntimeError) as error:
                    errors.append(f"FAN: {error}")
                columns = shutil.get_terminal_size(fallback=(80, 24)).columns
                frame = render(values, peaks, errors, color=interactive, demo=args.demo, fan=fan, columns=columns)
                if screen is not None:
                    screen.draw(frame)
                else:
                    print(frame, flush=True)
                if once:
                    return 1 if errors else 0
                time.sleep(max(0, args.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
