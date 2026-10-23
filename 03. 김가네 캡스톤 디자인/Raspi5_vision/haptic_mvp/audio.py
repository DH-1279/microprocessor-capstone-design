"""Local Korean voice input/output. Importing this module requires only stdlib.

Hardware audio is lazy-loaded by start(). All inference/playback is performed in
workers, so callers must hand on_text to a thread-safe command queue. This is
half-duplex audio: speech input is discarded during TTS and its echo cooldown.
"""

from __future__ import annotations

from array import array
from collections import deque
import hashlib
import importlib
import math
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable
import wave


DEFAULTS = {
    "enabled": False,
    "whisper_executable": "whisper-cli",
    "whisper_model": "",
    "whisper_threads": 2,
    "stt_timeout_s": 30.0,
    "tts_executable": "espeak-ng",
    "playback_executable": "aplay",
    "tts_voice": "ko",
    "tts_rate": 155,
    "tts_timeout_s": 20.0,
    "cache_dir": "~/.cache/kimgane-haptic/tts",
    "input_device": None,
    "output_device": None,
    "sample_rate": 16000,
    "block_ms": 20,
    "rms_threshold": 0.015,
    "min_speech_ms": 200,
    "silence_ms": 650,
    "max_utterance_ms": 5000,
    "pre_roll_ms": 200,
    "echo_cooldown_ms": 300,
}


def audio_config(values: dict) -> dict:
    """Validate timing before opening hardware or allocating audio buffers."""
    cfg = {**DEFAULTS, **values}
    if not isinstance(cfg["enabled"], bool):
        raise ValueError("audio.enabled must be true or false")
    if cfg["sample_rate"] != 16000:
        raise ValueError("audio.sample_rate must be 16000 for whisper-cli")
    bounds = {
        "block_ms": (10, 100), "rms_threshold": (0.0001, 0.9),
        "min_speech_ms": (40, 2000), "silence_ms": (100, 2000),
        "max_utterance_ms": (500, 15000), "pre_roll_ms": (0, 1000),
        "echo_cooldown_ms": (0, 2000), "whisper_threads": (1, 4),
        "stt_timeout_s": (1, 120), "tts_timeout_s": (1, 120),
        "tts_rate": (80, 300),
    }
    for key, (low, high) in bounds.items():
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"audio.{key} must be numeric")
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"audio.{key} must be between {low} and {high}")
    for key in ("block_ms", "whisper_threads", "tts_rate"):
        if int(cfg[key]) != cfg[key]:
            raise ValueError(f"audio.{key} must be an integer")
        cfg[key] = int(cfg[key])
    if cfg["min_speech_ms"] + cfg["silence_ms"] > cfg["max_utterance_ms"]:
        raise ValueError("audio max_utterance_ms must allow speech and trailing silence")
    model_name = Path(cfg["whisper_model"]).name.lower()
    if ".en." in model_name or ".en-" in model_name or model_name.endswith(".en"):
        raise ValueError("Korean STT requires a multilingual model, not an .en model")
    return cfg


def pcm16_rms(pcm: bytes) -> float:
    """RMS in [0, 1] for mono little-endian signed PCM16, without audioop/numpy."""
    if len(pcm) % 2:
        raise ValueError("PCM16 data must have an even byte count")
    if not pcm:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    return math.sqrt(sum(int(x) * int(x) for x in samples) / len(samples)) / 32768


class UtteranceDetector:
    """Simple energy VAD. Bounds utterances; requires voiced duration, not noise gaps.

    This is a prototype energy gate, not a trained speech classifier. Tune its
    threshold with the actual microphone, and validate STT against known commands.
    """

    def __init__(self, config: dict):
        self.config = audio_config(config)
        self.reset()

    def reset(self) -> None:
        self.active = False
        self._pre: deque[bytes] = deque()
        self._pre_samples = 0
        self._parts: list[bytes] = []
        self._samples = 0
        self._voiced = 0
        self._silence = 0

    def feed(self, pcm: bytes, *, muted: bool = False) -> bytes | None:
        if muted:
            self.reset()
            return None
        rms = pcm16_rms(pcm)
        samples = len(pcm) // 2
        if not samples:
            return None
        cfg = self.config
        loud = rms >= cfg["rms_threshold"]
        per_ms = cfg["sample_rate"] / 1000
        if not self.active:
            if not loud:
                self._pre.append(pcm)
                self._pre_samples += samples
                limit = int(cfg["pre_roll_ms"] * per_ms)
                while self._pre and self._pre_samples > limit:
                    self._pre_samples -= len(self._pre.popleft()) // 2
                return None
            self.active = True
            self._parts = list(self._pre)
            self._samples = self._pre_samples
            self._pre.clear()
            self._pre_samples = 0
        self._parts.append(pcm)
        self._samples += samples
        self._voiced += samples if loud else 0
        self._silence = 0 if loud else self._silence + samples
        max_samples = int(cfg["max_utterance_ms"] * per_ms)
        finished = (self._silence >= cfg["silence_ms"] * per_ms
                    or self._samples >= max_samples)
        if not finished:
            return None
        result = None
        if self._voiced >= cfg["min_speech_ms"] * per_ms:
            result = b"".join(self._parts)[:max_samples * 2]
        self.reset()
        return result


def build_whisper_command(config: dict, wav_path: Path, output_base: Path) -> list[str]:
    cfg = audio_config(config)
    return [str(Path(cfg["whisper_executable"]).expanduser()),
            "-m", str(Path(cfg["whisper_model"]).expanduser()),
            "-l", "ko", "-t", str(cfg["whisper_threads"]),
            "-f", str(wav_path), "-otxt", "-of", str(output_base), "-nt", "-np"]


def tts_cache_path(config: dict, text: str) -> Path:
    key = f"espeak-ng\0{config['tts_voice']}\0{config['tts_rate']}\0{text}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return Path(config["cache_dir"]).expanduser() / f"{digest}.wav"


class _Cancelled(Exception):
    pass


class AudioService:
    """Two bounded workers for capture/STT and TTS, with explicit error reporting.

    say() returns False if user speech, transcription or another announcement is
    active; callers should retry important messages, not queue repeated summaries.
    listening means a user utterance/transcription is active, not merely mic open.
    get_error() consumes one diagnostic, or returns None immediately.
    """

    def __init__(self, config: dict, on_text: Callable[[str], None]):
        self.config = audio_config(config)
        self.on_text = on_text
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pcm: queue.Queue[tuple[float, bytes]] = queue.Queue(maxsize=25)
        self._speech: queue.Queue[str] = queue.Queue(maxsize=1)
        self._errors: queue.Queue[str] = queue.Queue(maxsize=8)
        self._threads: list[threading.Thread] = []
        self._processes: set[subprocess.Popen] = set()
        self._stream: Any = None  # sounddevice is an optional, lazy-loaded native dependency.
        self._started = False
        self._speaking = False
        self._listening = False
        self._transcribing = False
        self._input_generation = 0
        self._mute_until = 0.0
        self._vad = UtteranceDetector(self.config)

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._speaking

    @property
    def listening(self) -> bool:
        with self._lock:
            return self._listening or self._transcribing

    def get_error(self) -> str | None:
        try:
            return self._errors.get_nowait()
        except queue.Empty:
            return None

    def discard_input(self) -> None:
        """Invalidate queued/in-flight voice requests after a BLE session change.

        Call this before clearing the main command mailbox. on_text must only
        enqueue its string and must not call back into AudioService.
        """
        with self._lock:
            self._input_generation += 1
            self._mute_until = max(self._mute_until, time.monotonic())
            self._vad.reset()
            self._listening = False
            while True:
                try:
                    self._pcm.get_nowait()
                except queue.Empty:
                    break

    def _report(self, message: str) -> None:
        try:
            self._errors.put_nowait(message)
        except queue.Full:
            try:
                self._errors.get_nowait()
            except queue.Empty:
                pass
            try:
                self._errors.put_nowait(message)
            except queue.Full:
                pass

    def start(self) -> None:
        if self._started:
            return
        if self._stop.is_set():
            raise RuntimeError("AudioService has been closed; create a new instance")
        if not self.config["enabled"]:
            return
        for key in ("whisper_executable", "tts_executable", "playback_executable"):
            binary = str(Path(self.config[key]).expanduser())
            resolved = shutil.which(binary)
            if resolved is None:
                raise RuntimeError(f"Audio executable missing: {key}={binary}")
            self.config[key] = resolved
        model = Path(self.config["whisper_model"]).expanduser()
        if not model.is_file() or model.stat().st_size == 0:
            raise RuntimeError(f"Whisper multilingual model missing or empty: {model}")
        self.config["whisper_model"] = str(model.resolve())
        Path(self.config["cache_dir"]).expanduser().mkdir(parents=True, exist_ok=True)
        try:
            sd = importlib.import_module("sounddevice")
        except ImportError as exc:
            raise RuntimeError("Install sounddevice and the system libportaudio2 package") from exc
        try:
            self._stream = sd.RawInputStream(
                samplerate=self.config["sample_rate"], channels=1, dtype="int16",
                blocksize=self.config["sample_rate"] * self.config["block_ms"] // 1000,
                device=self.config["input_device"], callback=self._capture,
                finished_callback=self._input_finished,
            )
            self._stream.start()
            self._started = True
            for name, target in (("audio-stt", self._listen_worker), ("audio-tts", self._speak_worker)):
                worker = threading.Thread(target=target, name=name, daemon=True)
                self._threads.append(worker)
                worker.start()
        except Exception:
            self.close()
            raise

    def say(self, text: str) -> bool:
        text = text.strip()
        if not self._started or self._stop.is_set() or not text:
            return False
        if len(text) > 300:
            raise ValueError("Announcement exceeds the 300-character limit")
        with self._lock:
            if self._speaking or self._listening or self._transcribing:
                return False
            self._speaking = True
            try:
                self._speech.put_nowait(text)
            except queue.Full:
                self._speaking = False
                return False
        return True

    def _muted(self, timestamp: float) -> bool:
        return self._speaking or self._transcribing or timestamp < self._mute_until

    def _capture(self, indata, frames, timing, status) -> None:
        if self._stop.is_set():
            return
        if status:
            self._report(f"Microphone stream: {status}")
        now = time.monotonic()
        with self._lock:
            if self._muted(now):
                return
        try:
            self._pcm.put_nowait((now, bytes(indata)))
        except queue.Full:
            self._report("Microphone queue overflow; stale audio will be discarded")

    def _input_finished(self) -> None:
        if not self._stop.is_set():
            self._report("Microphone stream stopped unexpectedly; check the USB device")
            self._stop.set()

    def _listen_worker(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    timestamp, pcm = self._pcm.get(timeout=0.1)
                except queue.Empty:
                    continue
                with self._lock:
                    muted = self._muted(timestamp) or time.monotonic() - timestamp > 0.5
                    utterance = self._vad.feed(pcm, muted=muted)
                    self._listening = self._vad.active
                    if utterance is not None:
                        self._transcribing = True
                        generation = self._input_generation
                if utterance is None:
                    continue
                try:
                    with tempfile.TemporaryDirectory(prefix="kimgane-stt-") as directory:
                        wav_path = Path(directory) / "utterance.wav"
                        with wave.open(str(wav_path), "wb") as output:
                            output.setnchannels(1)
                            output.setsampwidth(2)
                            output.setframerate(16000)
                            output.writeframes(utterance)
                        text = self.transcribe_wav(wav_path)
                    with self._lock:
                        if (text and not self._stop.is_set()
                                and generation == self._input_generation):
                            self.on_text(text)
                finally:
                    with self._lock:
                        self._transcribing = False
                        self._listening = False
                        self._vad.reset()
                        self._mute_until = max(self._mute_until, time.monotonic())
        except _Cancelled:
            pass
        except Exception as exc:
            self._report(f"Audio input/STT failed: {exc}")
            self._stop.set()

    def transcribe_wav(self, wav_path: str | Path) -> str:
        """Offline microphone check: accepts mono PCM16/16kHz WAV <=15 seconds."""
        wav_path = Path(wav_path).expanduser().resolve()
        with wave.open(str(wav_path), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
                raise ValueError("STT WAV must be mono, 16-bit PCM, 16000 Hz")
            if not 0 < source.getnframes() <= 15 * 16000:
                raise ValueError("STT WAV must contain between 0 and 15 seconds of audio")
        with tempfile.TemporaryDirectory(prefix="kimgane-transcript-") as directory:
            output_base = Path(directory) / "transcript"
            command = build_whisper_command(self.config, wav_path, output_base)
            self._run(command, self.config["stt_timeout_s"])
            transcript = output_base.with_suffix(".txt")
            if not transcript.is_file():
                raise RuntimeError("whisper-cli completed without its expected transcript file")
            text = transcript.read_text(encoding="utf-8-sig").strip()
            return " ".join(text.split())

    def _speak_worker(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    text = self._speech.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._play(text)
                finally:
                    with self._lock:
                        self._speaking = False
                        self._mute_until = time.monotonic() + self.config["echo_cooldown_ms"] / 1000
        except _Cancelled:
            pass
        except Exception as exc:
            self._report(f"Audio output/TTS failed: {exc}")
            self._stop.set()

    def _play(self, text: str) -> None:
        cfg = self.config
        target = tts_cache_path(cfg, text)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            # A unique temporary file plus replace avoids leaving a corrupt cache entry.
            with tempfile.TemporaryDirectory(prefix="tts-", dir=target.parent) as directory:
                temporary = Path(directory) / "speech.wav"
                command = [cfg["tts_executable"], "-b", "1", "-v", cfg["tts_voice"],
                           "-s", str(cfg["tts_rate"]), "-w", str(temporary), "--stdin"]
                self._run(command, cfg["tts_timeout_s"], input_text=text)
                with wave.open(str(temporary), "rb") as wav:
                    if wav.getnframes() == 0:
                        raise RuntimeError("TTS generated an empty WAV")
                temporary.replace(target)
        command = [cfg["playback_executable"], "-q"]
        if cfg["output_device"] is not None:
            command.extend(["-D", str(cfg["output_device"])])
        command.append(str(target))
        self._run(command, cfg["tts_timeout_s"])

    def _run(self, command: list[str], timeout: float, input_text: str | None = None) -> str:
        """Cancellable child process. No shell interpolation or leaked children."""
        with self._lock:
            if self._stop.is_set():
                raise _Cancelled()
            process = subprocess.Popen(command, stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, encoding="utf-8", errors="replace")
            self._processes.add(process)
        deadline = time.monotonic() + timeout
        try:
            pending_input = input_text
            while True:
                if self._stop.is_set():
                    raise _Cancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{Path(command[0]).name} exceeded {timeout:g}s")
                try:
                    stdout, stderr = process.communicate(input=pending_input, timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pending_input = None
            if self._stop.is_set():
                raise _Cancelled()
            if process.returncode:
                detail = (stderr or stdout).strip()[-1000:]
                raise RuntimeError(f"{Path(command[0]).name} exited {process.returncode}: {detail}")
            return stdout
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()
            with self._lock:
                self._processes.discard(process)

    def close(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.abort()
            except Exception as exc:
                self._report(f"Microphone stop failed: {exc}")
            finally:
                try:
                    self._stream.close()
                except Exception as exc:
                    self._report(f"Microphone close failed: {exc}")
                self._stream = None
        with self._lock:
            for process in self._processes:
                if process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
        for worker in self._threads:
            if worker is not threading.current_thread():
                worker.join(timeout=2)
                if worker.is_alive():
                    self._report(f"Audio worker did not stop: {worker.name}")
        with self._lock:
            self._speaking = self._listening = self._transcribing = False
        self._started = False
