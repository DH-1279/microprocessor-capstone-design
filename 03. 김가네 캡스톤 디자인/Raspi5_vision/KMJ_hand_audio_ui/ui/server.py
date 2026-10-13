"""Optional web status page. Runs beside the main loop and never blocks it.

The browser reads the same Decision the BLE transport sends, through the same
`lambda: tuple(latest)` contract, so the page cannot show a state the glove is
not in. Commands go back through the caller's on_text sink, which means voice,
keyboard and buttons share one path into the controller.

The page is a secondary surface: if it fails to start, disconnects, or is never
opened, guidance continues unchanged. Web dependencies are imported lazily so
the rest of the runtime still works without them.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from .. import speech as intents

STATIC_DIR = Path(__file__).resolve().parent / "static"
PUSH_INTERVAL_S = 0.1          # 10 Hz, inside the 5-10 Hz the UX spec asks for.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# Which buttons the page may offer. Derived from the controller's own state and
# reason, not from UI text, so an unreachable action is never shown as ready.
ACTIONS = {
    ("scanning", None):           ("select_target", "list"),
    ("guiding", "moving"):        ("list", "repeat"),
    ("guiding", "near"):          ("done", "list", "repeat"),
    ("paused", "hand_lost"):      ("repeat",),
    ("paused", "target_lost"):    ("select_target", "repeat"),
    ("paused", "stale_frame"):    ("repeat",),
    ("paused", "ble_disconnected"): ("repeat",),
    ("stopped", None):            ("select_target", "list"),
}

# Wording follows the 2026-10-07 prototype so the page and the design review
# use the same words.
STATE_TEXT = {
    "scanning": "준비됐어요",
    "guiding": "손을 안내하고 있어요",
    "paused": "안내를 잠시 멈췄어요",
    "stopped": "안내를 멈췄어요",
}

# Short, plain explanations. Technical reasons stay in the inspection panel.
REASON_TEXT = {
    "idle": "찾을 물건을 말하거나 아래에서 골라 주세요. 같은 종류가 여러 개면 위치를 고를 수 있어요.",
    "moving": "진동을 따라 손을 움직여 주세요.",
    "near": "영상에서 목표와 가까워졌어요. 손으로 확인해 주세요. 찾았다면 찾았어요를 눌러 주세요.",
    "hand_lost": "손을 확인하지 못해 안내를 멈췄어요. 손을 시작 영역에 두어 주세요.",
    "target_lost": "목표를 놓쳤어요. 다시 골라 주세요.",
    "stale_frame": "영상이 갱신되지 않아 멈췄어요.",
    "ble_disconnected": "장갑 연결이 끊겼어요. 연결 후 물건을 다시 선택해 주세요.",
    "link_change": "연결 상태가 바뀌었어요. 다시 골라 주세요.",
    "user_stop": "안내를 멈췄어요.",
    "shutdown": "프로그램을 종료하는 중이에요.",
}

FINGERS = ("엄지", "검지", "중지", "약지", "소지")


def allowed_actions(state: str, reason: str) -> tuple[str, ...]:
    """Stop is always offered; the spec requires it in every state."""
    actions = ACTIONS.get((state, reason)) or ACTIONS.get((state, None)) or ()
    return actions + ("stop",)


class Bridge:
    """Turns one Decision into the page contract and validates commands back."""

    def __init__(self, decision_source, on_text, targets):
        self.decision_source = decision_source
        self.on_text = on_text
        self.targets = targets
        self.revision = 0
        self._signature = None
        self._seen_commands: set[str] = set()
        self._lock = threading.Lock()

    def _signature_of(self, decision):
        return (decision.state, decision.reason, decision.target,
                tuple(decision.levels), decision.direction, decision.proximity)

    def snapshot(self) -> dict:
        decision, decided_at = self.decision_source()
        with self._lock:
            signature = self._signature_of(decision)
            if signature != self._signature:
                self._signature = signature
                self.revision += 1
            revision = self.revision

        target = decision.target
        item = self.targets.get(target) if target else None
        return {
            "revision": revision,
            "state": decision.state,
            "reason": decision.reason,
            "state_text": STATE_TEXT.get(decision.state, decision.state),
            "message": REASON_TEXT.get(decision.reason, ""),
            "target": target,
            "target_text": (item or {}).get("name", target or ""),
            "direction": decision.direction,
            "proximity": decision.proximity,
            "allowed_actions": list(allowed_actions(decision.state,
                                                    decision.reason)),
            "output_requested": list(decision.levels),
            "objects": [{"id": label, "name": item.get("name", label)}
                        for label, item in self.targets.items()],
            "age_s": round(max(0.0, time.monotonic() - decided_at), 2),
        }

    def handle(self, payload: dict) -> dict:
        """
        Accept a UI command, or say why it was refused.

        The controller re-checks everything; these checks only stop the page
        from replaying an old or duplicated request.
        """
        raw = str(payload.get("intent") or "").strip()
        if not raw:
            return {"accepted": False, "reason": "empty_intent"}

        command_id = payload.get("command_id")
        if command_id:
            with self._lock:
                if command_id in self._seen_commands:
                    return {"accepted": True, "reason": "duplicate_ignored"}
                self._seen_commands.add(command_id)
                if len(self._seen_commands) > 500:
                    self._seen_commands.clear()

        intent = intents.parse_input(raw, self.targets)
        is_stop = intent.kind == "control" and intent.name == "stop"

        # A late start must not restart guidance the user already stopped.
        # Stop itself is never refused for being late.
        observed = payload.get("revision")
        if observed is not None and not is_stop:
            if observed != self.snapshot()["revision"]:
                return {"accepted": False, "reason": "stale_revision"}

        if not intent:
            return {"accepted": False, "reason": intent.reason}

        # The controller parses text, so send the spoken form of the button.
        self.on_text(intents.BUTTON_ALIASES.get(raw, raw)
                     if intent.kind != "target" else
                     (self.targets[intent.name].get("aliases") or
                      [intent.name])[0])
        return {"accepted": True, "reason": "sent"}


def build_app(bridge: Bridge):
    import asyncio

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="haptic glove status")

    @app.get("/")
    def index():
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            return JSONResponse({"error": f"missing {page}"}, status_code=500)
        return FileResponse(page)

    @app.get("/api/state")
    def state():
        return bridge.snapshot()

    @app.post("/api/command")
    def command(payload: dict):
        return bridge.handle(payload)

    @app.websocket("/ws")
    async def stream(socket: WebSocket):
        await socket.accept()
        last = None

        async def read():
            while True:
                raw = await socket.receive_text()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                result = bridge.handle(payload)
                await socket.send_text(json.dumps(
                    {"ack": result, **bridge.snapshot()}, ensure_ascii=False))

        reader = asyncio.create_task(read())
        try:
            while not reader.done():
                snapshot = bridge.snapshot()
                if snapshot["revision"] != last:
                    last = snapshot["revision"]
                    await socket.send_text(json.dumps(snapshot,
                                                      ensure_ascii=False))
                await asyncio.sleep(PUSH_INTERVAL_S)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            reader.cancel()

    return app


def serve(decision_source, on_text, stop: threading.Event, targets: dict,
          host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    """
    Start the page in a daemon thread and return it.

    Mirrors transmit_loop's contract: decision_source() returns the same
    (Decision, monotonic time) tuple the BLE transport reads.

    Startup failures are reported and swallowed. A missing web dependency must
    not stop the glove.
    """
    bridge = Bridge(decision_source, on_text, targets)

    def run():
        try:
            import uvicorn
        except ImportError:
            print("[frontend] fastapi/uvicorn not installed; status page off",
                  flush=True)
            return
        try:
            config = uvicorn.Config(build_app(bridge), host=host, port=port,
                                    log_level="warning")
            server = uvicorn.Server(config)
            watcher = threading.Thread(
                target=lambda: (stop.wait(), setattr(server, "should_exit", True)),
                daemon=True)
            watcher.start()
            print(f"[frontend] http://localhost:{port}", flush=True)
            server.run()
        except Exception as error:
            print(f"[frontend] disabled: {error}", flush=True)

    thread = threading.Thread(target=run, name="frontend", daemon=True)
    thread.start()
    return thread
