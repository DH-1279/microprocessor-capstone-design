"""Synthetic closed-loop scenarios; does not claim camera/STT performance."""

from datetime import datetime, timezone
import json
from pathlib import Path

from .controller import Controller
from .models import Detection, Hand, Observation
from .protocol import Command, SimulatedReceiver

CUP = Detection("cup", (0.65, 0.3, 0.85, 0.55), 0.95)
LEFT_CUP = Detection("cup", (0.1, 0.3, 0.3, 0.55), 0.94)
HAND = Hand((0.4, 0.7), 0.99, "Right")


class Harness:
    def __init__(self, config):
        self.controller = Controller(config["controller"])
        self.receiver = SimulatedReceiver()
        self.receiver.connect()
        self.ttl = config["ble"]["ttl_ms"]
        self.now = 0.0
        self.frame = 0
        self.seq = 0
        self.events = []

    def step(self, objects=(CUP,), hand=HAND, command=None, advance=0.1, publish=True, deliver=True):
        self.now += advance
        observation = None
        if publish:
            observation = Observation(self.frame, self.now, tuple(objects), hand)
            self.frame += 1
        decision = self.controller.step(self.now, observation, command)
        packet = Command(self.seq, self.ttl, decision.levels, decision.stopped).encode()
        self.seq = (self.seq + 1) & 65535
        if deliver:
            self.receiver.receive(packet, self.now)
        self.receiver.tick(self.now)
        self.events.append({"t": round(self.now, 3), "command": command,
                            "state": decision.state, "reason": decision.reason,
                            "levels": list(decision.levels), "receiver_levels": list(self.receiver.levels),
                            "speech": decision.speech})
        return decision

    def warm(self, objects=(CUP,)):
        for _ in range(self.controller.config["stable_frames"]):
            self.step(objects)


def scenario_guidance(config):
    h = Harness(config)
    h.warm()
    assert h.step(command="컵 찾아줘").state == "guiding"
    assert any(h.receiver.levels)
    assert h.step(hand=Hand((0.72, 0.4), 0.99, "Right")).proximity == "near"
    assert h.step(hand=None).reason == "hand_lost"
    assert not any(h.receiver.levels)
    assert h.step().state == "guiding"
    assert h.step(command="정지").reason == "user_stop"
    assert not any(h.receiver.levels)
    return h.events


def scenario_target_loss(config):
    h = Harness(config)
    h.warm()
    h.step(command="컵")
    assert h.step(objects=()).reason == "target_lost"
    h.warm()
    assert h.step().reason == "target_lost"  # Reappearing cup must not restart output.
    assert not any(h.receiver.levels)
    assert h.step(command="컵").state == "guiding"
    return h.events


def scenario_stale_frame(config):
    h = Harness(config)
    h.warm()
    h.step(command="컵")
    assert h.step(publish=False, advance=config["controller"]["max_frame_age_s"] + 0.01).reason == "stale_frame"
    assert not any(h.receiver.levels)
    h.warm()
    assert h.step().reason == "target_lost"
    return h.events


def scenario_ambiguity(config):
    h = Harness(config)
    h.warm((CUP, LEFT_CUP))
    decision = h.step((CUP, LEFT_CUP), command="컵")
    assert decision.stopped and "여러" in decision.speech
    assert h.step((CUP, LEFT_CUP), command="오른쪽").state == "guiding"
    selected = h.controller.target_id
    h.step((LEFT_CUP, CUP))
    assert h.controller.target_id == selected
    assert h.step((LEFT_CUP,), command=None).reason == "target_lost"
    return h.events


def scenario_link_timeout(config):
    h = Harness(config)
    h.warm()
    h.step(command="컵")
    assert any(h.receiver.levels)
    h.step(advance=config["ble"]["ttl_ms"] / 1000 + 0.01, deliver=False)
    assert not any(h.receiver.levels) and h.receiver.expired
    h.receiver.disconnect()
    assert not any(h.receiver.levels)
    h.receiver.connect()
    assert not any(h.receiver.levels) and h.receiver.sequence is None
    return h.events


def scenario_invalid_input(config):
    h = Harness(config)
    h.warm()
    h.step(command="컵")
    assert not h.receiver.receive(b"invalid", h.now)
    assert not any(h.receiver.levels) and h.receiver.invalid
    assert h.step(command="컵과 물병 찾아줘").stopped
    assert h.step(command="열쇠 찾아줘").stopped
    malformed = Detection("cup", (float("nan"), 0.2, 0.4, 0.5), 0.9)
    assert h.step(objects=(malformed,)).stopped
    return h.events


SCENARIOS = {
    "guidance_hand_loss_stop": scenario_guidance,
    "target_loss_requires_reselection": scenario_target_loss,
    "stale_frames_stop": scenario_stale_frame,
    "multiple_cups_keep_identity": scenario_ambiguity,
    "packet_loss_watchdog_reconnect": scenario_link_timeout,
    "invalid_packet_and_request": scenario_invalid_input,
}


def simulate(config: dict, report: str | None = None) -> int:
    results = []
    for name, scenario in SCENARIOS.items():
        try:
            events = scenario(config)
            results.append({"name": name, "passed": True, "events": events})
            print(f"PASS {name} ({len(events)} steps)")
        except Exception as error:
            results.append({"name": name, "passed": False, "error": f"{type(error).__name__}: {error}"})
            print(f"FAIL {name}: {error}")
    result = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Synthetic detections + command text + controller + encoded BLE + simulated receiver; no model/hardware benchmark",
              "passed": all(item["passed"] for item in results), "scenarios": results}
    if report:
        path = Path(report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {path}")
    return 0 if result["passed"] else 1
