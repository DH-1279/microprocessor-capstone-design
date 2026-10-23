"""Grammar-based voice/button command parsing. No hardware imports.

Replaces substring target matching, which accepts any utterance that merely
contains an alias. "책상 정리했어" must not select the book target.

Every word of the utterance has to fit the command grammar. Unknown words,
negations, several targets or a target mixed with a control word are refused,
so a misheard sentence cannot start guidance. Jamo-level edit distance still
recovers STT noise inside one word ("컴" -> "컵"), but never across word
boundaries, because that is what made substring matching unsafe.

Target labels and aliases come from controller.targets, so COCO label names
such as "cell phone" stay in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

CHO = tuple("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = tuple("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = ("",) + tuple("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
HANGUL_START, HANGUL_END = 0xAC00, 0xD7A3

CONTROLS = {
    "stop":   ("정지", "멈춰", "멈춤", "그만", "중지", "종료", "stop", "off"),
    "list":   ("목록", "리스트", "주변안내", "list"),
    "start":  ("시작", "네", "응", "시작해"),
    "resume": ("계속", "계속해", "이어서"),
    "done":   ("찾았어", "찾았어요", "찾았다", "완료"),
    "repeat": ("다시", "다시말해줘", "다시안내", "뭐라고"),
}
DIRECTIONS = {"left": ("왼쪽", "왼편", "좌측"), "right": ("오른쪽", "오른편", "우측")}

REQUESTS = ("찾아줘", "찾아", "찾아주세요", "알려줘", "어디야", "어딨어",
            "어디", "줘", "주세요", "해줘", "해주세요", "보여줘")
FILLERS = ("좀", "그", "저", "음", "어", "나", "내", "제", "이제", "저기")
NEGATIONS = ("말고", "아니", "아니라", "아니고", "빼고", "제외")
PARTICLES = ("으로", "에서", "에게", "한테", "이랑", "랑", "하고", "와", "과",
             "을", "를", "이", "가", "은", "는", "도", "만", "의", "에")

# A word further than this from every alias is treated as a different word.
# 0.4 lets "마우스패드" reach "마우스"; 0.35 keeps them apart.
WORD_THRESHOLD = 0.35

# Utterances at most this long are retried as one joined word, because STT
# splits short commands ("오른쪽" -> "모른 쪽") and omits spaces ("컵찾아줘").
JOIN_MAX_CHARS = 6

# States map to the commands they accept, per the accessibility UX spec.
STATE_COMMANDS = {
    "scanning":  ("target", "list", "stop"),
    "selecting": ("left", "right", "stop", "repeat", "list"),
    "confirming": ("start", "target", "stop", "repeat"),
    "guiding":   ("stop", "list", "repeat"),
    "paused":    ("stop", "repeat"),
    "waiting":   ("resume", "stop", "repeat"),
    "near":      ("done", "repeat", "stop"),
    "stopped":   ("target", "list"),
    "any":       ("target", "left", "right", "stop", "list",
                  "start", "resume", "done", "repeat"),
}

# Stop outranks every other command, so no state may omit it.
STATE_COMMANDS = {state: tuple(dict.fromkeys(commands + ("stop",)))
                  for state, commands in STATE_COMMANDS.items()}

# Button identifiers reach the same parser as speech, so UI and voice share
# one transition path instead of two.
BUTTON_ALIASES = {
    "stop": "정지", "start": "시작", "resume": "계속", "done": "찾았어",
    "list": "목록", "repeat": "다시",
    "choose_left": "왼쪽", "choose_right": "오른쪽",
}


@dataclass(frozen=True)
class Intent:
    """Normalized command, kept separate from the raw STT sentence."""

    kind: str           # target | control | direction | none
    name: str | None    # COCO label, control name, or direction
    reason: str         # ok, or why the utterance was refused
    raw: str
    detail: str = ""

    def __bool__(self) -> bool:
        return self.name is not None


def to_jamo(text: str) -> list[str]:
    out: list[str] = []
    for char in text:
        code = ord(char)
        if HANGUL_START <= code <= HANGUL_END:
            offset = code - HANGUL_START
            out.append(CHO[offset // 588])
            out.append(JUNG[(offset % 588) // 28])
            if JONG[offset % 28]:
                out.append(JONG[offset % 28])
        else:
            out.append(char)
    return out


def edit_distance(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return len(a) or len(b)
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            cost = 0 if left == right else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + cost))
        previous = current
    return previous[-1]


def word_distance(word: str, other: str) -> float:
    """Normalized jamo distance. 0 is identical, 1 is unrelated."""
    left, right = to_jamo(word), to_jamo(other)
    if not left or not right:
        return 1.0
    return edit_distance(left, right) / max(len(left), len(right))


def strip_particle(word: str) -> str:
    for particle in sorted(PARTICLES, key=len, reverse=True):
        if word.endswith(particle) and len(word) > len(particle):
            return word[: -len(particle)]
    return word


def _alias_table(targets: dict) -> dict[str, tuple[str, ...]]:
    """Build label -> aliases from controller.targets, label itself included."""
    table: dict[str, tuple[str, ...]] = {}
    for label, item in targets.items():
        aliases = tuple(str(a).strip() for a in item.get("aliases", ())
                        if str(a).strip())
        name = str(item.get("name", "")).strip()
        extra = (name,) if name else ()
        table[label] = tuple(dict.fromkeys(aliases + extra + (label,)))
    return table


def _match(word: str, table: dict):
    """Closest entry within the threshold, or None. Whole words only."""
    best_name, best_alias, best_score = None, None, 1.0
    for name, aliases in table.items():
        for alias in aliases:
            score = word_distance(word, alias)
            if score < best_score:
                best_name, best_alias, best_score = name, alias, score
    if best_score <= WORD_THRESHOLD:
        return best_name, best_alias
    return None


def _match_plain(word: str, options) -> bool:
    return any(word_distance(word, option) <= WORD_THRESHOLD
               for option in options)


def _classify(word: str, aliases: dict, segment: bool = True):
    """Return (kind, name, matched_alias) for one whitespace-separated word."""
    bare = strip_particle(word)

    # Exact matches win. "찾아줘" and "찾았어" differ by two jamo, so fuzzy
    # matching alone turns the request word into the done command.
    for candidate in (word, bare):
        for table, kind in ((aliases, "target"), (CONTROLS, "control"),
                            (DIRECTIONS, "direction")):
            for name, options in table.items():
                if candidate in options:
                    return kind, name, candidate
        for options, kind in ((NEGATIONS, "negation"), (REQUESTS, "request"),
                              (FILLERS, "filler")):
            if candidate in options:
                return kind, None, candidate

    for table, kind in ((aliases, "target"), (CONTROLS, "control"),
                        (DIRECTIONS, "direction")):
        hit = _match(bare, table)
        if hit:
            return kind, hit[0], hit[1]

    # Split a joined word before the fuzzy request check, otherwise "컵찾아줘"
    # is close enough to "찾아줘" to be swallowed as a request word.
    if segment and len(word) >= 3:
        for i in range(1, len(word)):
            left = _classify(word[:i], aliases, segment=False)
            if left[0] not in ("target", "control", "direction"):
                continue
            if _classify(word[i:], aliases, segment=False)[0] in ("request",
                                                                 "filler"):
                return left

    for options, kind in ((NEGATIONS, "negation"), (REQUESTS, "request"),
                          (FILLERS, "filler")):
        if _match_plain(bare, options):
            return kind, None, bare

    return "unknown", None, bare


def parse(text: str, targets: dict, state: str = "any") -> Intent:
    """Parse one utterance. Refusals carry a reason the UI can explain."""
    raw = text or ""
    cleaned = unicodedata.normalize("NFC", raw).strip().lower()
    for mark in ".,!?;:":
        cleaned = cleaned.replace(mark, " ")
    words = cleaned.split()
    if not words:
        return Intent("none", None, "empty", raw)

    aliases = _alias_table(targets)
    found = [(*_classify(word, aliases), word) for word in words]

    unknown = [item[3] for item in found if item[0] == "unknown"]
    if unknown:
        joined = "".join(words)
        if len(joined) <= JOIN_MAX_CHARS:
            retry = _classify(joined, aliases)
            if retry[0] in ("target", "control", "direction"):
                found, unknown = [(*retry, joined)], []
        if unknown:
            return Intent("none", None, "unknown_word", raw,
                          " ".join(unknown))

    if any(item[0] == "negation" for item in found):
        return Intent("none", None, "negation", raw)

    picked = {kind: [i for i in found if i[0] == kind]
              for kind in ("target", "control", "direction")}

    if len(picked["target"]) > 1:
        return Intent("none", None, "multiple_targets", raw,
                      ", ".join(i[1] for i in picked["target"]))
    if len(picked["control"]) > 1:
        return Intent("none", None, "multiple_commands", raw)
    if len(picked["direction"]) > 1:
        return Intent("none", None, "multiple_directions", raw)
    if picked["control"] and (picked["target"] or picked["direction"]):
        return Intent("none", None, "mixed_command", raw)

    for kind in ("direction", "control", "target"):
        if picked[kind]:
            _, name, alias, _ = picked[kind][0]
            intent = Intent(kind, name, "ok", raw, alias)
            break
    else:
        return Intent("none", None, "no_command", raw)

    allowed = STATE_COMMANDS.get(state, STATE_COMMANDS["any"])
    key = "target" if intent.kind == "target" else intent.name
    if key not in allowed:
        return Intent("none", None, "not_allowed_in_state", raw,
                      f"{intent.name} / {state}")
    return intent


def parse_input(text: str, targets: dict, state: str = "any") -> Intent:
    """Entry point for both speech and UI buttons.

    Object buttons are sent as select_<label>; labels live in the config, so
    they are resolved here instead of in a fixed alias table.
    """
    key = (text or "").strip()

    if key.startswith("select_"):
        label = key[len("select_"):].replace("_", " ")
        if label in targets:
            allowed = STATE_COMMANDS.get(state, STATE_COMMANDS["any"])
            if "target" not in allowed:
                return Intent("none", None, "not_allowed_in_state", key,
                              f"{label} / {state}")
            return Intent("target", label, "ok", key, label)
        return Intent("none", None, "unknown_button", key, label)

    return parse(BUTTON_ALIASES.get(key, key), targets, state)


def button_id(label: str) -> str:
    """UI identifier for an object button. COCO labels may contain spaces."""
    return "select_" + label.replace(" ", "_")


REFUSAL_SPEECH = {
    "unknown_word": "알아듣지 못했어요. 다시 말씀해 주세요.",
    "negation": "하나만 말씀해 주세요.",
    "multiple_targets": "찾을 물건 하나만 말씀해 주세요.",
    "multiple_commands": "한 번에 하나만 말씀해 주세요.",
    "multiple_directions": "왼쪽과 오른쪽 중 하나만 말씀해 주세요.",
    "mixed_command": "한 번에 하나만 말씀해 주세요.",
    "no_command": "찾을 물건 이름을 말씀해 주세요.",
    "not_allowed_in_state": "지금은 할 수 없어요.",
    "empty": "",
}


def refusal_speech(intent: Intent) -> str:
    """What to say when an utterance is refused. Empty means stay silent."""
    return REFUSAL_SPEECH.get(intent.reason, "")
