"""Short haptic instructions and bounded retries for important announcements."""

from collections import deque


HAPTIC_HELP = ("엄지는 오른쪽, 새끼는 왼쪽, 검지는 앞, 약지는 뒤, 중지는 근접입니다. "
               "획득하면 전체가 두 번 울립니다.")
SHORT_HAPTIC_HELP = "진동을 따라 움직이세요. 획득하면 전체가 두 번 울립니다."
STARTUP_HELP = "시스템이 준비되었습니다. 짧은 소리가 끝나면 말씀해주세요. 캡스톤이라고 말씀해주세요."


def target_announcement_parts(name, first):
    return (f"{name}을 찾아 안내합니다.", HAPTIC_HELP if first else SHORT_HAPTIC_HELP)


def target_announcement(name, first):
    return " ".join(target_announcement_parts(name, first))


class Announcements:
    def __init__(self, audio):
        self.audio = audio
        self.pending = deque(maxlen=8)

    def say(self, text):
        if self.pending or not self.audio.say(text):
            if text not in self.pending:
                self.pending.append(text)
        return True

    def say_parts(self, parts):
        parts = tuple(parts)
        if self.pending or not self.audio.say_parts(parts):
            if parts not in self.pending:
                self.pending.append(parts)
        return True

    def flush(self):
        while self.pending:
            job = self.pending[0]
            accepted = self.audio.say(job) if isinstance(job, str) else self.audio.say_parts(job)
            if not accepted:
                break
            self.pending.popleft()
