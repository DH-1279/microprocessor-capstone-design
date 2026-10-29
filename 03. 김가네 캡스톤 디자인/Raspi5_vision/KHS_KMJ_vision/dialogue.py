"""Short haptic instructions and bounded retries for important announcements."""

from collections import deque


HAPTIC_HELP = ("엄지는 오른쪽, 새끼는 왼쪽, 검지는 앞, 약지는 뒤, 중지는 근접입니다. "
               "획득하면 전체가 두 번 울립니다.")


def target_announcement(name, first):
    return f"{name}을 찾아 안내합니다. " + (HAPTIC_HELP if first else
                                            "진동을 따라 움직이세요. 획득하면 전체가 두 번 울립니다.")


class Announcements:
    def __init__(self, audio):
        self.audio = audio
        self.pending = deque(maxlen=8)

    def say(self, text):
        if self.pending or not self.audio.say(text):
            if text not in self.pending:
                self.pending.append(text)
        return True

    def flush(self):
        while self.pending and self.audio.say(self.pending[0]):
            self.pending.popleft()
