#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace kimgane_bench {
constexpr size_t kChannels = 5;
constexpr uint32_t kArmWindowMs = 10000;
constexpr uint32_t kMaxPulseMs = 1000;
constexpr uint8_t kAllChannels = 0x1f;
constexpr const char* kNames[kChannels] = {"TH", "IN", "MI", "RI", "PI"};

// USB TX readiness is only a hint. Received bytes prove an active input path.
class SerialLink {
 public:
  bool observe(bool readyHint, bool hasInput, uint32_t now) {
    if (readyHint || hasInput) {
      online_ = true;
      lastLive_ = now;
      return false;
    }
    if (online_ && now - lastLive_ >= 500) {
      online_ = false;
      return true;
    }
    return false;
  }

 private:
  bool online_ = false;
  uint32_t lastLive_ = 0;
};

enum class FrameEvent { None, Line, Stop, Invalid, Timeout };
class LineReader {
 public:
  char* line() { return buffer_; }
  bool pending() const { return length_ != 0; }
  void disconnect() {
    length_ = 0;
    discard_ = true;
  }
  FrameEvent tick(uint32_t now) {
    if (length_ && now - lastByte_ >= 1000) {
      disconnect();
      return FrameEvent::Timeout;
    }
    return FrameEvent::None;
  }
  FrameEvent feed(char c, uint32_t now) {
    lastByte_ = now;
    if (c == '!') {
      disconnect();
      return FrameEvent::Stop;
    }
    if (c == '\r' || c == '\n') {
      const bool complete = !discard_ && length_ != 0;
      buffer_[length_] = '\0';
      length_ = 0;
      discard_ = false;
      return complete ? FrameEvent::Line : FrameEvent::None;
    }
    if (discard_) return FrameEvent::None;
    if ((c < 32 && c != '\t') || c > 126 || length_ == sizeof(buffer_) - 1) {
      disconnect();
      return FrameEvent::Invalid;
    }
    buffer_[length_++] = c;
    return FrameEvent::None;
  }

 private:
  char buffer_[80] = {};
  size_t length_ = 0;
  bool discard_ = false;
  uint32_t lastByte_ = 0;
};

enum class Action { Invalid, Help, Status, Stop, Dry, Motors, Arm, Pulse, Scan, Imu };
struct Command {
  Action action = Action::Invalid;
  uint8_t mask = 0;
  uint8_t duty = 0;
  uint32_t durationMs = 0;
};

inline bool number(const char* text, uint32_t limit, uint32_t& out) {
  if (*text == '\0') return false;
  uint32_t value = 0;
  for (; *text; ++text) {
    if (*text < '0' || *text > '9') return false;
    const uint32_t digit = static_cast<uint32_t>(*text - '0');
    if (value > limit / 10 || (value == limit / 10 && digit > limit % 10)) {
      return false;
    }
    value = value * 10 + digit;
  }
  out = value;
  return true;
}

inline bool channelMask(char* text, uint8_t& mask) {
  if (std::strcmp(text, "ALL") == 0) {
    mask = kAllChannels;
    return true;
  }
  mask = 0;
  char* part = text;
  for (;;) {
    char* comma = std::strchr(part, ',');
    if (comma) *comma = '\0';
    uint8_t bit = 0;
    for (size_t i = 0; i < kChannels; ++i) {
      if (std::strcmp(part, kNames[i]) == 0) bit = static_cast<uint8_t>(1u << i);
    }
    if (bit == 0 || (mask & bit) != 0) return false;
    mask = static_cast<uint8_t>(mask | bit);
    if (!comma) return true;
    part = comma + 1;
  }
}

// Bounded serial framing is handled by the sketch; this parser edits its buffer.
inline Command parse(char* line) {
  char* tokens[5] = {};
  size_t count = 0;
  for (char* p = line; *p;) {
    while (*p == ' ' || *p == '\t') ++p;
    if (!*p) break;
    if (count == 5) return {};
    tokens[count++] = p;
    while (*p && *p != ' ' && *p != '\t') ++p;
    if (*p) *p++ = '\0';
  }
  Command command;
  if (count == 1) {
    const char* name = tokens[0];
    if (std::strcmp(name, "help") == 0) command.action = Action::Help;
    else if (std::strcmp(name, "status") == 0) command.action = Action::Status;
    else if (std::strcmp(name, "stop") == 0) command.action = Action::Stop;
    else if (std::strcmp(name, "arm") == 0) command.action = Action::Arm;
    else if (std::strcmp(name, "scan") == 0) command.action = Action::Scan;
    else if (std::strcmp(name, "imu") == 0) command.action = Action::Imu;
  } else if (count == 2 && std::strcmp(tokens[0], "mode") == 0) {
    if (std::strcmp(tokens[1], "dry") == 0) command.action = Action::Dry;
    else if (std::strcmp(tokens[1], "motors") == 0) command.action = Action::Motors;
  } else if (count == 4 && std::strcmp(tokens[0], "pulse") == 0) {
    uint32_t duty = 0;
    uint32_t duration = 0;
    if (channelMask(tokens[1], command.mask) && number(tokens[2], 255, duty) &&
        duty > 0 && number(tokens[3], kMaxPulseMs, duration) && duration > 0) {
      command.action = Action::Pulse;
      command.duty = static_cast<uint8_t>(duty);
      command.durationMs = duration;
    }
  }
  return command;
}

enum class Result { Ok, Invalid, NotArmed, Busy, HardwareUnavailable };
class State {
 public:
  bool dry = true;
  bool armed = false;
  uint8_t duties[kChannels] = {};

  bool active() const {
    for (uint8_t duty : duties) if (duty != 0) return true;
    return false;
  }
  void stop() {
    for (uint8_t& duty : duties) duty = 0;
    armed = false;
    durationMs_ = 0;
  }
  bool tick(uint32_t now) {
    if (active() && now - pulseStart_ >= durationMs_) {
      stop();
      return true;
    }
    if (armed && now - armStart_ >= kArmWindowMs) {
      stop();
      return true;
    }
    return false;
  }
  uint32_t remaining(uint32_t now) const {
    const uint32_t elapsed = now - pulseStart_;
    return active() && elapsed < durationMs_ ? durationMs_ - elapsed : 0;
  }
  Result execute(const Command& command, uint32_t now, bool hardwareReady) {
    tick(now);
    switch (command.action) {
      case Action::Help:
      case Action::Status:
        return Result::Ok;
      case Action::Stop:
      case Action::Scan:
      case Action::Imu:
        stop();
        return Result::Ok;
      case Action::Dry:
        stop();
        dry = true;
        return Result::Ok;
      case Action::Motors:
        stop();
        if (!hardwareReady) return Result::HardwareUnavailable;
        dry = false;
        return Result::Ok;
      case Action::Arm:
        if (active()) return fail(Result::Busy);
        if (!dry && !hardwareReady) return fail(Result::HardwareUnavailable);
        armed = true;
        armStart_ = now;
        return Result::Ok;
      case Action::Pulse:
        // Validate even if a caller bypasses the text parser.
        if (command.mask == 0 || (command.mask & ~kAllChannels) != 0 ||
            command.duty == 0 || command.durationMs == 0 ||
            command.durationMs > kMaxPulseMs) return fail(Result::Invalid);
        if (active()) return fail(Result::Busy);
        if (!armed) return fail(Result::NotArmed);
        if (!dry && !hardwareReady) return fail(Result::HardwareUnavailable);
        armed = false;  // One arm authorizes exactly one pulse.
        for (size_t i = 0; i < kChannels; ++i) {
          duties[i] = (command.mask & (1u << i)) ? command.duty : 0;
        }
        pulseStart_ = now;
        durationMs_ = command.durationMs;
        return Result::Ok;
      default:
        return fail(Result::Invalid);
    }
  }

 private:
  uint32_t pulseStart_ = 0;
  uint32_t durationMs_ = 0;
  uint32_t armStart_ = 0;
  Result fail(Result result) {
    stop();
    return result;
  }
};
}  // namespace kimgane_bench
