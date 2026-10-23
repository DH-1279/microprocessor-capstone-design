#ifndef KIMGANE_HAPTIC_PROTOCOL_H
#define KIMGANE_HAPTIC_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

// Portable state machine: no Arduino, BLE, GPIO, allocation, or I/O calls.
namespace kimgane {

constexpr size_t kPacketSize = 11;
constexpr size_t kChannels = 5;
constexpr uint8_t kVersion = 1;
constexpr uint8_t kStop = 1;
constexpr uint8_t kConnected = 1;
constexpr uint8_t kActive = 2;
constexpr uint8_t kExpired = 4;
constexpr uint8_t kInvalid = 8;
constexpr uint8_t kDryRun = 16;
constexpr uint16_t kMinTtlMs = 100;
constexpr uint16_t kMaxTtlMs = 1000;

inline uint16_t read16(const uint8_t* data) {
  return static_cast<uint16_t>(data[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(data[1]) << 8);
}

inline void write16(uint8_t* data, uint16_t value) {
  data[0] = static_cast<uint8_t>(value & 0xff);
  data[1] = static_cast<uint8_t>(value >> 8);
}

class HapticState {
 public:
  explicit HapticState(bool dryRun) : dryRun_(dryRun) {}

  void connect() {
    resetSession();
    connected_ = true;
  }

  void disconnect() {
    resetSession();
    connected_ = false;
  }

  // Caller serializes access. Invalid/replayed commands always stop output.
  bool receive(const uint8_t* data, size_t length, uint32_t nowMs) {
    if (!connected_ || data == nullptr || length != kPacketSize ||
        data[0] != kVersion || (data[1] & ~kStop) != 0) {
      return reject();
    }
    const uint16_t sequence = read16(data + 2);
    const uint16_t ttl = read16(data + 4);
    if (ttl < kMinTtlMs || ttl > kMaxTtlMs) {
      return reject();
    }
    const bool stop = (data[1] & kStop) != 0;
    bool anyLevel = false;
    for (size_t index = 0; index < kChannels; ++index) {
      anyLevel = anyLevel || data[6 + index] != 0;
    }
    if (stop && anyLevel) {
      return reject();
    }
    const uint16_t delta = static_cast<uint16_t>(sequence - sequence_);
    const bool forward = !hasSequence_ || (delta > 0 && delta < 32768);
    if (!stop && !forward) {
      return reject();
    }
    // STOP is accepted even with an old sequence, but never rolls it back.
    if (forward) {
      sequence_ = sequence;
      hasSequence_ = true;
    }
    invalid_ = false;
    expired_ = false;
    receivedAt_ = nowMs;
    ttlMs_ = ttl;
    active_ = anyLevel && !stop;
    for (size_t index = 0; index < kChannels; ++index) {
      levels_[index] = stop ? 0 : data[6 + index];
    }
    return true;
  }

  void tick(uint32_t nowMs) {
    // Unsigned subtraction remains valid across the millis() wraparound.
    if (active_ && static_cast<uint32_t>(nowMs - receivedAt_) >= ttlMs_) {
      clearOutput();
      expired_ = true;
    }
  }

  void status(uint32_t nowMs, uint8_t out[kPacketSize]) {
    tick(nowMs);
    out[0] = kVersion;
    out[1] = (connected_ ? kConnected : 0) | (active_ ? kActive : 0) |
             (expired_ ? kExpired : 0) | (invalid_ ? kInvalid : 0) |
             (dryRun_ ? kDryRun : 0);
    write16(out + 2, hasSequence_ ? sequence_ : 0);
    const uint16_t remaining = active_
        ? static_cast<uint16_t>(ttlMs_ - static_cast<uint32_t>(nowMs - receivedAt_))
        : 0;
    write16(out + 4, remaining);
    for (size_t index = 0; index < kChannels; ++index) {
      out[6 + index] = levels_[index];
    }
  }

 private:
  void clearOutput() {
    active_ = false;
    for (size_t index = 0; index < kChannels; ++index) {
      levels_[index] = 0;
    }
  }

  bool reject() {
    clearOutput();
    invalid_ = true;
    return false;
  }

  void resetSession() {
    clearOutput();
    expired_ = false;
    invalid_ = false;
    hasSequence_ = false;
    sequence_ = 0;
    receivedAt_ = 0;
    ttlMs_ = kMinTtlMs;
  }

  bool dryRun_;
  bool connected_ = false;
  bool active_ = false;
  bool expired_ = false;
  bool invalid_ = false;
  bool hasSequence_ = false;
  uint16_t sequence_ = 0;
  uint16_t ttlMs_ = kMinTtlMs;
  uint32_t receivedAt_ = 0;
  uint8_t levels_[kChannels] = {};
};

}  // namespace kimgane
#endif
