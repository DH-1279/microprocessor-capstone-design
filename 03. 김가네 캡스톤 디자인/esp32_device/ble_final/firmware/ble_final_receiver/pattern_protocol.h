#ifndef KIMGANE_PATTERN_PROTOCOL_H
#define KIMGANE_PATTERN_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

// BLE v2: independent of Arduino. Carrier PWM is fixed; this schedules ON/OFF.
namespace haptic2 {
constexpr size_t kSize = 16;
constexpr size_t kChannels = 5;
constexpr uint8_t kVersion = 2;
constexpr uint8_t kStop = 1;
constexpr uint8_t kConnected = 1, kActive = 2, kExpired = 4;
constexpr uint8_t kInvalid = 8, kDry = 16, kFault = 32;
enum Mode : uint8_t { Rate = 2, Direction = 3 };
enum Zone : uint8_t { Far = 0, Middle = 1, Near = 2 };
// Prefix symbols: Arduino defines PI as a macro, and SDKs define IN/STATUS.
enum Channel : uint8_t { kThumb = 0, kIndex = 1, kMiddle = 2, kRing = 3, kLittle = 4 };

inline uint16_t read16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}
inline void write16(uint8_t* p, uint16_t n) {
  p[0] = static_cast<uint8_t>(n);
  p[1] = static_cast<uint8_t>(n >> 8);
}

class State {
 public:
  explicit State(bool dry = true) : dry_(dry) {}
  void connect() { reset(); connected_ = true; }
  void disconnect() { reset(); connected_ = false; }
  void hardwareFault() { clear(); fault_ = true; }

  bool receive(const uint8_t* p, size_t size, uint32_t now) {
    tick(now);  // A late refresh must restart, not revive an expired phase.
    if (!connected_ || fault_ || p == nullptr || size != kSize ||
        p[0] != kVersion || (p[1] & ~kStop) || p[9] || p[15]) return reject();
    const uint16_t seq = read16(p + 2), ttl = read16(p + 4);
    if (ttl < 100 || ttl > 1000) return reject();
    const bool stop = (p[1] & kStop) != 0;
    if (stop) {
      // Canonical STOP has no mode, proximity, channel or calibration values.
      for (size_t i = 6; i < kSize; ++i) if (p[i]) return reject();
    } else {
      if ((p[6] != Rate && p[6] != Direction) || p[7] > Near ||
          p[8] >= kChannels || (p[6] == Direction &&
          ((p[7] == Near && p[8] != kMiddle) || (p[7] != Near && p[8] == kMiddle)))) {
        return reject();
      }
      for (size_t i = 10; i < 15; ++i) if (p[i] == 0) return reject();
    }
    const uint16_t delta = static_cast<uint16_t>(seq - sequence_);
    const bool newer = !hasSequence_ || (delta > 0 && delta < 32768);
    if (!stop && !newer) return reject();
    if (newer) { sequence_ = seq; hasSequence_ = true; }
    invalid_ = expired_ = false;
    if (stop) { clear(); return true; }
    // Heartbeats do NOT restart the pulse or repeat a completed near event.
    if (!active_ || mode_ != p[6] || zone_ != p[7] || channel_ != p[8]) {
      phaseStart_ = now;
    }
    mode_ = p[6]; zone_ = p[7]; channel_ = p[8];
    for (size_t i = 0; i < kChannels; ++i) duties_[i] = p[10 + i];
    active_ = true; receivedAt_ = now; ttl_ = ttl;
    render(now);
    return true;
  }

  void tick(uint32_t now) {
    if (active_ && static_cast<uint32_t>(now - receivedAt_) >= ttl_) {
      clear(); expired_ = true;
    }
    render(now);
  }

  void status(uint32_t now, uint8_t out[kSize]) {
    tick(now);
    out[0] = kVersion;
    out[1] = (connected_ ? kConnected : 0) | (active_ ? kActive : 0) |
             (expired_ ? kExpired : 0) | (invalid_ ? kInvalid : 0) |
             (dry_ ? kDry : 0) | (fault_ ? kFault : 0);
    write16(out + 2, hasSequence_ ? sequence_ : 0);
    write16(out + 4, active_ ? static_cast<uint16_t>(ttl_ -
            static_cast<uint32_t>(now - receivedAt_)) : 0);
    out[6] = mode_; out[7] = zone_; out[8] = channel_;
    out[9] = outputs_[channel_] != 0 ? 1 : 0;
    for (size_t i = 0; i < kChannels; ++i) out[10 + i] = outputs_[i];
    out[15] = 0;
  }

 private:
  void render(uint32_t now) {
    for (size_t i = 0; i < kChannels; ++i) outputs_[i] = 0;
    if (!active_ || fault_) return;
    const uint32_t elapsed = static_cast<uint32_t>(now - phaseStart_);
    bool on;
    if (mode_ == Direction && zone_ == Near) {
      // One double tap on entry. Near is NOT proof of contact or grasp.
      on = elapsed < 120 || (elapsed >= 240 && elapsed < 360);
    } else {
      const uint32_t periods[] = {1200, 700, 400};  // ON=200ms, OFF=1000/500/200.
      on = elapsed % periods[zone_] < 200;
    }
    if (on) outputs_[channel_] = duties_[channel_];
  }
  void clear() {
    active_ = false;
    for (size_t i = 0; i < kChannels; ++i) outputs_[i] = 0;
  }
  bool reject() { clear(); invalid_ = true; return false; }
  void reset() {
    clear(); hasSequence_ = invalid_ = expired_ = false;
    sequence_ = 0; receivedAt_ = phaseStart_ = 0; ttl_ = 100;
    mode_ = Rate; zone_ = Far; channel_ = kIndex;
    // Hardware faults latch until RESET, including across reconnects.
  }
  bool dry_, connected_ = false, active_ = false, expired_ = false;
  bool invalid_ = false, fault_ = false, hasSequence_ = false;
  uint16_t sequence_ = 0, ttl_ = 100;
  uint32_t receivedAt_ = 0, phaseStart_ = 0;
  uint8_t mode_ = Rate, zone_ = Far, channel_ = kIndex;
  uint8_t duties_[kChannels] = {}, outputs_[kChannels] = {};
};
}  // namespace haptic2
#endif
