#include "../haptic_protocol.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
unsigned int checks = 0;

void check(bool passed, int line) {
  ++checks;
  if (!passed) {
    std::fprintf(stderr, "FAIL at line %d\n", line);
    std::exit(1);
  }
}
#define CHECK(condition) check((condition), __LINE__)

void command(uint8_t* packet, uint16_t seq, uint16_t ttl, uint8_t flags = 0,
             uint8_t level = 123) {
  packet[0] = kimgane::kVersion;
  packet[1] = flags;
  kimgane::write16(packet + 2, seq);
  kimgane::write16(packet + 4, ttl);
  for (size_t index = 0; index < kimgane::kChannels; ++index) {
    packet[6 + index] = level;
  }
}

void checkOff(kimgane::HapticState& state, uint32_t now) {
  uint8_t status[kimgane::kPacketSize];
  state.status(now, status);
  CHECK((status[1] & kimgane::kActive) == 0);
  CHECK(kimgane::read16(status + 4) == 0);
  for (size_t index = 6; index < kimgane::kPacketSize; ++index) {
    CHECK(status[index] == 0);
  }
}

void testWireFormatAndExpiry() {
  kimgane::HapticState state(true);
  uint8_t status[kimgane::kPacketSize];
  state.status(0, status);
  CHECK(status[0] == 1 && status[1] == kimgane::kDryRun);
  checkOff(state, 0);
  state.connect();
  const uint8_t packet[] = {1, 0, 0x34, 0x12, 0x2c, 0x01, 0, 1, 127, 254, 255};
  CHECK(state.receive(packet, sizeof(packet), 1000));
  state.status(1000, status);
  CHECK(status[1] == (kimgane::kDryRun | kimgane::kConnected | kimgane::kActive));
  CHECK(kimgane::read16(status + 2) == 0x1234);
  CHECK(kimgane::read16(status + 4) == 300);
  CHECK(std::memcmp(status + 6, packet + 6, kimgane::kChannels) == 0);
  state.status(1299, status);
  CHECK(kimgane::read16(status + 4) == 1);
  checkOff(state, 1300);
  state.status(1300, status);
  CHECK((status[1] & kimgane::kExpired) != 0);
  CHECK(kimgane::read16(status + 2) == 0x1234);
}

void testInvalidPacketsStopWithoutSequenceAdvance() {
  uint8_t packet[512] = {};
  uint8_t valid[kimgane::kPacketSize];
  uint8_t status[kimgane::kPacketSize];
  command(valid, 77, 500);
  // Every possible delivered GATT length other than exactly eleven is invalid.
  for (size_t length = 0; length <= sizeof(packet); ++length) {
    if (length == kimgane::kPacketSize) {
      continue;
    }
    kimgane::HapticState state(true);
    state.connect();
    CHECK(state.receive(valid, sizeof(valid), 10));
    command(packet, 78, 1000);
    CHECK(!state.receive(packet, length, 20));
    checkOff(state, 20);
    state.status(20, status);
    CHECK((status[1] & kimgane::kInvalid) != 0);
    CHECK(kimgane::read16(status + 2) == 77);
    // Invalid data must not consume the next valid sequence number.
    command(valid, 78, 100);
    CHECK(state.receive(valid, sizeof(valid), 30));
    command(valid, 77, 500);
  }
  kimgane::HapticState state(true);
  state.connect();
  CHECK(!state.receive(nullptr, kimgane::kPacketSize, 0));
  checkOff(state, 0);
  for (unsigned int flag = 2; flag <= 255; ++flag) {
    command(packet, 1, 100, static_cast<uint8_t>(flag), 0);
    CHECK(!state.receive(packet, kimgane::kPacketSize, 0));
  }
  for (unsigned int version = 0; version <= 255; ++version) {
    if (version == kimgane::kVersion) {
      continue;
    }
    command(packet, 1, 100);
    packet[0] = static_cast<uint8_t>(version);
    CHECK(!state.receive(packet, kimgane::kPacketSize, 0));
  }
  command(packet, 1, 100, kimgane::kStop, 1);
  CHECK(!state.receive(packet, kimgane::kPacketSize, 0));
  checkOff(state, 0);
}

void testTtlLimitsAndCounterWrap() {
  uint8_t packet[kimgane::kPacketSize];
  uint8_t status[kimgane::kPacketSize];
  const uint16_t invalidTtls[] = {0, 1, 99, 1001, 65535};
  for (uint16_t ttl : invalidTtls) {
    kimgane::HapticState state(true);
    state.connect();
    command(packet, 1, ttl);
    CHECK(!state.receive(packet, sizeof(packet), 0));
    checkOff(state, 0);
  }
  const uint16_t validTtls[] = {100, 1000};
  for (uint16_t ttl : validTtls) {
    kimgane::HapticState state(false);
    state.connect();
    command(packet, 1, ttl);
    CHECK(state.receive(packet, sizeof(packet), 0));
    state.status(ttl - 1U, status);
    CHECK((status[1] & kimgane::kActive) != 0);
    CHECK((status[1] & kimgane::kDryRun) == 0);
    checkOff(state, ttl);
  }
  kimgane::HapticState state(true);
  state.connect();
  command(packet, 65535, 100);
  CHECK(state.receive(packet, sizeof(packet), UINT32_MAX - 40U));
  state.status(58, status);
  CHECK((status[1] & kimgane::kActive) != 0);
  CHECK(kimgane::read16(status + 4) == 1);
  checkOff(state, 59);
  command(packet, 0, 100);
  CHECK(state.receive(packet, sizeof(packet), 60));
  state.status(60, status);
  CHECK(kimgane::read16(status + 2) == 0);
  CHECK((status[1] & (kimgane::kInvalid | kimgane::kExpired)) == 0);
}

void testReplayStopAndReconnect() {
  kimgane::HapticState state(true);
  uint8_t packet[kimgane::kPacketSize];
  uint8_t status[kimgane::kPacketSize];
  command(packet, 500, 300);
  CHECK(!state.receive(packet, sizeof(packet), 0));
  state.connect();
  CHECK(state.receive(packet, sizeof(packet), 0));
  CHECK(!state.receive(packet, sizeof(packet), 10));
  checkOff(state, 10);
  command(packet, 501, 300);
  CHECK(state.receive(packet, sizeof(packet), 20));
  command(packet, 500, 300);
  CHECK(!state.receive(packet, sizeof(packet), 30));
  command(packet, static_cast<uint16_t>(501 + 32768), 300);
  CHECK(!state.receive(packet, sizeof(packet), 30));
  command(packet, 502, 300);
  CHECK(state.receive(packet, sizeof(packet), 40));
  // Repeated, stale STOP must still stop immediately and keep sequence 502.
  command(packet, 1, 300, kimgane::kStop, 0);
  CHECK(state.receive(packet, sizeof(packet), 50));
  CHECK(state.receive(packet, sizeof(packet), 60));
  checkOff(state, 60);
  state.status(60, status);
  CHECK(kimgane::read16(status + 2) == 502);
  CHECK((status[1] & (kimgane::kInvalid | kimgane::kExpired)) == 0);
  command(packet, 503, 1000);
  CHECK(state.receive(packet, sizeof(packet), 70));
  state.disconnect();
  checkOff(state, 71);
  state.status(71, status);
  CHECK(status[1] == kimgane::kDryRun);
  CHECK(kimgane::read16(status + 2) == 0);
  state.connect();
  command(packet, 3, 100);
  CHECK(state.receive(packet, sizeof(packet), 80));
  // All-zero non-STOP is a valid normal command and does not expire active output.
  command(packet, 4, 100, 0, 0);
  CHECK(state.receive(packet, sizeof(packet), 90));
  checkOff(state, 200);
  state.status(200, status);
  CHECK((status[1] & kimgane::kExpired) == 0);
}
}  // namespace

int main() {
  testWireFormatAndExpiry();
  testInvalidPacketsStopWithoutSequenceAdvance();
  testTtlLimitsAndCounterWrap();
  testReplayStopAndReconnect();
  std::printf("PASS: %u protocol checks (wire format, malformed input, TTL, replay, STOP, reconnect)\n", checks);
}
