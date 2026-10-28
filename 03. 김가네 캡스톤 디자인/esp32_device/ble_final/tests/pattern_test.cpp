#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../firmware/ble_final_receiver/pattern_protocol.h"

static unsigned checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { \
  fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

void command(uint8_t* p, uint16_t seq, uint8_t mode = 2, uint8_t zone = 0,
             uint8_t channel = 1, uint16_t ttl = 1000) {
  memset(p, 0, haptic2::kSize);
  p[0] = 2; haptic2::write16(p + 2, seq); haptic2::write16(p + 4, ttl);
  p[6] = mode; p[7] = zone; p[8] = channel;
  for (size_t i = 0; i < 5; ++i) p[10+i] = static_cast<uint8_t>(150 + i*10);
}
void status(haptic2::State& s, uint32_t now, uint8_t channel, uint8_t duty,
            bool active = true) {
  uint8_t out[haptic2::kSize]; s.status(now, out);
  CHECK(((out[1] & haptic2::kActive) != 0) == active);
  for (size_t i = 0; i < 5; ++i) CHECK(out[10+i] == (i == channel ? duty : 0));
}

int main() {
  uint8_t p[haptic2::kSize], out[haptic2::kSize];
  haptic2::State s(false); s.connect();
  command(p, 65535); CHECK(s.receive(p, sizeof(p), 0));
  status(s, 0, 1, 160); status(s, 199, 1, 160); status(s, 200, 1, 0);
  // Far period exceeds one TTL: regular refreshes must preserve OFF phase.
  for (uint16_t seq = 0; seq < 13; ++seq) {
    command(p, seq); CHECK(s.receive(p, sizeof(p), 100U*(seq+1U)));
  }
  status(s, 1300, 1, 160); status(s, 1400, 1, 0);
  command(p, 13, 2, 1); CHECK(s.receive(p, sizeof(p), 1400));
  status(s, 1600, 1, 0); status(s, 2099, 1, 0); status(s, 2100, 1, 160);
  command(p, 14, 2, 2); CHECK(s.receive(p, sizeof(p), 2200));
  status(s, 2399, 1, 160); status(s, 2400, 1, 0); status(s, 2600, 1, 160);
  // TTL expiry during ON and OFF both stop the schedule; exact boundary.
  status(s, 3200, 1, 0, false);
  s.status(3200, out); CHECK(out[1] & haptic2::kExpired);
  command(p, 15, 2, 2); CHECK(s.receive(p, sizeof(p), 3201)); status(s, 3201, 1, 160);
  // Direction switches start a new cue and zero every previous channel.
  command(p, 16, 3, 0, 0); CHECK(s.receive(p, sizeof(p), 3300)); status(s, 3300, 0, 150);
  command(p, 17, 3, 0, 4); CHECK(s.receive(p, sizeof(p), 3350)); status(s, 3350, 4, 190);
  // Near repeats until acquisition/STOP; heartbeats preserve the phase.
  command(p, 18, 3, 2, 2); CHECK(s.receive(p, sizeof(p), 4000));
  status(s, 4119, 2, 170); status(s, 4120, 2, 170); status(s, 4240, 2, 0);
  status(s, 4359, 2, 0); status(s, 4360, 2, 0);
  for (uint16_t seq = 19; seq < 50; ++seq) {
    const uint32_t now = 4400U + 100U*(seq-19U);
    command(p, seq, 3, 2, 2); CHECK(s.receive(p, sizeof(p), now));
    status(s, now, 2, ((now - 4000U) % 400U < 200U) ? 170 : 0);
  }
  // Sequence replay stops, but cannot roll back the accepted sequence.
  CHECK(!s.receive(p, sizeof(p), 7500)); status(s, 7500, 2, 0, false);
  s.status(7500, out); CHECK(haptic2::read16(out+2) == 49);
  memset(p, 0, sizeof(p)); p[0] = 2; p[1] = 1; haptic2::write16(p+4, 500);
  CHECK(s.receive(p, sizeof(p), 7501)); s.status(7501, out);
  CHECK(haptic2::read16(out+2) == 49); CHECK(!(out[1] & haptic2::kInvalid));
  const uint16_t ttlCases[] = {99, 100, 1000, 1001};
  for (uint16_t ttl : ttlCases) {
    s.connect(); command(p, 0, 2, 0, 1, ttl);
    CHECK(s.receive(p, sizeof(p), 0) == (ttl >= 100 && ttl <= 1000));
  }
  s.connect(); command(p, 0);
  CHECK(!s.receive(nullptr, sizeof(p), 0));
  CHECK(!s.receive(p, sizeof(p)+1, 0));
  // Invalid fields, flags and all packet truncations fail closed.
  for (size_t n = 0; n < haptic2::kSize; ++n) {
    s.connect(); command(p, 1); CHECK(s.receive(p, sizeof(p), 0));
    CHECK(!s.receive(p, n, 10)); status(s, 10, 0, 0, false);
  }
  for (size_t field = 0; field < haptic2::kSize; ++field) {
    if (field == 2 || field == 3 || field == 4 || field == 5) continue;
    for (unsigned value = 0; value <= 255; ++value) {
      s.connect(); command(p, 1, 3, 0, 0); p[field] = static_cast<uint8_t>(value);
      const bool accepted = s.receive(p, sizeof(p), 0);
      if (!accepted) status(s, 0, 0, 0, false);
      else {
        s.status(0, out);
        unsigned activeChannels = 0;
        for (size_t i = 10; i < 15; ++i) activeChannels += out[i] != 0;
        CHECK(activeChannels <= 1);
      }
    }
  }
  // Disconnect/reconnect, millisecond wrap, and latching hardware fault.
  s.connect(); command(p, 8, 2, 2, 3, 500);
  CHECK(s.receive(p, sizeof(p), 0xfffffff0U)); status(s, 0x20U, 3, 180);
  status(s, 0x1e4U, 3, 0, false);
  s.disconnect(); CHECK(!s.receive(p, sizeof(p), 0));
  s.connect(); CHECK(s.receive(p, sizeof(p), 0));
  s.hardwareFault(); status(s, 1, 3, 0, false);
  s.disconnect(); s.connect(); CHECK(!s.receive(p, sizeof(p), 2));
  s.status(2, out); CHECK(out[1] & haptic2::kFault);
  printf("PASS | %u pattern/protocol checks\n", checks);
}
