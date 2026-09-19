#include "../haptic_protocol.h"

#if defined(_WIN32)
#define KIMGANE_EXPORT __declspec(dllexport)
#else
#define KIMGANE_EXPORT __attribute__((visibility("default")))
#endif

// Single-threaded host test bridge; production firmware serializes state access.
static kimgane::HapticState receiver(true);

extern "C" {
KIMGANE_EXPORT void reset() { receiver = kimgane::HapticState(true); }
KIMGANE_EXPORT void connect_receiver() { receiver.connect(); }
KIMGANE_EXPORT void disconnect_receiver() { receiver.disconnect(); }
KIMGANE_EXPORT int receive_packet(const uint8_t* data, size_t length, uint32_t now) {
  return receiver.receive(data, length, now);
}
KIMGANE_EXPORT void snapshot_receiver(uint32_t now, uint8_t* output) {
  receiver.status(now, output);
}
}
