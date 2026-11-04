// Optional host test bridge; compile the actual ESP pattern engine, no Arduino.
#include "../../../esp32_device/ble_final/firmware/ble_final_receiver/pattern_protocol.h"

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

static haptic2::State state(true);
EXPORT void v2_reset() { state = haptic2::State(true); state.connect(); }
EXPORT int v2_receive(const uint8_t* p, size_t n, uint32_t now) {
  return state.receive(p, n, now);
}
EXPORT void v2_status(uint32_t now, uint8_t* p) { state.status(now, p); }
