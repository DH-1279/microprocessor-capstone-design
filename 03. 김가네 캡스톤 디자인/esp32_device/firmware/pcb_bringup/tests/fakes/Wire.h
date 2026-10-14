#pragma once
#include <cstddef>
#include <cstdint>
struct FakeWire {
  bool begin(int, int, uint32_t) { return true; }
  void setTimeOut(int) {}
  void beginTransmission(uint8_t) {}
  void write(uint8_t) {}
  int endTransmission(bool = true) { return 2; }
  size_t requestFrom(uint8_t, size_t, bool) { return 0; }
  int read() { return -1; }
  void end() {}
};
inline FakeWire Wire;
