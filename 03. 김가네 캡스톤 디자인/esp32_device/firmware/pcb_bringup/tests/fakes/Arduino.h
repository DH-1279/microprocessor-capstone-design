#pragma once
// Host-only API substitutes: verify sketch control flow, not USB electronics.
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <deque>
#include <string>
using std::max;
constexpr int LOW = 0;
constexpr int HIGH = 1;
constexpr int OUTPUT = 1;
constexpr int pdPASS = 1;
constexpr unsigned int portMAX_DELAY = 0xffffffffu;
using TickType_t = uint32_t;
using SemaphoreHandle_t = void*;
namespace fake {
inline uint32_t now = 0;
inline unsigned int pwm[256] = {};
inline int gpio[256] = {};
inline bool attached[256] = {};
inline bool writeFails = false;
struct TaskYield {};
}
inline uint32_t millis() { return fake::now; }
inline void delay(uint32_t ms) { fake::now += ms; }
inline void digitalWrite(uint8_t pin, int value) { fake::gpio[pin] = value; }
inline void pinMode(uint8_t, int) {}
inline bool ledcAttach(uint8_t pin, uint32_t, uint8_t) {
  fake::attached[pin] = true;
  return true;
}
inline bool ledcDetach(uint8_t pin) {
  fake::attached[pin] = false;
  fake::pwm[pin] = 0;
  return true;
}
inline bool ledcWrite(uint8_t pin, uint32_t duty) {
  if (fake::writeFails) return false;
  fake::pwm[pin] = duty;
  return true;
}
inline SemaphoreHandle_t xSemaphoreCreateMutex() { return &fake::now; }
inline int xSemaphoreTake(SemaphoreHandle_t, unsigned int) { return pdPASS; }
inline int xSemaphoreGive(SemaphoreHandle_t) { return pdPASS; }
inline int xTaskCreate(void (*)(void*), const char*, int, void*, int, void*) {
  return pdPASS;
}
inline TickType_t xTaskGetTickCount() { return fake::now; }
inline TickType_t pdMS_TO_TICKS(uint32_t ms) { return ms; }
inline void vTaskDelayUntil(TickType_t*, TickType_t) { throw fake::TaskYield{}; }
struct FakeSerial {
  bool ready = false;
  std::deque<uint8_t> input;
  std::string output;
  explicit operator bool() const { return ready; }
  void begin(int) {}
  int available() const { return static_cast<int>(input.size()); }
  int read() {
    if (input.empty()) return -1;
    const uint8_t value = input.front();
    input.pop_front();
    return value;
  }
  void send(const std::string& text) {
    for (unsigned char c : text) input.push_back(c);
  }
  void println(const char* text) { output += std::string(text) + '\n'; }
  void printf(const char* format, ...) {
    char buffer[1024] = {};
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    output += buffer;
  }
};
inline FakeSerial Serial;
struct FakeEsp {
  uint32_t getFreeHeap() const { return 100000; }
  uint32_t getMinFreeHeap() const { return 90000; }
};
inline FakeEsp ESP;
