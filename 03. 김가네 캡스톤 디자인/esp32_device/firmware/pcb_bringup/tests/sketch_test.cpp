// Compile with -Itests/fakes. Includes the actual sketch, not a copied loop.
#include "../pcb_bringup.ino"
#include <cstdlib>
#include <iostream>
unsigned int checks = 0;
void check(bool ok, const char* message) {
  ++checks;
  if (!ok) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}
void send(const char* text) {
  Serial.send(text);
  loop();
}
void outputTick() {
  try {
    outputTask(nullptr);
  } catch (const fake::TaskYield&) {
    // Execute one actual task iteration without starting a host thread.
  }
}
bool physicalOff() {
  for (uint8_t pin : MOTOR_PINS) if (fake::pwm[pin] != 0) return false;
  return true;
}
int main() {
  setup();
  check(state.dry && physicalOff(), "boot OFF");
  Serial.output.clear();
  send("status\n");  // Regression: RX present while Serial bool is false.
  check(Serial.output.find("STATUS | mode=DRY_RUN") != std::string::npos,
        "false ready hint must not block actual status input");
  check(Serial.input.empty(), "input drained with false readiness");
  send("arm\npulse TH 96 300\n");
  check(state.active() && physicalOff(), "dry pulse leaves all hardware outputs zero");
  forceStop();
  send("mode motors\narm\npulse IN 160 300\n");
  check(fake::pwm[1] == 160, "IN GPIO1 driven");
  constexpr uint8_t otherPins[] = {0, 3, 4, 5};
  for (uint8_t pin : otherPins) check(fake::pwm[pin] == 0, "other channels OFF");
  send("!");
  check(physicalOff() && !state.armed, "bang without newline stops");
  send("\narm\npulse TH 96 300\n");
  fake::now += 300;
  outputTick();
  check(physicalOff() && !state.active(), "output task expires without serial loop");
  send("arm\npulse TH 96 1000\n");
  check(!physicalOff(), "active pulse before link loss");
  fake::now += 500;
  loop();
  check(physicalOff() && !state.armed, "sustained link loss stops pulse");
  Serial.output.clear();
  send("status\n");
  check(Serial.output.find("STATUS") != std::string::npos,
        "first fresh full command after disconnect not swallowed");
  fake::now += 2000;
  loop();
  check(Serial.output.find("BENCH v2 | rx=") != std::string::npos, "heartbeat diagnostics");
  send("arm\npulse TH 96 1000\n");
  send("pulse TH 999 300\n");
  check(physicalOff() && !state.armed, "invalid command stops actual PWM path");
  send("arm\npulse TH 96 300\n");
  fake::writeFails = true;
  outputTick();
  check(!pwmReady && state.dry && !state.active() && physicalOff(), "write failure OFF/dry");
  for (uint8_t pin : MOTOR_PINS) {
    check(!fake::attached[pin] && fake::gpio[pin] == LOW, "failed PWM detached, GPIO LOW");
  }
  send("mode motors\n");
  check(state.dry && physicalOff(), "fault blocks motor mode");
  std::cout << "PASS | " << checks << " sketch integration checks | mocked hardware\n";
}
