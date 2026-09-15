#include <Arduino.h>

// Board: ESP32C3 Dev Module
// Tools > USB CDC On Boot: Enabled
// First test: USB power only, with no external circuits connected.
// Common black C3 SuperMini: GPIO8 is an active-low onboard LED.
// If your board revision differs, check its schematic first.
constexpr uint8_t STATUS_LED_PIN = 8;
constexpr uint32_t BLINK_INTERVAL_MS = 500;

bool ledOn = false;
uint32_t lastToggleMs = 0;
uint32_t tick = 0;

void setup() {
  digitalWrite(STATUS_LED_PIN, HIGH);
  pinMode(STATUS_LED_PIN, OUTPUT);
  Serial.begin(115200);
}

void loop() {
  const uint32_t now = millis();
  if (now - lastToggleMs >= BLINK_INTERVAL_MS) {
    lastToggleMs = now;
    ledOn = !ledOn;
    digitalWrite(STATUS_LED_PIN, ledOn ? LOW : HIGH);
    Serial.print("C3 SuperMini OK | tick=");
    Serial.print(++tick);
    Serial.print(" | uptime_ms=");
    Serial.println(now);
  }
}
