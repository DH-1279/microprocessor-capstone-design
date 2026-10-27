#include <Arduino.h>
#include <NimBLEDevice.h>
#include "pattern_protocol.h"

// Arduino-ESP32 3.3.11, NimBLE-Arduino 2.5.1, C3 USB CDC enabled.
constexpr bool DRY_RUN = true;  // Set false for the already checked motor PCB.
constexpr uint8_t MOTOR_PINS[] = {0, 1, 3, 4, 5};  // TH IN MI RI PI
constexpr uint8_t LED_PIN = 8;
constexpr char NAME[] = "KIMGANE-HAPTIC-V2";
constexpr char SERVICE_UUID[] = "328e0020-8c2a-4e58-9a48-512fc6ab1279";
constexpr char COMMAND_UUID[] = "328e0021-8c2a-4e58-9a48-512fc6ab1279";
constexpr char STATUS_UUID[] = "328e0022-8c2a-4e58-9a48-512fc6ab1279";

haptic2::State state(DRY_RUN);
SemaphoreHandle_t stateMutex = nullptr;
bool ready = false;
uint8_t attached = 0;

// Startup or output task only. BLE callbacks never touch PWM.
void disableOutputs() {
  for (size_t i = 0; i < haptic2::kChannels; ++i) {
    if (i < attached) ledcDetach(MOTOR_PINS[i]);
    digitalWrite(MOTOR_PINS[i], LOW);
    pinMode(MOTOR_PINS[i], OUTPUT);
  }
  attached = 0;
  digitalWrite(LED_PIN, HIGH);
  state.hardwareFault();
}

void snapshot(uint8_t out[haptic2::kSize]) {
  xSemaphoreTake(stateMutex, portMAX_DELAY);
  state.status(millis(), out);
  xSemaphoreGive(stateMutex);
}

void outputTask(void*) {
  TickType_t wake = xTaskGetTickCount();
  const TickType_t period = max(static_cast<TickType_t>(1), pdMS_TO_TICKS(5));
  for (;;) {
    uint8_t packet[haptic2::kSize];
    // Serialize state and GPIO application: a snapshot cannot outlive a STOP.
    xSemaphoreTake(stateMutex, portMAX_DELAY);
    state.status(millis(), packet);
    bool ok = !(packet[1] & haptic2::kFault);
    if (!DRY_RUN && ok) {
      for (size_t i = 0; i < haptic2::kChannels; ++i) {
        if (!ledcWrite(MOTOR_PINS[i], packet[10 + i])) { ok = false; break; }
      }
      if (!ok) disableOutputs();
    }
    digitalWrite(LED_PIN, ok && packet[9] ? LOW : HIGH);
    xSemaphoreGive(stateMutex);
    vTaskDelayUntil(&wake, period);
  }
}

class Writes : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* characteristic, NimBLEConnInfo&) override {
    const auto value = characteristic->getValue();
    xSemaphoreTake(stateMutex, portMAX_DELAY);
    state.receive(value.data(), value.size(), millis());
    xSemaphoreGive(stateMutex);
  }
} writes;

class Reads : public NimBLECharacteristicCallbacks {
  void onRead(NimBLECharacteristic* characteristic, NimBLEConnInfo&) override {
    uint8_t packet[haptic2::kSize];
    snapshot(packet);
    characteristic->setValue(packet, sizeof(packet));
  }
} reads;

class Connections : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer*, NimBLEConnInfo&) override {
    xSemaphoreTake(stateMutex, portMAX_DELAY);
    state.connect();
    xSemaphoreGive(stateMutex);
  }
  void onDisconnect(NimBLEServer*, NimBLEConnInfo&, int) override {
    xSemaphoreTake(stateMutex, portMAX_DELAY);
    state.disconnect();
    xSemaphoreGive(stateMutex);
    NimBLEDevice::startAdvertising();
  }
} connections;

void setup() {
  digitalWrite(LED_PIN, HIGH); pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);
  // Also hold gates low in dry mode; no PWM is attached in that mode.
  for (auto pin : MOTOR_PINS) { digitalWrite(pin, LOW); pinMode(pin, OUTPUT); }
  stateMutex = xSemaphoreCreateMutex();
  if (!stateMutex) { Serial.println("FATAL | mutex | OFF"); return; }
  if (!DRY_RUN) {
    for (auto pin : MOTOR_PINS) {
      if (!ledcAttach(pin, 200, 8)) {
        disableOutputs(); Serial.println("FATAL | PWM attach | OFF"); return;
      }
      ++attached;
      if (!ledcWrite(pin, 0)) {
        disableOutputs(); Serial.println("FATAL | PWM write | OFF"); return;
      }
    }
  }
  if (xTaskCreate(outputTask, "haptic_v2", 3072, nullptr, 5, nullptr) != pdPASS) {
    disableOutputs(); Serial.println("FATAL | task | OFF"); return;
  }
  NimBLEDevice::init(NAME);
  auto* server = NimBLEDevice::createServer();
  server->setCallbacks(&connections);
  auto* service = server->createService(SERVICE_UUID);
  service->createCharacteristic(COMMAND_UUID, NIMBLE_PROPERTY::WRITE, 512)->setCallbacks(&writes);
  auto* status = service->createCharacteristic(STATUS_UUID, NIMBLE_PROPERTY::READ, haptic2::kSize);
  status->setCallbacks(&reads);
  uint8_t packet[haptic2::kSize]; snapshot(packet);
  status->setValue(packet, sizeof(packet));
  auto* advertising = NimBLEDevice::getAdvertising();
  advertising->enableScanResponse(true);
  advertising->addServiceUUID(SERVICE_UUID); advertising->setName(NAME);
  ready = advertising->start();
  Serial.printf("BLE V2 | %s | %s | modes=2,3 | TH=0 IN=1 MI=3 RI=4 PI=5\n",
                ready ? "READY" : "ADVERTISING FAILED", DRY_RUN ? "DRY_RUN" : "MOTORS");
}

void loop() {
  if (!ready) { delay(1000); return; }
  static uint32_t last = 0;
  if (millis() - last >= 1000) {
    last = millis();
    uint8_t p[haptic2::kSize]; snapshot(p);
    Serial.printf("V2 | flags=%02X seq=%u mode=%u zone=%u ch=%u on=%u | %u,%u,%u,%u,%u\n",
                  p[1], haptic2::read16(p + 2), p[6], p[7], p[8], p[9],
                  p[10], p[11], p[12], p[13], p[14]);
  }
  delay(20);
}
