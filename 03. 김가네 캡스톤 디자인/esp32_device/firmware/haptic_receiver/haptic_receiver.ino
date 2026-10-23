#include <Arduino.h>
#include <NimBLEDevice.h>
#include "haptic_protocol.h"

// ESP32C3 Dev Module, USB CDC On Boot: Enabled, NimBLE-Arduino 2.5.1.
// KEEP true until the motor supply, MOSFETs, diodes and pin mapping are checked.
constexpr bool DRY_RUN = true;
constexpr uint8_t LED_PIN = 8;  // Common C3 SuperMini: active LOW.
constexpr uint8_t MOTOR_PINS[kimgane::kChannels] = {0, 1, 3, 4, 5};
constexpr uint32_t PWM_FREQUENCY_HZ = 200;
constexpr uint8_t PWM_RESOLUTION_BITS = 8;
constexpr uint32_t OUTPUT_PERIOD_MS = 5;
constexpr char DEVICE_NAME[] = "KIMGANE-HAPTIC";
constexpr char SERVICE_UUID[] = "328e0010-8c2a-4e58-9a48-512fc6ab1279";
constexpr char COMMAND_UUID[] = "328e0011-8c2a-4e58-9a48-512fc6ab1279";
constexpr char STATUS_UUID[] = "328e0012-8c2a-4e58-9a48-512fc6ab1279";

kimgane::HapticState state(DRY_RUN);
portMUX_TYPE stateMux = portMUX_INITIALIZER_UNLOCKED;
bool outputReady = false;  // Set once during setup before advertising.

void snapshot(uint8_t out[kimgane::kPacketSize]) {
  portENTER_CRITICAL(&stateMux);
  state.status(millis(), out);
  portEXIT_CRITICAL(&stateMux);
}

// Only this task touches PWM after setup. It never performs Serial/BLE I/O.
void outputTask(void*) {
  TickType_t wakeTime = xTaskGetTickCount();
  const TickType_t period = max(static_cast<TickType_t>(1),
                               pdMS_TO_TICKS(OUTPUT_PERIOD_MS));
  for (;;) {
    uint8_t packet[kimgane::kPacketSize];
    snapshot(packet);
    if (!DRY_RUN) {
      for (size_t index = 0; index < kimgane::kChannels; ++index) {
        ledcWrite(MOTOR_PINS[index], packet[6 + index]);
      }
    }
    digitalWrite(LED_PIN, (packet[1] & kimgane::kActive) ? LOW : HIGH);
    vTaskDelayUntil(&wakeTime, period);
  }
}

class CommandCallbacks : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* characteristic, NimBLEConnInfo&) override {
    const auto value = characteristic->getValue();
    portENTER_CRITICAL(&stateMux);
    state.receive(value.data(), value.size(), millis());
    portEXIT_CRITICAL(&stateMux);
  }
} commandCallbacks;

class StatusCallbacks : public NimBLECharacteristicCallbacks {
  void onRead(NimBLECharacteristic* characteristic, NimBLEConnInfo&) override {
    uint8_t packet[kimgane::kPacketSize];
    snapshot(packet);
    characteristic->setValue(packet, sizeof(packet));
  }
} statusCallbacks;

class ServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer*, NimBLEConnInfo&) override {
    portENTER_CRITICAL(&stateMux);
    state.connect();
    portEXIT_CRITICAL(&stateMux);
    // Advertising stops after connection. This prototype accepts one central.
  }

  void onDisconnect(NimBLEServer*, NimBLEConnInfo&, int) override {
    portENTER_CRITICAL(&stateMux);
    state.disconnect();
    portEXIT_CRITICAL(&stateMux);
    NimBLEDevice::startAdvertising();
  }
} serverCallbacks;

void setup() {
  digitalWrite(LED_PIN, HIGH);
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);
  if (!DRY_RUN) {
    // Pull-down resistors must hold gates low before firmware starts.
    for (size_t index = 0; index < kimgane::kChannels; ++index) {
      digitalWrite(MOTOR_PINS[index], LOW);
      pinMode(MOTOR_PINS[index], OUTPUT);
    }
    for (size_t index = 0; index < kimgane::kChannels; ++index) {
      if (!ledcAttach(MOTOR_PINS[index], PWM_FREQUENCY_HZ, PWM_RESOLUTION_BITS)) {
        for (size_t stopIndex = 0; stopIndex < index; ++stopIndex) {
          ledcWrite(MOTOR_PINS[stopIndex], 0);
        }
        Serial.println("FATAL | PWM setup failed | outputs OFF | BLE disabled");
        return;
      }
      ledcWrite(MOTOR_PINS[index], 0);
    }
  }
  if (xTaskCreate(outputTask, "haptic_output", 3072, nullptr, 5, nullptr) != pdPASS) {
    Serial.println("FATAL | output task failed | outputs OFF | BLE disabled");
    return;
  }
  outputReady = true;
  NimBLEDevice::init(DEVICE_NAME);
  auto* server = NimBLEDevice::createServer();
  server->setCallbacks(&serverCallbacks);
  auto* service = server->createService(SERVICE_UUID);
  // Keep a larger GATT capacity so malformed long values reach our validator.
  auto* command = service->createCharacteristic(COMMAND_UUID, NIMBLE_PROPERTY::WRITE, 512);
  command->setCallbacks(&commandCallbacks);
  auto* status = service->createCharacteristic(STATUS_UUID, NIMBLE_PROPERTY::READ,
                                               kimgane::kPacketSize);
  status->setCallbacks(&statusCallbacks);
  uint8_t initialStatus[kimgane::kPacketSize];
  snapshot(initialStatus);
  status->setValue(initialStatus, sizeof(initialStatus));

  auto* advertising = NimBLEDevice::getAdvertising();
  advertising->enableScanResponse(true);
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setName(DEVICE_NAME);
  if (advertising->start()) {
    Serial.printf("BLE ready | name=%s | mode=%s | channels=TH,IN,MI,RI,PI\n",
                  DEVICE_NAME, DRY_RUN ? "DRY_RUN" : "MOTORS");
  } else {
    Serial.println("BLE advertising failed | outputs OFF | press RESET");
  }
}

void loop() {
  // Reporting is deliberately separate from the output/watchdog task.
  static uint8_t previous[kimgane::kPacketSize] = {};
  static uint32_t lastMemoryReport = 0;
  if (!outputReady) {
    delay(1000);
    return;
  }
  uint8_t packet[kimgane::kPacketSize];
  snapshot(packet);
  bool changed = false;
  for (size_t index = 0; index < kimgane::kPacketSize; ++index) {
    // Remaining TTL changes every tick; do not log that alone.
    if (index != 4 && index != 5 && packet[index] != previous[index]) {
      changed = true;
    }
    previous[index] = packet[index];
  }
  if (changed) {
    Serial.printf("STATE | seq=%u | flags=0x%02X | TH=%u IN=%u MI=%u RI=%u PI=%u\n",
                  kimgane::read16(packet + 2), packet[1], packet[6], packet[7],
                  packet[8], packet[9], packet[10]);
  }
  const uint32_t now = millis();
  if (now - lastMemoryReport >= 5000) {
    lastMemoryReport = now;
    Serial.printf("MEM | free_heap=%lu | min_free_heap=%lu | uptime_ms=%lu\n",
                  static_cast<unsigned long>(ESP.getFreeHeap()),
                  static_cast<unsigned long>(ESP.getMinFreeHeap()),
                  static_cast<unsigned long>(now));
  }
  delay(50);
}
