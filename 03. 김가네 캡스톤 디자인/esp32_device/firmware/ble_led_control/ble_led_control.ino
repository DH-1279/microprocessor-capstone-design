#include <Arduino.h>
#include <NimBLEDevice.h>

// ESP32C3 Dev Module, USB CDC On Boot: Enabled, NimBLE-Arduino 2.5.1.
// Common C3 SuperMini onboard LED: GPIO8, active LOW. USB power only.
constexpr uint8_t LED_PIN = 8;
constexpr char DEVICE_NAME[] = "KIMGANE-LED";
constexpr char SERVICE_UUID[] = "328e0001-8c2a-4e58-9a48-512fc6ab1279";
constexpr char LED_UUID[] = "328e0002-8c2a-4e58-9a48-512fc6ab1279";

NimBLECharacteristic* ledCharacteristic = nullptr;
bool ledOn = false;  // After setup, accessed only by the BLE host callbacks.

void setLed(bool on) {
  ledOn = on;
  digitalWrite(LED_PIN, on ? LOW : HIGH);
  if (ledCharacteristic != nullptr) {
    ledCharacteristic->setValue(on ? "1" : "0");
  }
}

class LedCallbacks : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* characteristic, NimBLEConnInfo&) override {
    const auto value = characteristic->getValue();
    if (value.size() == 1 && (value[0] == '0' || value[0] == '1')) {
      setLed(value[0] == '1');
      Serial.printf("RX | LED=%s\n", ledOn ? "ON" : "OFF");
    } else {
      // Restore readable state after an invalid write.
      setLed(ledOn);
      Serial.println("RX rejected | expected ASCII 0 or 1");
    }
  }
} ledCallbacks;

class ServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer*, NimBLEConnInfo&) override {
    Serial.println("BLE connected");
  }

  void onDisconnect(NimBLEServer*, NimBLEConnInfo&, int reason) override {
    // Runs once the BLE stack detects the lost connection, not instantly.
    setLed(false);
    Serial.printf("BLE disconnected | reason=%d | LED=OFF\n", reason);
    NimBLEDevice::startAdvertising();
  }
} serverCallbacks;

void setup() {
  digitalWrite(LED_PIN, HIGH);
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);
  NimBLEDevice::init(DEVICE_NAME);
  auto* server = NimBLEDevice::createServer();
  server->setCallbacks(&serverCallbacks);
  auto* service = server->createService(SERVICE_UUID);
  ledCharacteristic = service->createCharacteristic(
      LED_UUID, NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::WRITE, 1);
  ledCharacteristic->setCallbacks(&ledCallbacks);
  setLed(false);

  auto* advertising = NimBLEDevice::getAdvertising();
  // Put the full name in the scan response; the UUID stays in advertising data.
  advertising->enableScanResponse(true);
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setName(DEVICE_NAME);
  if (advertising->start()) {
    Serial.println("BLE ready | name=KIMGANE-LED | LED=OFF");
  } else {
    Serial.println("BLE advertising failed | press RESET and check serial output");
  }
}

void loop() {
  static uint32_t lastReportMs = 0;
  const uint32_t now = millis();
  if (now - lastReportMs >= 5000) {
    lastReportMs = now;
    Serial.printf("MEM | free_heap=%lu | min_free_heap=%lu | uptime_ms=%lu\n",
                  static_cast<unsigned long>(ESP.getFreeHeap()),
                  static_cast<unsigned long>(ESP.getMinFreeHeap()),
                  static_cast<unsigned long>(now));
  }
  delay(10);
}
