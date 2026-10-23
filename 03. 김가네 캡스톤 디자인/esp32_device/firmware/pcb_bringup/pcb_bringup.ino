#include <Arduino.h>
#include <Wire.h>
#include <esp_system.h>
#include "bench_control.h"

// Arduino-ESP32 3.x, ESP32C3 Dev Module, USB CDC On Boot: Enabled.
// USB powers ESP; a separate current-limited supply powers the motor rail.
namespace bench = kimgane_bench;
constexpr uint8_t MOTOR_PINS[bench::kChannels] = {0, 1, 3, 4, 5};
constexpr uint8_t LED_PIN = 8;  // Active LOW on the tested C3 SuperMini revision.
constexpr uint8_t SDA_PIN = 6;
constexpr uint8_t SCL_PIN = 7;
constexpr uint32_t PWM_HZ = 200;

bench::State state;
SemaphoreHandle_t stateMutex = nullptr;
bool pwmReady = false;
bool taskReady = false;
bool timedOut = false;
uint8_t attachedCount = 0;

// Called only at startup or while holding stateMutex. No serial I/O here.
void disablePwm() {
  for (size_t i = 0; i < bench::kChannels; ++i) {
    if (i < attachedCount) ledcDetach(MOTOR_PINS[i]);
    digitalWrite(MOTOR_PINS[i], LOW);
    pinMode(MOTOR_PINS[i], OUTPUT);
  }
  attachedCount = 0;
  pwmReady = false;
  state.stop();
  state.dry = true;
  digitalWrite(LED_PIN, HIGH);
}

void applyOutputs() {
  if (pwmReady) {
    for (size_t i = 0; i < bench::kChannels; ++i) {
      const uint8_t duty = state.dry ? 0 : state.duties[i];
      if (!ledcWrite(MOTOR_PINS[i], duty)) {
        disablePwm();
        return;
      }
    }
  }
  digitalWrite(LED_PIN, state.active() ? LOW : HIGH);
}

void forceStop() {
  if (!stateMutex) return;
  xSemaphoreTake(stateMutex, portMAX_DELAY);
  state.stop();
  applyOutputs();
  xSemaphoreGive(stateMutex);
}

// Pulse timing never waits for Serial or I2C. All state/PWM writes use one mutex.
void outputTask(void*) {
  TickType_t wakeTime = xTaskGetTickCount();
  const TickType_t period = max(static_cast<TickType_t>(1), pdMS_TO_TICKS(5));
  for (;;) {
    xSemaphoreTake(stateMutex, portMAX_DELAY);
    if (state.tick(millis())) timedOut = true;
    applyOutputs();
    xSemaphoreGive(stateMutex);
    vTaskDelayUntil(&wakeTime, period);
  }
}

void printStatus() {
  bench::State snapshot;
  bool hardwareReady;
  uint32_t now;
  xSemaphoreTake(stateMutex, portMAX_DELAY);
  now = millis();
  state.tick(now);
  applyOutputs();
  snapshot = state;
  hardwareReady = pwmReady && taskReady;
  xSemaphoreGive(stateMutex);
  Serial.printf("STATUS | mode=%s | pwm=%s | armed=%u | active=%u | left_ms=%lu\n",
                snapshot.dry ? "DRY_RUN" : "MOTORS", hardwareReady ? "OK" : "FAULT",
                snapshot.armed, snapshot.active(),
                static_cast<unsigned long>(snapshot.remaining(now)));
  Serial.printf("COMMAND DUTY | TH=%u IN=%u MI=%u RI=%u PI=%u | 0..255\n",
                snapshot.duties[0], snapshot.duties[1], snapshot.duties[2],
                snapshot.duties[3], snapshot.duties[4]);
  Serial.printf("MEM | free_heap=%lu | min_free_heap=%lu | uptime_ms=%lu\n",
                static_cast<unsigned long>(ESP.getFreeHeap()),
                static_cast<unsigned long>(ESP.getMinFreeHeap()),
                static_cast<unsigned long>(now));
}

void printHelp() {
  Serial.println("help | status | stop | ! | mode dry | mode motors | arm");
  Serial.println("pulse TH 96 300 | pulse TH,IN 96 300 | pulse ALL 96 300");
  Serial.println("Channels: TH IN MI RI PI (uppercase). Duty 1..255; time 1..1000 ms.");
  Serial.println("Each pulse needs arm within 10 s. No continuous/automatic sequence.");
  Serial.println("scan | imu (outputs OFF first; connect IMU only after voltage checks)");
}

bool beginI2c() {
  // Only explicit scan/imu commands touch GPIO6/7; no unpowered sensor backfeed.
  if (!Wire.begin(SDA_PIN, SCL_PIN, 100000)) {
    Serial.println("I2C ERROR | begin failed | outputs OFF");
    return false;
  }
  Wire.setTimeOut(20);
  return true;
}

bool readRegisters(uint8_t address, uint8_t reg, uint8_t* data, size_t count) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(address, count, true) != count) return false;
  for (size_t i = 0; i < count; ++i) data[i] = static_cast<uint8_t>(Wire.read());
  return true;
}

void checkImu(bool scanOnly) {
  // execute(Scan/Imu) already synchronously applied OFF before this call.
  if (!beginI2c()) return;
  if (scanOnly) {
    unsigned int found = 0;
    for (uint8_t address = 8; address < 120; ++address) {
      Wire.beginTransmission(address);
      if (Wire.endTransmission() == 0) {
        Serial.printf("I2C | address=0x%02X\n", address);
        ++found;
      }
    }
    Serial.printf("SCAN | found=%u | expected MPU6050 AD0=GND: 0x68\n", found);
  } else {
    constexpr uint8_t address = 0x68;
    uint8_t identity = 0;
    if (!readRegisters(address, 0x75, &identity, 1) || identity != 0x68) {
      Serial.printf("IMU ERROR | address=0x68 | WHO_AM_I=0x%02X | no valid MPU6050\n",
                    identity);
    } else {
      Wire.beginTransmission(address);
      Wire.write(0x6b);  // PWR_MGMT_1: wake, internal clock.
      Wire.write(0x00);
      if (Wire.endTransmission() != 0) {
        Serial.println("IMU ERROR | wake failed");
      } else {
        delay(100);  // Motor outputs are OFF throughout this optional command.
        uint8_t bytes[14] = {};
        if (!readRegisters(address, 0x3b, bytes, sizeof(bytes))) {
          Serial.println("IMU ERROR | sample read failed");
        } else {
          int16_t raw[7] = {};
          for (size_t i = 0; i < 7; ++i) {
            const uint16_t bits = static_cast<uint16_t>(
                (static_cast<uint16_t>(bytes[2 * i]) << 8) | bytes[2 * i + 1]);
            raw[i] = static_cast<int16_t>(bits);
          }
          Serial.printf("IMU OK | WHO_AM_I=0x68 | RAW ax=%d ay=%d az=%d temp=%d gx=%d gy=%d gz=%d\n",
                        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6]);
        }
      }
    }
  }
  Wire.end();  // Release pins after the explicit test.
}

void dispatch(char* line) {
  const bench::Command command = bench::parse(line);
  xSemaphoreTake(stateMutex, portMAX_DELAY);
  bench::Result result;
  if (!taskReady && (command.action == bench::Action::Arm ||
                     command.action == bench::Action::Pulse ||
                     command.action == bench::Action::Motors)) {
    state.stop();
    result = bench::Result::HardwareUnavailable;
  } else {
    result = state.execute(command, millis(), pwmReady && taskReady);
  }
  applyOutputs();
  const bool hardwareReady = pwmReady && taskReady;
  xSemaphoreGive(stateMutex);
  if (result != bench::Result::Ok) {
    const char* reason = "invalid command";
    if (result == bench::Result::NotArmed) reason = "arm required / expired";
    else if (result == bench::Result::Busy) reason = "busy; stopped; arm again";
    else if (result == bench::Result::HardwareUnavailable) reason = "PWM/task fault";
    Serial.printf("ERROR | %s | outputs OFF, disarmed\n", reason);
    return;
  }
  if (!hardwareReady) Serial.println("FAULT | motor hardware disabled; dry-run only");
  if (command.action == bench::Action::Help) printHelp();
  else if (command.action == bench::Action::Scan || command.action == bench::Action::Imu) {
    checkImu(command.action == bench::Action::Scan);
  } else {
    Serial.println("OK | command accepted (not a physical motor measurement)");
    printStatus();
  }
}

void setup() {
  // External 10k gate pull-downs are needed even before firmware executes.
  digitalWrite(LED_PIN, HIGH);
  pinMode(LED_PIN, OUTPUT);
  for (uint8_t pin : MOTOR_PINS) {
    digitalWrite(pin, LOW);
    pinMode(pin, OUTPUT);
  }
  Serial.begin(115200);
  stateMutex = xSemaphoreCreateMutex();
  if (!stateMutex) {
    Serial.println("FATAL | mutex allocation failed | outputs OFF");
    return;
  }
  pwmReady = true;
  for (uint8_t pin : MOTOR_PINS) {
    if (!ledcAttach(pin, PWM_HZ, 8)) {
      disablePwm();
      break;
    }
    ++attachedCount;
    if (!ledcWrite(pin, 0)) {
      disablePwm();
      break;
    }
  }
  taskReady = xTaskCreate(outputTask, "bench_output", 3072, nullptr, 5, nullptr) == pdPASS;
  if (!taskReady) {
    disablePwm();
    Serial.println("FATAL | output task allocation failed | motors disabled");
  }
  Serial.printf("KIMGANE PCB BENCH | boot=DRY_RUN | reset_reason=%d | TH=0 IN=1 MI=3 RI=4 PI=5\n",
                static_cast<int>(esp_reset_reason()));
  // Do not wait for Serial: a USB-less boot must still leave outputs OFF.
}

void loop() {
  if (!stateMutex) {
    delay(100);
    return;
  }
  static bench::LineReader reader;
  static bench::SerialLink link;
  static uint32_t rxBytes = 0;
  static uint32_t commandLines = 0;
  static uint32_t lastHeartbeat = 0;
  const bool readyHint = static_cast<bool>(Serial);
  if (link.observe(readyHint, Serial.available() > 0, millis())) {
    forceStop();
    // Drop a partial line, but do not swallow the first complete fresh command.
    if (reader.pending()) reader.disconnect();
  }
  if (reader.tick(millis()) == bench::FrameEvent::Timeout) {
    forceStop();
    Serial.println("ERROR | incomplete line timed out | outputs OFF");
  }
  // A bounded read loop cannot starve the output task under a serial flood.
  // Never gate RX on Serial.operator bool(): HWCDC TX readiness may flap.
  for (unsigned int processed = 0; processed < 64 && Serial.available() > 0; ++processed) {
    const int byte = Serial.read();
    if (byte < 0) break;
    ++rxBytes;
    const bench::FrameEvent event = reader.feed(static_cast<char>(byte), millis());
    if (event == bench::FrameEvent::Stop) {
      forceStop();
      Serial.println("STOP | outputs OFF, disarmed");
    } else if (event == bench::FrameEvent::Line) {
      ++commandLines;
      dispatch(reader.line());
    } else if (event == bench::FrameEvent::Invalid) {
      forceStop();
      Serial.println("ERROR | invalid/too long line | outputs OFF");
    }
  }
  xSemaphoreTake(stateMutex, portMAX_DELAY);
  const bool expired = timedOut;
  timedOut = false;
  xSemaphoreGive(stateMutex);
  if (expired) Serial.println("TIMEOUT EVENT | previous pulse/arm expired; use status for current state");
  const uint32_t now = millis();
  if (now - lastHeartbeat >= 2000) {
    lastHeartbeat = now;
    Serial.printf("BENCH v2 | rx=%lu | lines=%lu | uptime_ms=%lu\n",
                  static_cast<unsigned long>(rxBytes),
                  static_cast<unsigned long>(commandLines),
                  static_cast<unsigned long>(now));
  }
  delay(1);
}
