#include "../bench_control.h"
#include <cstdlib>
#include <iostream>
#include <string>

namespace bench = kimgane_bench;
unsigned int checks = 0;
void check(bool ok, const char* message) {
  ++checks;
  if (!ok) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}
bench::Command command(const char* text) {
  char line[256] = {};
  const size_t length = std::strlen(text);
  if (length >= sizeof(line)) {
    check(false, "test input fits parser buffer");
    return {};
  }
  std::memcpy(line, text, length + 1);
  return bench::parse(line);
}

void parserTests() {
  check(command(" help \t").action == bench::Action::Help, "whitespace");
  check(command("mode motors").action == bench::Action::Motors, "motor mode");
  check(command("mode dry").action == bench::Action::Dry, "dry mode");
  check(command("scan").action == bench::Action::Scan, "scan");
  check(command("imu").action == bench::Action::Imu, "imu");
  const auto group = command("pulse TH,IN 255 1000");
  check(group.mask == 3 && group.duty == 255 && group.durationMs == 1000,
        "two channel maximum boundaries");
  check(command("pulse ALL 1 1").mask == 31, "all minimum boundaries");
  const char* bad[] = {"", " ", "pulse TH 0 300", "pulse TH 256 300",
      "pulse TH -1 300", "pulse TH +1 300", "pulse TH 1.0 300",
      "pulse TH 96 0", "pulse TH 96 1001", "pulse TH 96 -1",
      "pulse TH 42949672960 300", "pulse TH 96 9999999999999999999",
      "pulse TH,TH 96 300", "pulse TH, 96 300", "pulse ,TH 96 300",
      "pulse TH,,IN 96 300", "pulse UNKNOWN 96 300", "pulse th 96 300",
      "pulse TH 96 300 junk", "pulse TH 96", "arm junk", "mode", "on",
      "arm one two three four five"};
  for (const char* text : bad) {
    check(command(text).action == bench::Action::Invalid, "reject malformed input");
  }
}

void stateTests() {
  bench::State state;
  check(state.dry && !state.armed && !state.active(), "boot OFF/dry/disarmed");
  check(state.execute(command("pulse TH 96 300"), 0, true) == bench::Result::NotArmed,
        "pulse without arm rejected");
  state.execute(command("arm"), 10, true);
  check(state.execute(command("pulse TH 96 300"), 11, true) == bench::Result::Ok,
        "armed pulse accepted");
  check(!state.armed && state.duties[0] == 96 && state.duties[1] == 0, "single channel");
  check(!state.tick(310) && state.remaining(310) == 1, "pulse before deadline");
  check(state.tick(311) && !state.active() && !state.armed, "pulse exact deadline OFF");
  state.execute(command("arm"), 1000, true);
  check(state.execute(command("pulse TH 96 300"), 11000, true) == bench::Result::NotArmed,
        "arm exact expiry rejected");
  state.execute(command("arm"), 12000, true);
  state.execute(command("pulse ALL 96 1000"), 12001, true);
  check(state.execute(command("pulse TH 96 300"), 12002, true) == bench::Result::Busy &&
        !state.active(), "busy pulse stops, cannot extend");
  state.execute(command("arm"), 13000, true);
  state.execute(command("pulse ALL 96 1000"), 13001, true);
  check(state.execute(command("arm"), 13002, true) == bench::Result::Busy &&
        !state.active(), "cannot arm during pulse");
  check(state.execute(command("mode motors"), 14000, false) ==
        bench::Result::HardwareUnavailable && state.dry, "hardware fault blocks motors");
  state.execute(command("mode motors"), 15000, true);
  state.execute(command("arm"), 15001, true);
  check(state.execute(command("pulse TH 96 300"), 15002, false) ==
        bench::Result::HardwareUnavailable && !state.active(), "fault after arming stops");
  state.execute(command("mode dry"), 15003, false);
  check(state.dry, "dry recovery");
  for (const char* action : {"stop", "scan", "imu", "mode dry", "mode motors", "bogus"}) {
    state.execute(command("arm"), 20000, true);
    state.execute(command("pulse ALL 255 1000"), 20001, true);
    state.execute(command(action), 20002, true);
    check(!state.active() && !state.armed, "stop/mode/I2C/invalid always OFF");
  }
  state.execute(command("arm"), 0xffffff00u, true);
  state.execute(command("pulse PI 96 500"), 0xffffff01u, true);
  check(!state.tick(0x000000f4u), "millis wrap pulse before deadline");
  check(state.tick(0x000000f5u), "millis wrap pulse deadline");
  state.execute(command("arm"), 0xfffff000u, true);
  check(!state.tick(0x0000170fu), "millis wrap arm before deadline");
  check(state.tick(0x00001710u), "millis wrap arm deadline");
  const char* names[] = {"TH", "IN", "MI", "RI", "PI"};
  for (size_t channel = 0; channel < 5; ++channel) {
    for (uint32_t duty = 1; duty <= 255; ++duty) {
      const std::string text = std::string("pulse ") + names[channel] + " " +
                              std::to_string(duty) + " 1000";
      state.execute(command("arm"), 30000, true);
      state.execute(command(text.c_str()), 30001, true);
      for (size_t i = 0; i < 5; ++i) {
        check(state.duties[i] == (i == channel ? duty : 0), "channel isolation at each duty");
      }
      state.tick(31001);
      check(!state.active(), "every duty expires");
    }
  }
  bench::Command bypass = command("pulse TH 96 300");
  bypass.mask = 0x80;
  state.execute(command("arm"), 40000, true);
  check(state.execute(bypass, 40001, true) == bench::Result::Invalid && !state.armed,
        "invalid direct command fails closed");
}

void frameTests() {
  bench::LineReader reader;
  for (const char c : std::string("pulse TH 96 300")) {
    check(reader.feed(c, 0) == bench::FrameEvent::None, "buffer until newline");
  }
  check(reader.feed('\r', 0) == bench::FrameEvent::Line, "CR commits line");
  check(command(reader.line()).action == bench::Action::Pulse, "framed command");
  check(reader.feed('\n', 0) == bench::FrameEvent::None, "CRLF doesn't duplicate");
  reader.feed('p', 100);
  check(reader.feed('!', 101) == bench::FrameEvent::Stop, "bang immediate stop");
  for (const char c : std::string("arm\n")) {
    check(reader.feed(c, 102) == bench::FrameEvent::None, "discard tail after stop");
  }
  reader.feed('a', 200);
  check(reader.tick(1199) == bench::FrameEvent::None, "partial before timeout");
  check(reader.tick(1200) == bench::FrameEvent::Timeout, "partial timeout stops");
  check(reader.feed('\n', 1201) == bench::FrameEvent::None, "timeout resync");
  for (unsigned int i = 0; i < 79; ++i) reader.feed('x', 1300);
  check(reader.feed('x', 1300) == bench::FrameEvent::Invalid, "overflow stops");
  for (unsigned int i = 0; i < 10000; ++i) {
    check(reader.feed('a', 1300) == bench::FrameEvent::None, "overflow tail ignored");
  }
  reader.feed('\n', 1301);
  check(reader.feed('\0', 1400) == bench::FrameEvent::Invalid, "NUL injection rejected");
  reader.feed('\n', 1401);
  reader.feed('a', 1500);
  reader.disconnect();
  for (const char c : std::string("rm\n")) {
    check(reader.feed(c, 1501) == bench::FrameEvent::None, "disconnect drops partial");
  }
  reader.feed('a', 0xffffff00u);
  check(reader.tick(0x000002e7u) == bench::FrameEvent::None, "line wrap before timeout");
  check(reader.tick(0x000002e8u) == bench::FrameEvent::Timeout, "line wrap timeout");
}

void serialLinkTests() {
  bench::SerialLink link;
  check(!link.observe(false, false, 0), "boot without USB doesn't invent a disconnect");
  check(!link.observe(true, false, 1), "ready link");
  check(!link.observe(false, false, 2), "single false readiness sample ignored");
  check(!link.observe(false, true, 501), "RX proves live link even with false TX hint");
  check(!link.observe(false, false, 1000), "loss before debounce boundary ignored");
  check(link.observe(false, false, 1001), "continuous loss triggers stop");
  check(!link.observe(false, false, 1002), "loss emitted once");
  check(!link.observe(false, true, 1003), "fresh RX reconnects without ready hint");
  bench::LineReader reader;
  check(!reader.pending(), "no partial input at clean reconnect");
  if (reader.pending()) reader.disconnect();
  for (char c : std::string("status")) reader.feed(c, 1004);
  check(reader.feed('\n', 1004) == bench::FrameEvent::Line,
        "first complete command at clean reconnect is accepted");
  check(command(reader.line()).action == bench::Action::Status, "fresh status retained");
  link.observe(true, false, 0xffffff00u);
  check(!link.observe(false, false, 0x000000f3u), "link debounce across millis wrap");
  check(link.observe(false, false, 0x000000f4u), "link loss at wrapped deadline");
}

int main() {
  parserTests();
  stateTests();
  frameTests();
  serialLinkTests();
  std::cout << "PASS | " << checks << " checks | parser, pulse, arm, wrap, serial framing\n";
}
