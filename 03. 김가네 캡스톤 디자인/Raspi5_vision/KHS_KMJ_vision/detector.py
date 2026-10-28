import sys
import os
import cv2
import math
import queue
import time
import traceback
import uuid
import subprocess
from pathlib import Path

# PortAudio(sounddevice)가 hw 대신 plughw를 사용하도록 강제
os.environ["PA_ALSA_PLUGHW"] = "1"

# 1. 모듈 임포트를 위한 최상위 경로 추가
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(PROJECT_ROOT)

import haptic_mvp.vision
from haptic_mvp.vision import HailoDetector, CameraSource
from haptic_mvp.models import Detection
from hand_tracker import HandTracker, draw_hand
from audio import AudioService
import speech  # 문법 및 자모 편집 거리 기반 음성 파서
from ble_sender.integration import DetectorBLE, detector_arguments

# ==========================================
# [긴급 패치 1] HailoRT physical.identify().device_architecture 우회
# ==========================================
class _DummyHailoIdentity:
    device_architecture = "HAILO8"
    board_name = "Hailo-8"
    serial_number = "0000"

    def __str__(self):
        return self.device_architecture

try:
    import hailo_platform
    for _mod in (hailo_platform, getattr(hailo_platform, "pyhailort", None), getattr(hailo_platform, "_pyhailort", None)):
        if _mod is None:
            continue
        for _cls_name in ("Device", "VDevice", "PhysicalDevice"):
            _cls = getattr(_mod, _cls_name, None)
            if _cls is not None and not hasattr(_cls, "identify"):
                setattr(_cls, "identify", lambda self, *a, **kw: _DummyHailoIdentity())
except Exception:
    pass

# ==========================================
# [긴급 패치 2] 신뢰도 및 박스 좌표 오차 우회
# ==========================================
def patched_decode_hailo_nms(raw, labels, geometry, threshold=0.5, max_objects=20):
    found = []
    for label, rows in zip(labels, raw):
        for row in rows:
            if len(row) != 5:
                continue
            
            score = float(row[4])
            if not math.isfinite(score):
                continue
            score = max(0.0, min(1.0, score)) 
            if score < threshold:
                continue
            
            ymin = max(0.0, min(1.0, float(row[0])))
            xmin = max(0.0, min(1.0, float(row[1])))
            ymax = max(0.0, min(1.0, float(row[2])))
            xmax = max(0.0, min(1.0, float(row[3])))
            
            if xmin >= xmax or ymin >= ymax:
                continue

            box = haptic_mvp.vision.restore_box((ymin, xmin, ymax, xmax), geometry)
            if box is not None:
                found.append(Detection(label=label, box=box, score=score))
                
    return tuple(sorted(found, key=lambda item: item.score, reverse=True)[:max_objects])

haptic_mvp.vision.decode_hailo_nms = patched_decode_hailo_nms
# ==========================================

# 2. 파일 경로 및 한글 라벨 매핑 설정
BASE_DIR = Path(__file__).resolve().parent
HEF_PATH = str(BASE_DIR / "models" / "best.hef")
LABELS_PATH = str(BASE_DIR / "config" / "labels.txt")
HAND_MODEL_PATH = str(Path(PROJECT_ROOT) / "models" / "hand_landmarker.task")

TARGET_MAPPING = {
    "Earbud": {"name": "이어폰", "aliases": ["이어폰", "무선이어폰", "이어버드"]},
    "bottle": {"name": "물병", "aliases": ["물병", "병", "페트병"]},
    "cell phone": {"name": "휴대폰", "aliases": ["휴대폰", "스마트폰", "핸드폰", "폰"]},
    "cup": {"name": "컵", "aliases": ["컵", "머그컵", "종이컵"]},
    "remote": {"name": "리모컨", "aliases": ["리모컨", "리모콘"]},
    "Glasses": {"name": "안경", "aliases": ["안경", "돋보기"]},
    "Medicine": {"name": "약통", "aliases": ["약통", "약"]},
    "key": {"name": "열쇠", "aliases": ["열쇠", "키", "차키"]},
    "Pen": {"name": "볼펜", "aliases": ["볼펜", "펜", "연필"]},
    "Tumbler": {"name": "텀블러", "aliases": ["텀블러", "보온병"]},
    "Wallet": {"name": "지갑", "aliases": ["지갑", "카드지갑"]}
}

WAKE_WORDS = ("캡스톤", "갭스톤", "캡손", "캐스톤", "캡스", "캡스턴")
WAKE_THRESHOLD = 0.35

REACH_THRESHOLD = 50.0        
REQUIRED_REACH_FRAMES = 5     
MAX_JUMP_DIST = 150.0         
HAND_MISSING_ALERT_INTERVAL = 5.0  # 손 미인식 시 안내 주기 (5초)

def get_korean_name(label: str) -> str:
    if label in TARGET_MAPPING:
        return TARGET_MAPPING[label]["name"]
    return label

def format_desk_items_speech(korean_list: list[str]) -> str:
    """현재 책상 위 인식된 물건 목록 안내 문구를 생성합니다."""
    if korean_list:
        items_str = ", ".join(korean_list)
        return f"현재 책상위엔 {items_str}이 있습니다."
    return "현재 책상위엔 인식된 물건이 없습니다."

def format_prompt_target_speech(korean_list: list[str]) -> str:
    """(2)번 상태 진입 시 안내할 기본 문구를 생성합니다."""
    return f"찾을 물건을 말해주세요. {format_desk_items_speech(korean_list)}"

def check_wake_word(text: str) -> bool:
    if not text:
        return False
    cleaned = text.strip()
    if "캡스톤" in cleaned:
        return True
    
    words = cleaned.split()
    for word in words:
        bare = speech.strip_particle(word)
        for wake in WAKE_WORDS:
            if speech.word_distance(bare, wake) <= WAKE_THRESHOLD:
                return True
                
    joined = "".join(words)
    if len(joined) <= 6:
        bare_joined = speech.strip_particle(joined)
        for wake in WAKE_WORDS:
            if speech.word_distance(bare_joined, wake) <= WAKE_THRESHOLD:
                return True
    return False

def draw_dashed_line(img, pt1, pt2, color=(30, 30, 30), thickness=2, dash_len=8, gap_len=6):
    dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist == 0:
        return
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist

    curr = 0
    while curr < dist:
        start_x = int(pt1[0] + dx * curr)
        start_y = int(pt1[1] + dy * curr)
        end_curr = min(curr + dash_len, dist)
        end_x = int(pt1[0] + dx * end_curr)
        end_y = int(pt1[1] + dy * end_curr)

        cv2.line(img, (start_x, start_y), (end_x, end_y), (220, 220, 220), thickness + 2, cv2.LINE_AA)
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)
        curr += dash_len + gap_len

def put_readable_text(img, text, pos, font_scale=0.5, text_color=(20, 20, 20), outline_color=(245, 245, 245), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

def main(argv=None):
    arguments = detector_arguments(argv)
    ble = DetectorBLE(arguments)
    config = {
        "object_model": HEF_PATH,
        "labels_file": LABELS_PATH,
        "score_threshold": 0.2,
        "max_objects": 20,
        "infer_timeout_ms": 1500,
        
        "hand_model": HAND_MODEL_PATH,
        "hand": "Any",
        "worn_hand": "Left",
        "hand_score_threshold": 0.5,
        
        "source": "picamera2", 
        "width": 1280,
        "height": 720,
        "fps": 15,
        "device": 0
    }

    audio_config = {
        "enabled": True,
        "whisper_executable": "/home/dh1279/Desktop/kimGA_raspi/whisper.cpp/build/bin/whisper-cli",
        "whisper_model": str(Path(PROJECT_ROOT) / "models" / "ggml-tiny.bin"),
        "tts_executable": "/usr/bin/espeak-ng",
        "playback_executable": "/usr/bin/aplay",
        "tts_voice": "M1",  # Supertonic 보이스 스타일 (M1, F1 등)
        "cache_dir": "~/.cache/kimgane-haptic/tts",
        "output_device": "plughw:3,0",
        "input_device": 1 
    }

    command_queue = queue.Queue()
    audio_service = AudioService(audio_config, on_text=command_queue.put_nowait)
    state = "WAIT_WAKE"
    target_eng_label = None
    lock_id = None
    frame_id = 0
    
    locked_target_box = None     
    locked_center = None         
    missing_frame_count = 0      
    reach_frame_count = 0        
    no_hand_since = None  # 손 미인식 시작 시각 기록용 타이머

    fingertip_indices = (4, 8, 12, 16, 20)
    tip_names = ("엄지", "검지", "중지", "약지", "새끼")

    try:
        ble.start()
        # 기존 볼륨 설정을 유지하되 믹서 응답 때문에 초기화가 멈추지 않게 한다.
        try:
            subprocess.run(["amixer", "-c", "3", "set", "PCM", "100%"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            subprocess.run(["amixer", "set", "Master", "100%"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
        except Exception:
            pass
        audio_service.start()
        print("초기화 완료. 시동어 '캡스톤'을 기다리는 중...")
        audio_service.say("시스템이 준비되었습니다. 캡스톤이라고 말씀해주세요.")
        with CameraSource(config) as cam, \
             HailoDetector(config) as detector, \
             HandTracker(config) as tracker:
             
            window_name = "Capstone Vision Controller"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # 디스플레이 화면에 꽉 차게 전체 화면 설정
            cv2.moveWindow(window_name, 50, 50)
            
            while True:
                if not ble.check():
                    print("BLE 터미널 종료 요청: 전체 프로그램을 종료합니다.")
                    break
                read_result = cam.read()
                if read_result is None:
                    ble.stop()
                    print("프레임을 읽어올 수 없습니다.")
                    break
                    
                frame, timestamp = read_result
                frame_id += 1
                
                obj_detections = detector.detect(frame)
                hand_result = tracker.detect(frame, timestamp)
                frame_fresh = 0 <= time.monotonic() - timestamp <= arguments.max_age
                landmarks = (tracker.last_observed_landmarks
                             if hand_result is not None and frame_fresh else None)
                found_target = None

                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                height, width = display_frame.shape[:2]

                current_detected_labels = {det.label for det in obj_detections} if obj_detections else set()
                current_korean_list = [get_korean_name(l) for l in current_detected_labels]

                command_text = None
                if not command_queue.empty():
                    command_text = command_queue.get_nowait()
                    print(f"\n🗣️ [음성 인식]: {command_text}")

                # (1) 시동어("캡스톤") 대기 상태
                if state == "WAIT_WAKE":
                    if command_text and check_wake_word(command_text):
                        print("👉 시동어 '캡스톤' 감지됨!")
                        state = "WAIT_TARGET"
                        # (2) 시동어 감지 시 안내 멘트 + 현재 인식된 물건 안내
                        audio_service.say(format_prompt_target_speech(current_korean_list))

                # (2) 찾을 물건 입력 대기 상태
                elif state == "WAIT_TARGET":
                    if command_text:
                        intent = speech.parse_input(command_text, TARGET_MAPPING, state="scanning")
                        print(f"🔍 [파싱 결과] kind={intent.kind}, name={intent.name}, reason={intent.reason}, detail={intent.detail}")

                        if intent.kind == "target" and intent.name:
                            matched = intent.name
                            if matched in current_detected_labels:
                                target_eng_label = matched
                                lock_id = uuid.uuid4().hex
                                state = "SEARCHING"
                                target_kor = get_korean_name(matched)
                                
                                found_target = next((det for det in obj_detections if det.label == target_eng_label), None)
                                if found_target:
                                    locked_target_box = found_target.box
                                    x1, y1, x2, y2 = locked_target_box
                                    locked_center = (int(((x1 + x2) / 2) * width), int(((y1 + y2) / 2) * height))
                                missing_frame_count = 0
                                reach_frame_count = 0
                                no_hand_since = time.monotonic()  # 탐색 진입 시 손 미인식 타이머 시작

                                print(f"🎯 타겟 설정 완료: {target_kor} (매칭 별칭: {intent.detail})")
                                audio_service.say(f"{target_kor}을(를) 찾아 안내합니다.")
                            else:
                                target_kor = get_korean_name(matched)
                                print(f"⚠️ 화면에 '{target_kor}' 없음 (현재 감지됨: {current_korean_list})")
                                audio_service.say(f"현재 화면에 {target_kor}이 없습니다. 다시 말씀해 주세요.")

                        elif intent.kind == "control":
                            # (3) "리스트" 명령 시 현재 책상 위 물건 안내
                            if intent.name == "list":
                                audio_service.say(format_desk_items_speech(current_korean_list))
                            elif intent.name == "stop":
                                print("⏹️ 대기 상태로 복귀합니다.")
                                state = "WAIT_WAKE"
                                audio_service.say("취소되었습니다. 다시 부르시려면 캡스톤이라고 말씀해주세요.")
                        else:
                            refusal_msg = speech.refusal_speech(intent)
                            if refusal_msg:
                                audio_service.say(refusal_msg)

                # (4) 목표 물건 추적 상태
                elif state == "SEARCHING":
                    if command_text:
                        intent = speech.parse_input(command_text, TARGET_MAPPING, state="guiding")
                        if intent.kind == "control":
                            # (5) 추적 중간에 "취소" 등의 명령 시 안내 종료 후 처음 대기 상태(WAIT_WAKE)로 복귀
                            if intent.name == "stop":
                                ble.stop()
                                print("⏹️ 사용자 요청으로 안내를 중지하고 처음 대기 상태로 복귀합니다.")
                                state = "WAIT_WAKE"
                                target_eng_label = None
                                locked_target_box = None
                                locked_center = None
                                missing_frame_count = 0
                                reach_frame_count = 0
                                no_hand_since = None
                                audio_service.say("안내를 중지합니다. 다시 부르시려면 캡스톤이라고 말씀해주세요.")
                                continue
                            # (3) 추적 중에도 "리스트"를 외치면 동일하게 안내
                            elif intent.name == "list":
                                audio_service.say(format_desk_items_speech(current_korean_list))
                            elif intent.name == "repeat" and target_eng_label:
                                audio_service.say(f"현재 {get_korean_name(target_eng_label)}을 찾고 있습니다.")

                    candidates = [det for det in obj_detections if det.label == target_eng_label] if obj_detections else []
                    found_target = None

                    if candidates:
                        if locked_center is None:
                            found_target = candidates[0]
                        else:
                            def dist_from_locked(det):
                                x1, y1, x2, y2 = det.box
                                cx = ((x1 + x2) / 2) * width
                                cy = ((y1 + y2) / 2) * height
                                return math.hypot(locked_center[0] - cx, locked_center[1] - cy)

                            closest_det = min(candidates, key=dist_from_locked)
                            if dist_from_locked(closest_det) < MAX_JUMP_DIST:
                                found_target = closest_det

                    target_kor = get_korean_name(target_eng_label)
                    hand_near_target = False

                    # 손 인식 여부에 따른 5초 주기 안내 타이머 처리
                    now_mono = time.monotonic()
                    if landmarks is not None:
                        no_hand_since = None
                    else:
                        if no_hand_since is None:
                            no_hand_since = now_mono
                        elif now_mono - no_hand_since >= HAND_MISSING_ALERT_INTERVAL:
                            if audio_service.say("손이 인식되지 않습니다."):
                                no_hand_since = now_mono

                    if found_target:
                        locked_target_box = found_target.box
                        x1, y1, x2, y2 = locked_target_box
                        locked_center = (int(((x1 + x2) / 2) * width), int(((y1 + y2) / 2) * height))
                        missing_frame_count = 0
                    else:
                        missing_frame_count += 1

                    if locked_center is not None:
                        center_x, center_y = locked_center

                        if landmarks is not None and found_target is not None:
                            hand_cx = int(landmarks[9].x * width)
                            hand_cy = int(landmarks[9].y * height)
                            if math.hypot(center_x - hand_cx, center_y - hand_cy) < REACH_THRESHOLD:
                                hand_near_target = True

                        if hand_near_target:
                            reach_frame_count += 1
                        else:
                            reach_frame_count = 0

                        # (4) 손이 목표 물건에 도달할 때까지 진행 후 완료 처리 (찾은 경우 시동어 대기 상태로 복귀)
                        if reach_frame_count >= REQUIRED_REACH_FRAMES:
                            ble.stop()
                            print(f"✨ [획득 성공!] {target_kor}을(를) 집어 들었습니다! ({reach_frame_count}프레임 연속 도달)")
                            state = "WAIT_WAKE"
                            target_eng_label = None
                            locked_target_box = None
                            locked_center = None
                            missing_frame_count = 0
                            reach_frame_count = 0
                            no_hand_since = None
                            audio_service.say(f"{target_kor}을 찾았습니다. 다른 물건을 찾으려면 캡스톤을 말씀해주세요.")
                            continue

                        if found_target is None and not hand_near_target and missing_frame_count > 45:
                            ble.stop()
                            print(f"⚠️ {target_kor}을(를) 장시간 찾을 수 없어 타겟을 해제합니다.")
                            audio_service.say("물건을 찾을 수 없습니다. 다시 말씀해 주세요.")
                            state = "WAIT_TARGET"
                            target_eng_label = None
                            locked_target_box = None
                            locked_center = None
                            missing_frame_count = 0
                            reach_frame_count = 0
                            no_hand_since = None
                            continue

                        is_occluded = (found_target is None)
                        status_str = " (⚠️ 손에 가려짐)" if is_occluded else ""
                        print(f"\n🎯 [탐색 중{status_str}] 목표: {target_kor} | 중심: ({center_x}, {center_y})")

                        if landmarks is not None:
                            coord_strs = []
                            dist_strs = []
                            overlay = display_frame.copy()

                            for idx, name in zip(fingertip_indices, tip_names):
                                tip_x = int(landmarks[idx].x * width)
                                tip_y = int(landmarks[idx].y * height)
                                dist = math.hypot(center_x - tip_x, center_y - tip_y)
                                
                                coord_strs.append(f"{name}({tip_x},{tip_y})")
                                dist_strs.append(f"{name}: {dist:.1f}px")

                                draw_dashed_line(overlay, (tip_x, tip_y), (center_x, center_y), color=(30, 30, 30), thickness=2)
                                mid_x = int((tip_x + center_x) / 2)
                                mid_y = int((tip_y + center_y) / 2)
                                put_readable_text(overlay, f"{dist:.0f}px", (mid_x - 15, mid_y), font_scale=0.45)

                            cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

                            print(f"    ├ 📍 손가락 절대 좌표: " + ", ".join(coord_strs))
                            print(f"    └ 📏 목표와의 거리: " + " / ".join(dist_strs))
                        else:
                            print("    └ 🖐️ 손이 감지되지 않았습니다.")
                    else:
                        print(f"⚠️ {target_kor}의 위치를 찾을 수 없습니다.")

                ble.publish(
                    active=(state == "SEARCHING" and found_target is not None
                            and landmarks is not None),
                    target_id=lock_id,
                    frame_id=frame_id,
                    captured_at=timestamp,
                    size=(width, height),
                    landmarks=landmarks,
                    target_box=found_target.box if found_target is not None else None,
                )

                if obj_detections:
                    base_y = height - 20
                    line_height = 25

                    for i, det in enumerate(obj_detections):
                        x1, y1, x2, y2 = det.box
                        cx = int(((x1 + x2) / 2) * width)
                        cy = int(((y1 + y2) / 2) * height)
                        pt1 = (int(x1 * width), int(y1 * height))
                        pt2 = (int(x2 * width), int(y2 * height))

                        kor_name = get_korean_name(det.label)
                        is_target = (state == "SEARCHING" and det is found_target)
                        box_color = (0, 255, 255) if is_target else (0, 255, 0)

                        cv2.rectangle(display_frame, pt1, pt2, box_color, 2)
                        cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(display_frame, f"{det.label}: {det.score:.2f}", (pt1[0], max(pt1[1] - 8, 15)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                        list_y = base_y - (i * line_height)
                        coord_info = f"{det.label} : ({cx}, {cy})"
                        put_readable_text(display_frame, coord_info, (15, list_y), font_scale=0.6, 
                                          text_color=(20, 20, 20), outline_color=(255, 255, 255), thickness=2)

                if state == "SEARCHING" and locked_target_box is not None and found_target is None:
                    x1, y1, x2, y2 = locked_target_box
                    pt1 = (int(x1 * width), int(y1 * height))
                    pt2 = (int(x2 * width), int(y2 * height))
                    cx, cy = locked_center

                    cv2.rectangle(display_frame, pt1, pt2, (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.circle(display_frame, (cx, cy), 6, (0, 165, 255), -1)
                    put_readable_text(display_frame, f"[LOCKED] {target_eng_label}", (pt1[0], max(pt1[1] - 8, 15)), 
                                      text_color=(0, 100, 255), outline_color=(255, 255, 255), font_scale=0.55, thickness=2)

                if hand_result is not None:
                    draw_hand(cv2, display_frame, tracker, hand_result)

                list_str = ", ".join(current_detected_labels) if current_detected_labels else "None"
                status_msg = f"State: {state} | Detected: [{list_str}]"
                put_readable_text(display_frame, status_msg, (10, 30), font_scale=0.65, 
                                  text_color=(255, 255, 255), outline_color=(0, 0, 0), thickness=2)
                
                cv2.imshow(window_name, display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n프로그램을 안전하게 종료합니다.")
    except Exception as e:
        print(f"실행 중 에러가 발생했습니다: {e}")
        traceback.print_exc()
        return 1
    finally:
        try:
            ble.close()
        finally:
            audio_service.close()
            cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
