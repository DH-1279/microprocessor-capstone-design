import sys
import os
import cv2
import math
import queue
import time
from pathlib import Path

# PortAudio(sounddevice)가 hw 대신 plughw를 사용하도록 강제
os.environ["PA_ALSA_PLUGHW"] = "1"

# 1. 모듈 임포트를 위한 최상위 경로 추가
PROJECT_ROOT = "/home/dh1279/Desktop/kimGA_raspi/microprocessor-capstone-design/03. 김가네 캡스톤 디자인/Raspi5_vision"
sys.path.append(PROJECT_ROOT)

import haptic_mvp.vision
from haptic_mvp.vision import HailoDetector, CameraSource
from haptic_mvp.models import Detection
from hand_tracker import HandTracker, draw_hand
from audio import AudioService
import speech  # 문법 및 자모 편집 거리 기반 음성 파서

# ==========================================
# [긴급 패치] 신뢰도 및 박스 좌표 오차 우회
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

# 시동어 후보군 (tiny 모델의 발음 뭉개짐 대비)
WAKE_WORDS = ("캡스톤", "갭스톤", "캡손", "캐스톤", "캡스", "캡스턴")
WAKE_THRESHOLD = 0.35

# 손 중심과 물건 중심이 만났다고 판정할 거리 임계값 (픽셀 단위)
REACH_THRESHOLD = 45.0
# 손 중심으로 사용할 MediaPipe 랜드마크 인덱스 (9번: 중지 뿌리 관절 / 손바닥 중앙)
HAND_CENTER_IDX = 9

def get_korean_name(label: str) -> str:
    if label in TARGET_MAPPING:
        return TARGET_MAPPING[label]["name"]
    return label

def check_wake_word(text: str) -> bool:
    """speech.py의 자모 거리 계산 및 조사 제거를 활용해 시동어('캡스톤') 감지"""
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
                
    # 띄어쓰기가 잘못된 짧은 발화(예: "캡 스톤") 결합 후 재확인
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

def main():
    config = {
        "object_model": HEF_PATH,
        "labels_file": LABELS_PATH,
        "score_threshold": 0.3,
        "max_objects": 20,
        "infer_timeout_ms": 1500,
        
        "hand_model": HAND_MODEL_PATH,
        "hand": "Any",
        "worn_hand": "Any",
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
        "whisper_model": "/home/dh1279/Desktop/kimGA_raspi/microprocessor-capstone-design/03. 김가네 캡스톤 디자인/Raspi5_vision/models/ggml-tiny.bin",
        "tts_executable": "/usr/bin/espeak-ng",
        "playback_executable": "/usr/bin/aplay",
        "cache_dir": "~/.cache/kimgane-haptic/tts",
        "output_device": "plughw:3,0",
        "input_device": 1 
    }

    command_queue = queue.Queue()
    audio_service = AudioService(audio_config, on_text=command_queue.put_nowait)
    audio_service.start()

    print("초기화 완료. 시동어 '캡스톤'을 기다리는 중...")
    audio_service.say("시스템이 준비되었습니다. 캡스톤이라고 말씀해주세요.")

    state = "WAIT_WAKE"
    target_eng_label = None
    
    locked_target_box = None     
    locked_center = None         
    missing_frame_count = 0      

    try:
        with CameraSource(config) as cam, \
             HailoDetector(config) as detector, \
             HandTracker(config) as tracker:
             
            window_name = "Capstone Vision Controller"
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name, 50, 50)
            
            while True:
                read_result = cam.read()
                if read_result is None:
                    print("프레임을 읽어올 수 없습니다.")
                    break
                    
                frame, timestamp = read_result
                
                obj_detections = detector.detect(frame)
                hand_result = tracker.detect(frame, timestamp)

                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                height, width = display_frame.shape[:2]

                current_detected_labels = {det.label for det in obj_detections} if obj_detections else set()
                current_korean_list = [get_korean_name(l) for l in current_detected_labels]

                command_text = None
                if not command_queue.empty():
                    command_text = command_queue.get_nowait()
                    print(f"\n🗣️ [음성 인식]: {command_text}")

                # ====================================================
                # 1. 시동어 대기 상태 (WAIT_WAKE)
                # ====================================================
                if state == "WAIT_WAKE":
                    if command_text and check_wake_word(command_text):
                        print("👉 시동어 '캡스톤' 감지됨!")
                        state = "WAIT_TARGET"
                        audio_service.say("찾을 물건을 말해주세요.")

                # ====================================================
                # 2. 목표 물건 선택 상태 (WAIT_TARGET) - speech.py 연동
                # ====================================================
                elif state == "WAIT_TARGET":
                    if command_text:
                        # speech.py의 'scanning' 상태는 ("target", "list", "stop")을 허용함
                        intent = speech.parse_input(command_text, TARGET_MAPPING, state="scanning")
                        print(f"🔍 [파싱 결과] kind={intent.kind}, name={intent.name}, reason={intent.reason}, detail={intent.detail}")

                        if intent.kind == "target" and intent.name:
                            matched = intent.name
                            if matched in current_detected_labels:
                                target_eng_label = matched
                                state = "SEARCHING"
                                target_kor = get_korean_name(matched)
                                
                                found_target = next((det for det in obj_detections if det.label == target_eng_label), None)
                                if found_target:
                                    locked_target_box = found_target.box
                                    x1, y1, x2, y2 = locked_target_box
                                    locked_center = (int(((x1 + x2) / 2) * width), int(((y1 + y2) / 2) * height))
                                missing_frame_count = 0

                                print(f"🎯 타겟 설정 완료: {target_kor} (매칭 별칭: {intent.detail})")
                                audio_service.say(f"{target_kor}을(를) 찾아 안내합니다.")
                            else:
                                target_kor = get_korean_name(matched)
                                print(f"⚠️ 화면에 '{target_kor}' 없음 (현재 감지됨: {current_korean_list})")
                                audio_service.say(f"현재 화면에 {target_kor}이 없습니다. 다시 말씀해 주세요.")

                        elif intent.kind == "control":
                            if intent.name == "list":
                                if current_korean_list:
                                    items_str = ", ".join(current_korean_list)
                                    audio_service.say(f"현재 주변에 {items_str}이 있습니다.")
                                else:
                                    audio_service.say("현재 인식된 물체가 없습니다.")
                            elif intent.name == "stop":
                                print("⏹️ 대기 상태로 복귀합니다.")
                                state = "WAIT_WAKE"
                                audio_service.say("취소되었습니다. 다시 부르시려면 캡스톤이라고 말씀해주세요.")
                        else:
                            # speech.py에 정의된 거절 사유별 안내 음성 출력
                            refusal_msg = speech.refusal_speech(intent)
                            if refusal_msg:
                                audio_service.say(refusal_msg)

                # ====================================================
                # 3. 타겟 탐색 및 손 추적 상태 (SEARCHING)
                # ====================================================
                elif state == "SEARCHING":
                    # 탐색 중 음성 명령 처리 ("정지", "목록", "다시")
                    if command_text:
                        intent = speech.parse_input(command_text, TARGET_MAPPING, state="guiding")
                        if intent.kind == "control":
                            if intent.name == "stop":
                                print("⏹️ 사용자 요청으로 안내를 중지합니다.")
                                audio_service.say("안내를 중지합니다.")
                                state = "WAIT_WAKE"
                                target_eng_label = None
                                locked_target_box = None
                                locked_center = None
                                missing_frame_count = 0
                                continue
                            elif intent.name == "list":
                                if current_korean_list:
                                    audio_service.say(f"주변에 {', '.join(current_korean_list)}이 있습니다.")
                                else:
                                    audio_service.say("인식된 물체가 없습니다.")
                            elif intent.name == "repeat" and target_eng_label:
                                audio_service.say(f"현재 {get_korean_name(target_eng_label)}을 찾고 있습니다.")

                    found_target = next((det for det in obj_detections if det.label == target_eng_label), None)
                    target_kor = get_korean_name(target_eng_label)
                    landmarks = getattr(tracker, "last_landmarks", None)
                    hand_near_target = False

                    if found_target:
                        locked_target_box = found_target.box
                        x1, y1, x2, y2 = locked_target_box
                        locked_center = (int(((x1 + x2) / 2) * width), int(((y1 + y2) / 2) * height))
                        missing_frame_count = 0
                    else:
                        missing_frame_count += 1

                    if locked_center is not None:
                        center_x, center_y = locked_center
                        hand_center = None
                        dist_to_target = float("inf")

                        # [방법 A] 9번 랜드마크(중지 뿌리 관절)를 손의 중심으로 계산
                        if landmarks is not None:
                            hand_cx = int(landmarks[HAND_CENTER_IDX].x * width)
                            hand_cy = int(landmarks[HAND_CENTER_IDX].y * height)
                            hand_center = (hand_cx, hand_cy)

                            dist_to_target = math.hypot(center_x - hand_cx, center_y - hand_cy)
                            if dist_to_target <= REACH_THRESHOLD:
                                hand_near_target = True

                        # 손의 중심과 물건의 중심이 만나면(임계값 이하) 즉시 획득 성공 판정
                        if hand_near_target:
                            print(f"✨ [획득 성공!] 손 중심과 {target_kor} 중심이 만났습니다! (거리: {dist_to_target:.1f}px)")
                            audio_service.say(f"{target_kor}을(를) 찾았습니다.")
                            state = "WAIT_WAKE"
                            target_eng_label = None
                            locked_target_box = None
                            locked_center = None
                            missing_frame_count = 0
                            continue

                        # 물건이 장시간(45프레임) 안 보일 때 타겟 해제
                        if found_target is None and missing_frame_count > 45:
                            print(f"⚠️ {target_kor}을(를) 장시간 찾을 수 없어 타겟을 해제합니다.")
                            audio_service.say("물건을 찾을 수 없습니다. 다시 말씀해 주세요.")
                            state = "WAIT_TARGET"
                            target_eng_label = None
                            locked_target_box = None
                            locked_center = None
                            missing_frame_count = 0
                            continue

                        is_occluded = (found_target is None)
                        status_str = " (⚠️ 손에 가려짐)" if is_occluded else ""
                        print(f"\n🎯 [탐색 중{status_str}] 목표: {target_kor} | 물건 중심: ({center_x}, {center_y})")

                        # 손 중심 <-> 물건 중심 점선 및 거리 시각화
                        if hand_center is not None:
                            hand_cx, hand_cy = hand_center
                            overlay = display_frame.copy()

                            # 손 중심점에 보라색 원 표시
                            cv2.circle(overlay, (hand_cx, hand_cy), 7, (255, 0, 255), -1)
                            draw_dashed_line(overlay, (hand_cx, hand_cy), (center_x, center_y), color=(30, 30, 30), thickness=2)
                            
                            mid_x = int((hand_cx + center_x) / 2)
                            mid_y = int((hand_cy + center_y) / 2)
                            put_readable_text(overlay, f"{dist_to_target:.0f}px", (mid_x - 15, mid_y), font_scale=0.5)

                            cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

                            print(f"    ├ 📍 손 중심 좌표: ({hand_cx}, {hand_cy})")
                            print(f"    └ 📏 목표 중심과의 거리: {dist_to_target:.1f}px")
                        else:
                            print("    └ 🖐️ 손이 감지되지 않았습니다.")
                    else:
                        print(f"⚠️ {target_kor}의 위치를 찾을 수 없습니다.")

                # 화면 시각화 (인식된 물체들 + 좌측 하단 좌표 리스트)
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
                        is_target = (state == "SEARCHING" and det.label == target_eng_label)
                        box_color = (0, 255, 255) if is_target else (0, 255, 0)

                        cv2.rectangle(display_frame, pt1, pt2, box_color, 2)
                        cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(display_frame, f"{det.label}: {det.score:.2f}", (pt1[0], max(pt1[1] - 8, 15)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                        list_y = base_y - (i * line_height)
                        coord_info = f"{det.label} : ({cx}, {cy})"
                        put_readable_text(display_frame, coord_info, (15, list_y), font_scale=0.6, 
                                          text_color=(20, 20, 20), outline_color=(255, 255, 255), thickness=2)

                # 손에 가려져 실시간 감지는 안 되지만 고정된 타겟이 있을 때 주황색 박스 표시
                if state == "SEARCHING" and locked_target_box is not None and found_target is None:
                    x1, y1, x2, y2 = locked_target_box
                    pt1 = (int(x1 * width), int(y1 * height))
                    pt2 = (int(x2 * width), int(y2 * height))
                    cx, cy = locked_center

                    cv2.rectangle(display_frame, pt1, pt2, (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.circle(display_frame, (cx, cy), 6, (0, 165, 255), -1)
                    put_readable_text(display_frame, f"[LOCKED] {target_eng_label}", (pt1[0], max(pt1[1] - 8, 15)), 
                                      text_color=(0, 100, 255), outline_color=(255, 255, 255), font_scale=0.55, thickness=2)

                # 손 뼈대 및 랜드마크 시각화 함수
                if hand_result is not None:
                    draw_hand(cv2, display_frame, tracker, hand_result)

                # 상단 상태 표시줄
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
    finally:
        audio_service.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()