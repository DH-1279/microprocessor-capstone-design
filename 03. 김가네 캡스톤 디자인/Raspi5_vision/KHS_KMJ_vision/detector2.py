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

# 영어 모델 레이블을 한글 이름 및 별칭으로 매핑
TARGET_MAPPING = {
    "Earbud": {"name": "이어폰", "aliases": ["이어폰", "무선이어폰", "이어버드"]},
    "bottle": {"name": "물병", "aliases": ["물병", "병", "페트병"]},
    "cell phone": {"name": "휴대폰", "aliases": ["휴대폰", "스마트폰", "핸드폰"]},
    "cup": {"name": "컵", "aliases": ["컵", "머그컵", "종이컵"]},
    "remote": {"name": "리모컨", "aliases": ["리모컨", "리모콘"]},
    "Glasses": {"name": "안경", "aliases": ["안경", "돋보기"]},
    "Medicine": {"name": "약통", "aliases": ["약통", "약"]},
    "key": {"name": "열쇠", "aliases": ["열쇠", "키", "차키"]},
    "Pen": {"name": "볼펜", "aliases": ["볼펜", "펜", "연필"]},
    "Tumbler": {"name": "텀블러", "aliases": ["텀블러", "보온병"]},
    "Wallet": {"name": "지갑", "aliases": ["지갑", "카드지갑"]}
}

def get_korean_name(label: str) -> str:
    if label in TARGET_MAPPING:
        return TARGET_MAPPING[label]["name"]
    return label

def match_target(text: str, detected_labels: set) -> str | None:
    text = text.replace(" ", "").lower()
    for eng_label in detected_labels:
        info = TARGET_MAPPING.get(eng_label, {"name": eng_label, "aliases": [eng_label]})
        for alias in info["aliases"]:
            if alias.replace(" ", "").lower() in text:
                return eng_label
    return None

def main():
    config = {
        "object_model": HEF_PATH,
        "labels_file": LABELS_PATH,
        "score_threshold": 0.3, # 민감도를 높이기 위해 0.3으로 조정
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

    # 상태 정의: "WAIT_WAKE" -> "WAIT_TARGET" -> "SEARCHING"
    state = "WAIT_WAKE"
    target_eng_label = None

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
                
                # 비전 추론
                obj_detections = detector.detect(frame)
                hand_result = tracker.detect(frame, timestamp)

                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                height, width = display_frame.shape[:2]

                # 현재 화면에 인식된 객체 한글 이름 리스트업
                current_detected_labels = {det.label for det in obj_detections} if obj_detections else set()
                current_korean_list = [get_korean_name(l) for l in current_detected_labels]

                # [디버깅 추가] 현재 화면에 감지된 원본 영어 라벨과 점수 출력
                if obj_detections:
                    detected_str = ", ".join([f"{det.label}({det.score:.2f})" for det in obj_detections])
                    print(f"🔍 [실시간 감지됨]: {detected_str}")

                # STT 명령 처리
                command_text = None
                if not command_queue.empty():
                    command_text = command_queue.get_nowait()
                    print(f"\n🗣️ [음성 인식]: {command_text}")

                # 상태별 로직 처리
                if state == "WAIT_WAKE":
                    if command_text and "캡스톤" in command_text:
                        print("👉 시동어 '캡스톤' 감지됨!")
                        state = "WAIT_TARGET"
                        audio_service.say("찾을 물건을 말해주세요.")

                elif state == "WAIT_TARGET":
                    if command_text:
                        matched = match_target(command_text, current_detected_labels)
                        if matched:
                            target_eng_label = matched
                            state = "SEARCHING"
                            target_kor = get_korean_name(matched)
                            print(f"🎯 타겟 설정 완료: {target_kor}")
                            audio_service.say(f"{target_kor}을(를) 찾아 안내합니다.")
                        else:
                            audio_service.say("인식된 물체 중에 없습니다. 다시 말씀해 주세요.")

                elif state == "SEARCHING":
                    # 타겟이 여전히 화면에 있는지 확인
                    found_target = next((det for det in obj_detections if det.label == target_eng_label), None)
                    if found_target:
                        x1, y1, x2, y2 = found_target.box
                        center_x = int(((x1 + x2) / 2) * width)
                        center_y = int(((y1 + y2) / 2) * height)

                        # 손과의 거리 계산 (손이 인식될 경우)
                        landmarks = getattr(tracker, "last_landmarks", None)
                        if landmarks is not None:
                            tip_x = int(landmarks[8].x * width)
                            tip_y = int(landmarks[8].y * height)
                            distance = math.hypot(center_x - tip_x, center_y - tip_y)
                            print(f"📍 [탐색 중] {get_korean_name(target_eng_label)} | 손과 거리: {distance:.1f}px")
                    else:
                        print(f"⚠️ {get_korean_name(target_eng_label)}을(를) 화면에서 찾을 수 없습니다.")

                # 화면 시각화 (바운딩 박스 및 중앙 점 유지)
                if obj_detections:
                    for det in obj_detections:
                        x1, y1, x2, y2 = det.box
                        center_x = int(((x1 + x2) / 2) * width)
                        center_y = int(((y1 + y2) / 2) * height)
                        pt1 = (int(x1 * width), int(y1 * height))
                        pt2 = (int(x2 * width), int(y2 * height))

                        kor_name = get_korean_name(det.label)
                        is_target = (state == "SEARCHING" and det.label == target_eng_label)

                        box_color = (0, 255, 255) if is_target else (0, 255, 0)

                        cv2.rectangle(display_frame, pt1, pt2, box_color, 2)
                        # [요청 사항] 바운딩 박스 중앙에 점 찍기 유지
                        cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                        cv2.putText(display_frame, f"{kor_name}: {det.score:.2f}", (pt1[0], max(pt1[1] - 8, 15)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                if hand_result is not None:
                    draw_hand(cv2, display_frame, tracker, hand_result)

                # 상단에 현재 인식된 물체 리스트 및 상태 실시간 갱신 표시
                list_str = ", ".join(current_korean_list) if current_korean_list else "없음"
                status_msg = f"State: {state} | Detected: [{list_str}]"
                cv2.putText(display_frame, status_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
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