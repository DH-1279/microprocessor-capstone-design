import sys
import os
import cv2
import math
import queue
import time
from pathlib import Path

# PortAudio(sounddevice)가 hw 대신 plughw(리샘플링)를 사용하도록 강제하여 주파수 에러 우회
os.environ["PA_ALSA_PLUGHW"] = "1"

PROJECT_ROOT = "/home/dh1279/Desktop/kimGA_raspi/microprocessor-capstone-design/03. 김가네 캡스톤 디자인/Raspi5_vision"
sys.path.append(PROJECT_ROOT)

import haptic_mvp.vision
from haptic_mvp.vision import HailoDetector, CameraSource
from haptic_mvp.models import Detection, Observation
from hand_tracker import HandTracker, draw_hand

from audio import AudioService
from controller import Controller

# ==========================================
# [긴급 패치] NMS 오차 우회
# ==========================================
def patched_decode_hailo_nms(raw, labels, geometry, threshold=0.5, max_objects=20):
    found = []
    for label, rows in zip(labels, raw):
        for row in rows:
            if len(row) != 5: continue
            score = float(row[4])
            if not math.isfinite(score) or score < threshold: continue
            
            ymin, xmin = max(0.0, min(1.0, float(row[0]))), max(0.0, min(1.0, float(row[1])))
            ymax, xmax = max(0.0, min(1.0, float(row[2]))), max(0.0, min(1.0, float(row[3])))
            if xmin >= xmax or ymin >= ymax: continue

            box = haptic_mvp.vision.restore_box((ymin, xmin, ymax, xmax), geometry)
            if box is not None:
                found.append(Detection(label=label, box=box, score=score))
    return tuple(sorted(found, key=lambda item: item.score, reverse=True)[:max_objects])

haptic_mvp.vision.decode_hailo_nms = patched_decode_hailo_nms

# ==========================================
# 메인 비전 루프
# ==========================================
def main():
    BASE_DIR = Path(__file__).resolve().parent
    
    # 1. 비전 설정
    vision_config = {
        "object_model": str(BASE_DIR / "models" / "best.hef"),
        "labels_file": str(BASE_DIR / "config" / "labels.txt"),
        "score_threshold": 0.5, "max_objects": 20, "infer_timeout_ms": 1500,
        "hand_model": str(Path(PROJECT_ROOT) / "models" / "hand_landmarker.task"),
        "hand": "Left", "worn_hand": "Left", "hand_score_threshold": 0.6,
        "source": "picamera2", "width": 1280, "height": 720, "fps": 15, "device": 0
    }
    
    # 2. 오디오 설정 (장치 번호 및 절대 경로 적용)
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
    
    # 3. 컨트롤러(추적/음성 대사) 설정 (타겟 별칭 및 민감도 대폭 보완)
    controller_config = {
        "targets": {
            "Earbud": {"name": "이어폰", "aliases": ["무선 이어폰", "이어버드", "이어폰"]},
            "bottle": {"name": "물병", "aliases": ["병", "페트병", "물병"]},
            "cell phone": {"name": "휴대폰", "aliases": ["스마트폰", "핸드폰", "휴대폰"]},
            "cup": {"name": "컵", "aliases": ["머그컵", "종이컵", "컵"]},
            "remote": {"name": "리모컨", "aliases": ["리모컨", "리모콘", "리모칸"]},
            "Glasses": {"name": "안경", "aliases": ["돋보기안경", "돋보기", "안경"]},
            "Medicine": {"name": "약통", "aliases": ["약", "알약", "약통"]},
            "key": {"name": "열쇠", "aliases": ["차키", "키", "열쇠"]},
            "Pen": {"name": "볼펜", "aliases": ["펜", "연필", "볼펜"]},
            "Tumbler": {"name": "텀블러", "aliases": ["보온병", "물통", "텀블러"]},
            "Wallet": {"name": "지갑", "aliases": ["카드지갑", "동전지갑", "지갑"]}
        },
        "tracking_iou": 0.05,       # 기존 0.2에서 대폭 낮춰서 박스가 조금 흔들려도 같은 물체로 인정
        "ambiguity_margin": 0.1, 
        "max_frame_age_s": 2.0,     # 기존 0.5초에서 늘려 물체가 잠깐 가려져도 기억하도록 수정
        "object_score": 0.3,        # 스코어 허용치도 더 낮춤
        "hand_score": 0.4, 
        "stable_frames": 1,
        "image_aspect": 1280/720, "side_separation": 0.1,
        "near_distance": 0.2, "middle_distance": 0.5, "rotation_deg": 0, "flip_x": False,
        "strengths": {"near": 3, "middle": 2, "far": 1},
        "channel_map": {"forward": 0, "back": 1, "left": 2, "right": 3, "near": 4},
        "announce_interval_s": 5.0
    }

    # 서비스 인스턴스 초기화
    command_queue = queue.Queue()
    audio_service = AudioService(audio_config, on_text=command_queue.put_nowait)
    controller = Controller(controller_config)
    
    audio_service.start()
    audio_service.say("시스템이 준비되었습니다. 찾을 물건을 말씀해주세요.")
    print("초기화 완료. 카메라 및 추론 시작...")

    try:
        with CameraSource(vision_config) as cam, \
             HailoDetector(vision_config) as detector, \
             HandTracker(vision_config) as tracker:
             
            window_name = "Integrated Controller Vision"
            cv2.namedWindow(window_name)
            
            while True:
                now = time.monotonic()
                
                # 큐에서 STT 파싱된 텍스트 수신
                command_text = None
                if not command_queue.empty():
                    command_text = command_queue.get_nowait()
                    print(f"\n🗣️ [명령 수신]: {command_text}")

                read_result = cam.read()
                if read_result is None: break
                frame, timestamp = read_result
                
                # 비전 추론 (객체 및 손)
                obj_detections = detector.detect(frame)
                hand_result = tracker.detect(frame, timestamp)

                # 컨트롤러에 상태 전달 및 판단 요청
                obs = Observation(frame_id=getattr(cam, "frame_id", 0), captured_at=now, 
                                  objects=obj_detections, hand=hand_result)
                decision = controller.step(now, observation=obs, command=command_text, audio_busy=audio_service.speaking)
                
                # 컨트롤러가 내린 안내 음성명령을 스피커로 재생
                if decision.speech:
                    print(f"🔊 [TTS 출력]: {decision.speech}")
                    audio_service.say(decision.speech)

                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                height, width = display_frame.shape[:2]
                landmarks = getattr(tracker, "last_landmarks", None)

                # 화면 렌더링 및 손가락 거리 연산
                if obj_detections:
                    for det in obj_detections:
                        x1, y1, x2, y2 = det.box
                        center_x = int(((x1 + x2) / 2) * width)
                        center_y = int(((y1 + y2) / 2) * height)
                        pt1, pt2 = (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height))

                        # 현재 타겟 객체: 노란색 강조 + 중심점(빨간색 원) + 거리 계산
                        if controller.target_label and det.label == controller.target_label:
                            cv2.rectangle(display_frame, pt1, pt2, (0, 255, 255), 3)
                            cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)  # 중앙 점 복구
                            cv2.putText(display_frame, f"TARGET: {det.label} {det.score:.2f}", 
                                        (pt1[0], max(pt1[1] - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            if landmarks is not None:
                                dist_strs = []
                                for idx, name in zip((4, 8, 12, 16, 20), ("엄지", "검지", "중지", "약지", "새끼")):
                                    tip_x, tip_y = int(landmarks[idx].x * width), int(landmarks[idx].y * height)
                                    dist = math.hypot(center_x - tip_x, center_y - tip_y)
                                    dist_strs.append(f"{name}: {dist:.0f}px")
                                print(f"🎯 [타겟 발견] {det.label} | 손과의 거리: " + " / ".join(dist_strs))
                        
                        # 일반 객체: 초록색 상자
                        else:
                            cv2.rectangle(display_frame, pt1, pt2, (0, 255, 0), 2)
                            cv2.putText(display_frame, f"{det.label}: {det.score:.2f}", 
                                        (pt1[0], max(pt1[1] - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if hand_result is not None:
                    draw_hand(cv2, display_frame, tracker, hand_result)

                # 상단 상태 텍스트 표시
                status_text = f"Target: {controller.target_label if controller.target_label else 'None'} | State: {decision.state}"
                cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow(window_name, display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\n프로그램을 안전하게 종료합니다.")
    finally:
        audio_service.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()