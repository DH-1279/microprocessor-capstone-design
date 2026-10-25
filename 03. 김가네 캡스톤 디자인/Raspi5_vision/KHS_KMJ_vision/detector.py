import sys
import cv2
import math
import threading
import queue
import subprocess
import time
from pathlib import Path

# 1. 경로 설정 및 모듈 임포트
PROJECT_ROOT = "/home/dh1279/Desktop/kimGA_raspi/microprocessor-capstone-design/03. 김가네 캡스톤 디자인/Raspi5_vision"
sys.path.append(PROJECT_ROOT)

import haptic_mvp.vision
from haptic_mvp.vision import HailoDetector, CameraSource
from haptic_mvp.models import Detection
from hand_tracker import HandTracker, draw_hand #[cite: 1, 2]
import speech #[cite: 3]

# STT -> 메인 비전 루프 통신 큐
command_queue = queue.Queue()

# 목표 대상 사전 (labels.txt 기준 업데이트)[cite: 3]
TARGETS = {
    "Earbud": {"name": "이어폰", "aliases": ["무선 이어폰", "이어버드"]},
    "bottle": {"name": "물병", "aliases": ["병", "페트병"]},
    "cell phone": {"name": "휴대폰", "aliases": ["스마트폰", "핸드폰"]},
    "cup": {"name": "컵", "aliases": ["머그컵", "종이컵"]},
    "remote": {"name": "리모컨", "aliases": ["리모콘"]},
    "Glasses": {"name": "안경", "aliases": ["돋보기안경", "돋보기"]},
    "Medicine": {"name": "약통", "aliases": ["약", "알약"]},
    "key": {"name": "열쇠", "aliases": ["차키", "키"]},
    "Pen": {"name": "볼펜", "aliases": ["펜", "연필"]},
    "Tumbler": {"name": "텀블러", "aliases": ["보온병", "물통"]},
    "Wallet": {"name": "지갑", "aliases": ["카드지갑", "동전지갑"]}
}

# ==========================================
# [긴급 패치] 신뢰도 및 박스 좌표 오차 우회[cite: 1]
# ==========================================
def patched_decode_hailo_nms(raw, labels, geometry, threshold=0.5, max_objects=20):
    found = []
    for label, rows in zip(labels, raw):
        for row in rows:
            if len(row) != 5: continue
            score = float(row[4])
            if not math.isfinite(score) or score < threshold: continue
            
            ymin = max(0.0, min(1.0, float(row[0])))
            xmin = max(0.0, min(1.0, float(row[1])))
            ymax = max(0.0, min(1.0, float(row[2])))
            xmax = max(0.0, min(1.0, float(row[3])))
            if xmin >= xmax or ymin >= ymax: continue

            box = haptic_mvp.vision.restore_box((ymin, xmin, ymax, xmax), geometry)
            if box is not None:
                found.append(Detection(label=label, box=box, score=score))
    return tuple(sorted(found, key=lambda item: item.score, reverse=True)[:max_objects])

haptic_mvp.vision.decode_hailo_nms = patched_decode_hailo_nms

# ==========================================
# 음성 출력 (TTS) 함수[cite: 4, 5]
# ==========================================
def play_tts(text):
    """espeak-ng를 사용해 텍스트를 WAV로 변환하고 aplay로 재생합니다."""
    try:
        tts_wav = "/tmp/kimgane-tts.wav"
        subprocess.run(["espeak-ng", "-b", "1", "-v", "ko", "-s", "155", "-w", tts_wav, text], check=True)
        subprocess.run(["aplay", tts_wav], check=True)
    except Exception as e:
        print(f"TTS 재생 실패: {e}")

# ==========================================
# 음성 인식 (STT) 및 파싱 스레드 (시동어 적용)[cite: 4, 5]
# ==========================================
def audio_pipeline_thread():
    """마이크 녹음 후 whisper.cpp로 텍스트를 추출하고 시동어 확인 및 파싱을 수행합니다."""
    print("🎙️ [음성 스레드] 대기 중... (시동어 기반 인식 루프 시작)")
    mic_wav = "/tmp/kimgane-mic.wav"
    whisper_bin = "/home/dh1279/Desktop/kimGA_raspi/whisper.cpp/build/bin/whisper-cli"
    model_bin = str(Path(PROJECT_ROOT) / "models" / "ggml-tiny.bin")

    WAKE_WORD = "캡스톤"
    is_awake = False

    play_tts(f"시스템이 준비되었습니다. '{WAKE_WORD}'이라고 부른 뒤 찾을 물건을 말씀해주세요.")

    while True:
        try:
            # 1. 3초간 마이크 녹음[cite: 4, 5]
            subprocess.run(["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", "3", mic_wav], 
                           stderr=subprocess.DEVNULL, check=True)
            
            # 2. Whisper.cpp STT 변환[cite: 5]
            result = subprocess.run([whisper_bin, "-m", model_bin, "-l", "ko", "-f", mic_wav, "-nt"], 
                                    capture_output=True, text=True)
            raw_text = result.stdout.strip()
            
            if not raw_text: continue
            print(f"\n🗣️ [사용자 발화]: {raw_text}")

            command_text = raw_text

            # 3. 시동어 감지
            if WAKE_WORD in raw_text:
                is_awake = True
                
                # speech.py 오류 방지를 위해 시동어 및 문장 부호 제거
                command_text = raw_text.replace(WAKE_WORD, "").strip()
                for mark in ".,!?;:": 
                    command_text = command_text.replace(mark, " ")
                command_text = command_text.strip()
                
                if not command_text:
                    play_tts("네, 말씀하세요.")
                    continue

            # 대기 상태일 때 일상 대화 무시
            if not is_awake:
                continue

            # 4. 명령어 파싱 및 의도 추출[cite: 3]
            intent = speech.parse(command_text, TARGETS, state="any")
            
            if intent.kind == "target":
                command_queue.put(intent.name)
                target_kr = TARGETS.get(intent.name, {}).get("name", intent.name)
                play_tts(f"{target_kr} 안내를 시작합니다.")
                is_awake = False  # 타겟 설정 완료 후 대기 모드 복귀
                
            elif intent.reason not in ("empty", "ok"):
                refusal_msg = speech.refusal_speech(intent)
                if refusal_msg:
                    play_tts(refusal_msg)
                is_awake = False  # 실패 시 안내 후 대기 모드 복귀

        except Exception as e:
            print(f"STT 스레드 에러: {e}")
            time.sleep(1)

# ==========================================
# 메인 비전 루프[cite: 1, 6]
# ==========================================
def main():
    BASE_DIR = Path(__file__).resolve().parent
    HEF_PATH = str(BASE_DIR / "models" / "best.hef")
    LABELS_PATH = str(BASE_DIR / "config" / "labels.txt")
    HAND_MODEL_PATH = str(Path(PROJECT_ROOT) / "models" / "hand_landmarker.task")

    config = {
        "object_model": HEF_PATH, "labels_file": LABELS_PATH,
        "score_threshold": 0.5, "max_objects": 20, "infer_timeout_ms": 1500,
        "hand_model": HAND_MODEL_PATH, "hand": "Left", "worn_hand": "Left", #[cite: 1, 2]
        "hand_score_threshold": 0.6,
        "source": "picamera2", "width": 1280, "height": 720, "fps": 15, "device": 0
    }

    # 오디오 스레드 백그라운드 실행
    threading.Thread(target=audio_pipeline_thread, daemon=True).start()

    current_target = None
    print("초기화 중... (카메라 및 두 모델 웜업)")

    try:
        with CameraSource(config) as cam, \
             HailoDetector(config) as detector, \
             HandTracker(config) as tracker: #[cite: 1, 2]
             
            window_name = "Integrated Vision & Voice"
            cv2.namedWindow(window_name)
            
            while True:
                # 1. 큐에서 갱신된 타겟 확인
                if not command_queue.empty():
                    current_target = command_queue.get()

                read_result = cam.read()
                if read_result is None: break
                frame, timestamp = read_result
                
                # 2. AI 추론 (Hailo 객체 탐지 및 MediaPipe 손 추적)[cite: 1, 6]
                obj_detections = detector.detect(frame)
                hand_result = tracker.detect(frame, timestamp)

                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                height, width = display_frame.shape[:2]
                landmarks = getattr(tracker, "last_landmarks", None)

                # 3. 객체 시각화 및 타겟 거리 계산
                if obj_detections:
                    for det in obj_detections:
                        x1, y1, x2, y2 = det.box
                        center_x = int(((x1 + x2) / 2) * width)
                        center_y = int(((y1 + y2) / 2) * height)
                        pt1, pt2 = (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height))

                        # 목표물인 경우: 노란색 강조 + 거리 계산
                        if current_target and det.label == current_target:
                            cv2.rectangle(display_frame, pt1, pt2, (0, 255, 255), 3)
                            cv2.putText(display_frame, f"TARGET: {det.label} {det.score:.2f}", 
                                        (pt1[0], max(pt1[1] - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            # 손가락 관절과의 거리 연산[cite: 1]
                            if landmarks is not None:
                                dist_strs = []
                                for idx, name in zip((4, 8, 12, 16, 20), ("엄지", "검지", "중지", "약지", "새끼")):
                                    tip_x, tip_y = int(landmarks[idx].x * width), int(landmarks[idx].y * height)
                                    dist = math.hypot(center_x - tip_x, center_y - tip_y)
                                    dist_strs.append(f"{name}: {dist:.0f}px")
                                print(f"🎯 [타겟 발견] {det.label} | 손과의 거리: " + " / ".join(dist_strs))
                        
                        # 일반 객체인 경우: 초록색 렌더링[cite: 1]
                        else:
                            cv2.rectangle(display_frame, pt1, pt2, (0, 255, 0), 2)
                            cv2.putText(display_frame, f"{det.label}: {det.score:.2f}", 
                                        (pt1[0], max(pt1[1] - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 4. 손 상태 렌더링[cite: 2]
                if hand_result is not None:
                    draw_hand(cv2, display_frame, tracker, hand_result)

                # 화면 상단 상태 표시
                status_text = f"Target: {current_target if current_target else 'None (Waiting for Wake-word)'}"
                cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow(window_name, display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\n프로그램을 안전하게 종료합니다.")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()