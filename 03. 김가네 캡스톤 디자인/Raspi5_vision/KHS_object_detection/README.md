# KHS_object_detection (객체 인식 모듈)

스마트 글러브 프로젝트의 전방 객체 인식을 담당하는 모듈입니다.
YOLOv8 모델을 사용하여 입력 프레임을 분석하고, 0~1로 정규화된 Bounding Box 좌표를 반환합니다.

## 1. 입출력 규격 (Data Contract)
- **입력:** 통합 런타임에서 전달받은 단일 RGB 프레임 (`numpy.ndarray`)
- **출력:** `Detection(label, box, score)` 객체의 리스트.
  - `box`는 원본 영상 기준 `(x1, y1, x2, y2)` 형태로 0.0 ~ 1.0 범위로 정규화됨.

## 2. 모델 정보
- **사용 모델:** YOLOv8n (Custom Trained)
- **가중치 경로:** `models/best.pt` 
- *(참고: `.pt` 가중치 파일은 용량 문제로 `.gitignore`에 등록되어 있으므로 개인 로컬 환경에 별도 배치해야 합니다.)*

## 3. 인식 가능한 객체 클래스 (Trained Classes)
총 18종의 일상 객체를 타겟으로 학습을 진행했습니다.
- bottle, cell phone, cup, handbag, keyboard, laptop, mouse, remote, scissors, tie
- Glasses, key, Medicine, Tumbler, Pen, Wallet, watch, Earbud
ㄴhandbag, keyboard, laptop, mouse, scissors, tie, watch 를 제외한 11개 클래스로 다시 진행할 예정 