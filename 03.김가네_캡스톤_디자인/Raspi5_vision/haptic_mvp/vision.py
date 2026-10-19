"""Camera, local YOLO and MediaPipe adapters. Hardware imports are deliberately lazy.

All observations use normalized coordinates on the SAME RGB image. Hailo only
accepts RGB HEFs with one NMS-by-class output, not arbitrary YOLO tensors.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import math
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable

from .models import Detection, Hand, Observation


@dataclass(frozen=True)
class Letterbox:
    width: int
    height: int
    model_width: int
    model_height: int
    resized_width: int
    resized_height: int
    left: int
    top: int


def letterbox_geometry(width: int, height: int, model_width: int, model_height: int) -> Letterbox:
    if min(width, height, model_width, model_height) <= 0:
        raise ValueError("Frame and model dimensions must be positive")
    scale = min(model_width / width, model_height / height)
    resized_width = max(1, min(model_width, round(width * scale)))
    resized_height = max(1, min(model_height, round(height * scale)))
    return Letterbox(width, height, model_width, model_height, resized_width,
                     resized_height, (model_width - resized_width) // 2,
                     (model_height - resized_height) // 2)


def restore_box(yxyx: Any, geometry: Letterbox) -> tuple[float, float, float, float] | None:
    """Undo rounded resize + letterbox; reject padding-only and invalid boxes."""
    if len(yxyx) != 4:
        raise ValueError("Expected four normalized yxyx coordinates")
    ymin, xmin, ymax, xmax = (float(value) for value in yxyx)
    if not all(math.isfinite(v) for v in (ymin, xmin, ymax, xmax)):
        raise ValueError("Detection coordinates must be finite")
    if not (0 <= xmin <= xmax <= 1 and 0 <= ymin <= ymax <= 1):
        raise ValueError("Expected normalized, ordered NMS coordinates")
    g = geometry
    x1 = (xmin * g.model_width - g.left) / g.resized_width
    x2 = (xmax * g.model_width - g.left) / g.resized_width
    y1 = (ymin * g.model_height - g.top) / g.resized_height
    y2 = (ymax * g.model_height - g.top) / g.resized_height
    x1, y1, x2, y2 = (max(0.0, min(1.0, value)) for value in (x1, y1, x2, y2))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def decode_hailo_nms(raw: Any, labels: tuple[str, ...], geometry: Letterbox,
                     threshold: float = 0.5, max_objects: int = 20) -> tuple[Detection, ...]:
    """Hailo get_buffer(tf_format=False): one Nx5 array per class, no batch axis.

    Row order is ymin,xmin,ymax,xmax,score. Invalid layouts fail closed rather
    than being interpreted as a different model's coordinates.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != len(labels):
        raise ValueError("Hailo NMS class count/layout does not match labels_file")
    found = []
    for label, rows in zip(labels, raw):
        for row in rows:
            if len(row) != 5:
                raise ValueError("Hailo NMS must have exactly 5 values per detection")
            score = float(row[4])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Detection confidence must be finite and within 0..1")
            if score < threshold:
                continue
            box = restore_box(row[:4], geometry)
            if box is not None:
                found.append(Detection(label=label, box=box, score=score))
    return tuple(sorted(found, key=lambda item: item.score, reverse=True)[:max_objects])


def read_labels(path: str) -> tuple[str, ...]:
    labels = tuple(Path(path).read_text(encoding="utf-8-sig").splitlines())
    if not labels or any(not label or label != label.strip() for label in labels):
        raise ValueError("labels_file must contain one nonempty class name per line")
    if len(set(labels)) != len(labels):
        raise ValueError("labels_file has duplicate class names")
    return labels


def _local_file(value: Any, suffix: str) -> str:
    path = Path(str(value))
    if path.suffix.lower() != suffix or not path.is_file():
        raise ValueError(f"Local {suffix} model is missing: {path}; see VISION_SETUP.md")
    return str(path.resolve())


def validate_rgb(frame: Any) -> None:
    if (getattr(frame, "ndim", None) != 3 or frame.shape[2] != 3
            or min(frame.shape[:2]) <= 0 or str(frame.dtype) != "uint8"):
        raise ValueError("Camera must supply a nonempty HxWx3 uint8 RGB image")


def preprocess_rgb(frame: Any, model_width: int, model_height: int) -> tuple[Any, Letterbox]:
    import cv2
    import numpy as np

    validate_rgb(frame)
    height, width = frame.shape[:2]
    geometry = letterbox_geometry(width, height, model_width, model_height)
    resized = cv2.resize(frame, (geometry.resized_width, geometry.resized_height))
    output = np.full((model_height, model_width, 3), 114, dtype=np.uint8)
    output[geometry.top:geometry.top + geometry.resized_height,
           geometry.left:geometry.left + geometry.resized_width] = resized
    return np.ascontiguousarray(output), geometry


class HailoDetector:
    """HailoRT InferModel adapter, explicitly targeting the user's Hailo-10H."""

    def __init__(self, config: dict):
        self.config = config
        self.model_path = _local_file(config.get("object_model", ""), ".hef")
        self.labels = read_labels(config.get("labels_file", ""))
        self.resources = ExitStack()

    def __enter__(self):
        import numpy as np
        from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice
        from hailo_platform.pyhailort.pyhailort import FormatOrder

        try:
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            device = self.resources.enter_context(VDevice(params))
            physical_devices = device.get_physical_devices()
            if len(physical_devices) != 1:
                for physical in physical_devices:
                    physical.release()
                raise ValueError("Expected one Hailo-10H device")
            with physical_devices[0] as physical:
                architecture = str(physical.identify().device_architecture)
            if architecture != "HAILO10H":
                raise ValueError(f"AI HAT+ 2 needs HAILO10H; detected {architecture}")
            self.model = device.create_infer_model(self.model_path)
            self.model.set_batch_size(1)
            if len(self.model.inputs) != 1 or len(self.model.outputs) != 1:
                raise ValueError("Use a single RGB input / NMS output detection HEF")
            input_stream = self.model.input()
            output_stream = self.model.output()
            shape = tuple(input_stream.shape)
            if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0:
                raise ValueError(f"HEF input must be HxWx3 RGB, got {shape}")
            if input_stream.format.order != FormatOrder.NHWC:
                raise ValueError("Use the RGB HEF, not NV12/RGBX or another input layout")
            if (not output_stream.is_nms
                    or output_stream.format.order != FormatOrder.HAILO_NMS_BY_CLASS):
                raise ValueError("HEF needs HAILO_NMS_BY_CLASS; raw YOLO heads are unsupported")
            self.model_height, self.model_width = shape[:2]
            input_stream.set_format_type(FormatType.UINT8)
            output_stream.set_format_type(FormatType.FLOAT32)
            self.configured = self.resources.enter_context(self.model.configure())
            self.bindings = self.configured.create_bindings(output_buffers={
                output_stream.name: np.empty(output_stream.shape, dtype=np.float32)
            })
            return self
        except BaseException:
            self.resources.close()
            raise

    def detect(self, rgb: Any) -> tuple[Detection, ...]:
        frame, geometry = preprocess_rgb(rgb, self.model_width, self.model_height)
        self.bindings.input().set_buffer(frame)
        self.configured.run([self.bindings], timeout=int(self.config.get("infer_timeout_ms", 1500)))
        raw = self.bindings.output().get_buffer(tf_format=False)
        return decode_hailo_nms(raw, self.labels, geometry,
                                float(self.config.get("score_threshold", 0.5)),
                                int(self.config.get("max_objects", 20)))

    def __exit__(self, *args):
        return self.resources.__exit__(*args)


class UltralyticsDetector:
    """Explicit CPU fallback. YOLO is passed an existing .pt; never a model name."""

    def __init__(self, config: dict):
        self.config = config
        self.model_path = _local_file(config.get("object_model", ""), ".pt")

    def __enter__(self):
        from ultralytics import YOLO

        self.model = YOLO(self.model_path, task="detect")
        return self

    def detect(self, rgb: Any) -> tuple[Detection, ...]:
        import numpy as np

        # Ultralytics numpy sources are BGR; MediaPipe and our camera contract are RGB.
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        result = self.model.predict(source=bgr, device="cpu", verbose=False,
                                    conf=float(self.config.get("score_threshold", 0.5)),
                                    max_det=int(self.config.get("max_objects", 20)))[0]
        detections = []
        for box, score, class_id in zip(result.boxes.xyxyn.cpu().tolist(),
                                        result.boxes.conf.cpu().tolist(),
                                        result.boxes.cls.cpu().tolist()):
            normalized = restore_box((box[1], box[0], box[3], box[2]),
                                     letterbox_geometry(1, 1, 1, 1))
            confidence = float(score)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Invalid YOLO confidence")
            if normalized is not None:
                detections.append(Detection(label=result.names[int(class_id)],
                                            box=normalized, score=confidence))
        return tuple(detections)

    def __exit__(self, *args):
        self.model = None
        return False


def hand_from_result(result: Any, selected: str, threshold: float) -> Hand | None:
    """Reject ambiguous hands; handedness score is not landmark confidence."""
    candidates = []
    if len(result.hand_landmarks) != len(result.handedness):
        raise ValueError("MediaPipe landmark/handedness counts differ")
    for landmarks, categories in zip(result.hand_landmarks, result.handedness):
        if len(landmarks) != 21 or not categories:
            continue
        category = categories[0]
        handedness = category.category_name
        score = float(category.score)
        if (handedness not in {"Left", "Right"} or not math.isfinite(score)
                or not 0 <= score <= 1 or score < threshold):
            continue
        if selected != "Any" and selected != handedness:
            continue
        points = [(float(landmarks[i].x), float(landmarks[i].y)) for i in (0, 5, 9, 13, 17)]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for point in points for v in point):
            continue
        point = (sum(p[0] for p in points) / 5, sum(p[1] for p in points) / 5)
        candidates.append(Hand(point=point, score=score, handedness=handedness))
    return candidates[0] if len(candidates) == 1 else None


class MediaPipeHandTracker:
    def __init__(self, config: dict):
        self.model_path = _local_file(config.get("hand_model", ""), ".task")
        self.selected = config.get("hand", "Right")
        self.threshold = float(config.get("hand_score_threshold", 0.6))
        self.last_timestamp = -1

    def __enter__(self):
        import mediapipe as mp

        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self.threshold,
            min_hand_presence_confidence=self.threshold,
            min_tracking_confidence=self.threshold,
        )
        self.tracker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        return self

    def detect(self, rgb: Any, captured_at: float) -> Hand | None:
        # VIDEO timestamps must strictly increase even with sub-ms mock captures.
        timestamp = max(self.last_timestamp + 1, int(captured_at * 1000))
        self.last_timestamp = timestamp
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.tracker.detect_for_video(image, timestamp)
        return hand_from_result(result, self.selected, self.threshold)

    def __exit__(self, *args):
        self.tracker.close()
        return False


class CameraSource:
    """CSI via Picamera2 or OpenCV USB/local-video input; returns RGB."""

    def __init__(self, config: dict):
        self.config = config
        self.camera: Any = None
        self.capture: Any = None
        self.replay = False

    def __enter__(self):
        source = self.config.get("source", "picamera2")
        width, height = int(self.config.get("width", 640)), int(self.config.get("height", 480))
        fps = float(self.config.get("fps", 15))
        if source == "picamera2":
            from picamera2 import Picamera2

            self.camera = Picamera2(camera_num=int(self.config.get("device", 0)))
            try:
                # libcamera BGR888 produces RGB bytes; RGB888 produces BGR bytes.
                self.camera.configure(self.camera.create_video_configuration(
                    main={"size": (width, height), "format": "BGR888"},
                    controls={"FrameRate": fps}, buffer_count=4, queue=False))
                self.camera.start()
            except BaseException:
                self.camera.close()
                raise
        elif source == "opencv":
            import cv2

            device = self.config.get("device", 0)
            if not isinstance(device, int):
                path = Path(str(device))
                if not path.is_file():
                    raise ValueError("OpenCV device must be a USB index or existing local video")
                device = str(path.resolve())
                self.replay = True
            self.capture = cv2.VideoCapture(device)
            if not self.capture.isOpened():
                self.capture.release()
                raise RuntimeError(f"Could not open camera/video: {device}")
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.replay:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.capture.set(cv2.CAP_PROP_FPS, fps)
            self.replay_period = 1 / fps
        else:
            raise ValueError("vision.source must be picamera2 or opencv")
        return self

    def read(self) -> tuple[Any, float] | None:
        import numpy as np

        # Conservative host capture-start time. It does not claim an exact sensor
        # exposure time; CSI queue=False and the latest-frame reader avoid backlog.
        captured_at = time.monotonic()
        if self.camera is not None:
            frame = self.camera.capture_array("main")
        else:
            ok, bgr = self.capture.read()
            if not ok:
                if self.replay:
                    return None
                raise RuntimeError("USB camera frame read failed")
            frame = np.ascontiguousarray(bgr[:, :, ::-1])
        validate_rgb(frame)
        if self.config.get("mirror", False):
            frame = np.ascontiguousarray(frame[:, ::-1])
        return frame, captured_at

    def __exit__(self, *args):
        if self.camera is not None:
            self.camera.stop()
            self.camera.close()
        if self.capture is not None:
            self.capture.release()
        return False


class LatestFrames:
    """One-frame mailbox: slow inference cannot accumulate old camera frames."""

    def __init__(self, source: CameraSource):
        self.source = source
        self.frames: queue.Queue = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.finished = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._capture, name="camera-capture", daemon=True)

    def _capture(self):
        try:
            frame_id = 0
            while not self.stopped.is_set():
                start = time.monotonic()
                item = self.source.read()
                if item is None:
                    break
                frame_id += 1
                frame, captured_at = item
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put_nowait((frame_id, captured_at, frame))
                if self.source.replay:
                    self.stopped.wait(max(0, self.source.replay_period - (time.monotonic() - start)))
        except BaseException as error:
            self.error = error
        finally:
            self.finished.set()

    def __enter__(self):
        self.thread.start()
        return self

    def get(self, stop: threading.Event) -> tuple[int, float, Any] | None:
        while not stop.is_set():
            if self.error is not None:
                raise RuntimeError(f"Camera reader failed: {self.error}") from self.error
            try:
                return self.frames.get(timeout=0.1)
            except queue.Empty:
                if self.finished.is_set():
                    return None
        return None

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join(timeout=2)
        return False


def run_vision(config: dict, emit: Callable[[Observation], None], stop: threading.Event) -> None:
    """Blocking worker; exceptions propagate to the controller, which stops output."""
    detector_name = config.get("detector", "hailo")
    if detector_name not in {"hailo", "ultralytics"}:
        raise ValueError("vision.detector must be hailo or ultralytics")
    if config.get("hand", "Right") not in {"Left", "Right", "Any"}:
        raise ValueError("vision.hand must be Left, Right or Any")
    for key, default in (("width", 640.0), ("height", 480.0), ("fps", 15.0),
                         ("max_objects", 20.0), ("infer_timeout_ms", 1500.0)):
        value = float(config.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"vision.{key} must be positive and finite")
    for key, default in (("score_threshold", 0.5), ("hand_score_threshold", 0.6)):
        value = float(config.get(key, default))
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"vision.{key} must be within (0, 1]")
    detector_type = HailoDetector if detector_name == "hailo" else UltralyticsDetector
    with ExitStack() as resources:
        detector = resources.enter_context(detector_type(config))
        tracker = resources.enter_context(MediaPipeHandTracker(config))
        source = resources.enter_context(CameraSource(config))
        frames = resources.enter_context(LatestFrames(source))
        if config.get("show_preview", False):
            import cv2
            resources.callback(cv2.destroyAllWindows)
        while not stop.is_set():
            item = frames.get(stop)
            if item is None:
                break
            frame_id, captured_at, rgb = item
            actual_aspect = rgb.shape[1] / rgb.shape[0]
            configured_aspect = int(config.get("width", 640)) / int(config.get("height", 480))
            if not math.isclose(actual_aspect, configured_aspect, rel_tol=0.005):
                raise ValueError("Actual frame aspect differs from vision.width/height; "
                                 "match them and controller.image_aspect to the camera/video")
            objects = detector.detect(rgb)
            hand = tracker.detect(rgb, captured_at)
            if stop.is_set():
                break
            emit(Observation(frame_id=frame_id, captured_at=captured_at, objects=objects, hand=hand))
            if config.get("show_preview", False):
                preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                height, width = rgb.shape[:2]
                for detection in objects:
                    x1, y1, x2, y2 = detection.box
                    origin = (int(x1 * width), int(y1 * height))
                    cv2.rectangle(preview, origin, (int(x2 * width), int(y2 * height)), (0, 255, 0), 2)
                    cv2.putText(preview, detection.label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if hand is not None:
                    cv2.circle(preview, (int(hand.point[0] * width), int(hand.point[1] * height)), 7, (0, 0, 255), -1)
                cv2.imshow("KIMGANE vision (q: stop)", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop.set()
