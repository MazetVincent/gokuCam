"""MediaPipe models: download on first run, then person / hand / face detection."""

from __future__ import annotations

import os
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DETECT_SIZE = (640, 360)  # detection runs on a downscaled copy of the frame

_GCS = "https://storage.googleapis.com/mediapipe-models/"
MODELS = {
    "segmenter": (
        MODEL_DIR / "selfie_segmenter_landscape.tflite",
        _GCS + "image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
    ),
    "hands": (
        MODEL_DIR / "hand_landmarker.task",
        _GCS + "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    ),
    "face": (
        MODEL_DIR / "blaze_face_short_range.tflite",
        _GCS + "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    ),
    "objects": (
        MODEL_DIR / "efficientdet_lite0.tflite",
        _GCS + "object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
    ),
}
OBJECT_LABELS = ["cup"]  # COCO categories worth looking for
STILL_HANDS_EVERY = 3    # the still-image hand detection is slow (~35 ms): reuse it for a few frames


@dataclass
class DetectedObject:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in detection-image pixels


@dataclass
class Detections:
    mask: np.ndarray                  # raw person probability, detection resolution
    hands: list[np.ndarray]           # (21, 2) landmarks per hand
    face: object | None               # largest MediaPipe face detection, or None
    objects: list[DetectedObject] = field(default_factory=list)


def ensure_models() -> None:
    """Downloads the official MediaPipe models from Google the first time."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in MODELS.values():
        if path.is_file() and path.stat().st_size > 50_000:
            continue
        print(f"[GokuCam] Downloading {path.name}...")
        partial = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
                while chunk := response.read(1 << 18):
                    out.write(chunk)
            os.replace(partial, path)
        except OSError as error:
            partial.unlink(missing_ok=True)
            raise SystemExit(f"Download failed ({error}). Check your connection.") from error


def _landmarks(result_landmarks) -> list[np.ndarray]:
    w, h = DETECT_SIZE
    return [np.array([[lm.x * w, lm.y * h] for lm in landmarks], np.float32) for landmarks in result_landmarks]


class Detectors:
    """Person segmentation, hand landmarks, face and object detection on each frame (video
    mode, with tracking), plus an on-demand hand detection on the still image."""

    def __init__(self) -> None:
        video = vision.RunningMode.VIDEO

        def base(name: str) -> mp_python.BaseOptions:
            return mp_python.BaseOptions(model_asset_path=str(MODELS[name][0]))

        self.segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=base("segmenter"), running_mode=video,
            output_confidence_masks=True, output_category_mask=False,
        ))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base("hands"), running_mode=video, num_hands=2,
            min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        ))
        self.face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=base("face"), running_mode=video, min_detection_confidence=0.65,
        ))
        # Clasped hands: video tracking merges them into one hand; a detection on the
        # still image, without tracking, finds both.
        self.hands_still = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base("hands"), running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
        ))
        # Object detection costs as much as everything else together, so it runs in a
        # background thread (on another core), only while a hand is in view: a mug moves
        # slowly, a result a few frames old is good enough.
        self.objects = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=base("objects"), running_mode=vision.RunningMode.IMAGE, max_results=3,
            score_threshold=0.3, category_allowlist=OBJECT_LABELS,
        ))
        self._object_worker = ThreadPoolExecutor(max_workers=1)
        self._object_job: Future | None = None
        self._objects: list[DetectedObject] = []
        self._t0 = time.monotonic()
        self._last_ts = -1
        self._image: mp.Image | None = None
        self._frame = 0
        self._still_frame = -STILL_HANDS_EVERY
        self._still_hands: list[np.ndarray] = []

    def __enter__(self) -> Detectors:
        return self

    def __exit__(self, *_exc) -> None:
        self._object_worker.shutdown(wait=True)
        for task in (self.segmenter, self.hands, self.face, self.hands_still, self.objects):
            task.close()

    def _detect_objects(self, image: mp.Image) -> list[DetectedObject]:
        return [DetectedObject(d.categories[0].category_name, d.categories[0].score,
                               (d.bounding_box.origin_x, d.bounding_box.origin_y,
                                d.bounding_box.origin_x + d.bounding_box.width,
                                d.bounding_box.origin_y + d.bounding_box.height))
                for d in self.objects.detect(image).detections]

    def _update_objects(self, image: mp.Image, hands_in_view: bool) -> None:
        """Collects the background detection when it is done and starts the next one."""
        job = self._object_job
        if job is not None and job.done():
            self._objects, job = job.result(), None
        if not hands_in_view:
            self._objects = []
        elif job is None:
            job = self._object_worker.submit(self._detect_objects, image)
        self._object_job = job

    def hands_on_still_image(self) -> list[np.ndarray]:
        """Hands of a recent image, detected without video tracking (refreshed every
        STILL_HANDS_EVERY frames: plenty to confirm a sign held for a third of a second)."""
        if self._image is None:
            return []
        if self._frame - self._still_frame >= STILL_HANDS_EVERY:
            self._still_hands = _landmarks(self.hands_still.detect(self._image).hand_landmarks)
            self._still_frame = self._frame
        return self._still_hands

    def process(self, small_bgr: np.ndarray) -> Detections:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))
        self._image = image
        self._frame += 1
        ts = max(self._last_ts + 1, int((time.monotonic() - self._t0) * 1000))
        self._last_ts = ts

        masks = self.segmenter.segment_for_video(image, ts).confidence_masks
        # 1 mask (person) or 2 (background, person) depending on the model version
        raw = (np.squeeze(masks[-1].numpy_view()).astype(np.float32) if masks
               else np.zeros(DETECT_SIZE[::-1], np.float32))
        hands = _landmarks(self.hands.detect_for_video(image, ts).hand_landmarks)
        faces = self.face.detect_for_video(image, ts).detections
        face = max(faces, key=lambda d: d.bounding_box.width, default=None)
        self._update_objects(image, bool(hands))
        return Detections(raw, hands, face, self._objects)

    def wait_for_objects(self) -> None:
        """Blocks until the running object detection is done (tools and tests only)."""
        if self._object_job is not None:
            self._objects, self._object_job = self._object_job.result(), None
