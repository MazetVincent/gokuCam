"""MediaPipe models: download on first run, then person / hand / face detection."""

from __future__ import annotations

import os
import time
import urllib.request
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
}


def ensure_models() -> None:
    """Downloads the three official MediaPipe models from Google the first time."""
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
    """Person segmentation, hand landmarks and face detection on each frame (video mode,
    with tracking), plus an on-demand hand detection on the still image."""

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
        self._t0 = time.monotonic()
        self._last_ts = -1
        self._image: mp.Image | None = None

    def __enter__(self) -> Detectors:
        return self

    def __exit__(self, *_exc) -> None:
        for task in (self.segmenter, self.hands, self.face, self.hands_still):
            task.close()

    def hands_on_still_image(self) -> list[np.ndarray]:
        """Hands of the last processed image, detected without video tracking."""
        if self._image is None:
            return []
        return _landmarks(self.hands_still.detect(self._image).hand_landmarks)

    def process(self, small_bgr: np.ndarray):
        """Returns (raw person mask, hands, largest face detection or None)."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))
        self._image = image
        ts = max(self._last_ts + 1, int((time.monotonic() - self._t0) * 1000))
        self._last_ts = ts

        masks = self.segmenter.segment_for_video(image, ts).confidence_masks
        # 1 mask (person) or 2 (background, person) depending on the model version
        raw = (np.squeeze(masks[-1].numpy_view()).astype(np.float32) if masks
               else np.zeros(DETECT_SIZE[::-1], np.float32))
        hands = _landmarks(self.hands.detect_for_video(image, ts).hand_landmarks)
        detections = self.face.detect_for_video(image, ts).detections
        face = max(detections, key=lambda d: d.bounding_box.width, default=None)
        return raw, hands, face
