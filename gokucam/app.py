"""The application state machine: webcam frame -> detections -> signs -> effect -> output.

    plate ──(countdown, photo of the empty room)──> visible
    visible ──sign──> vanishing ──(effect done)──> hidden
    hidden  ──sign──> appearing ──(effect done)──> visible

While hidden, the call receives a frozen image (the empty room, or whatever the jutsu
left behind) with a little sensor grain; the live webcam is only used to watch for the
next sign.
"""

from __future__ import annotations

import cv2
import numpy as np

from .hands import INDEX_TIP, MIDDLE_TIP, Face, GestureTracker, face_from_detection
from .jutsus import Box, Jutsu, Scene, SignInput, default_jutsus
from .models import DETECT_SIZE

PLATE_COUNTDOWN = 5.0   # seconds to step out of the frame before the room photo
PLATE_FRAMES = 15       # the photo is the median of the last frames (removes noise and brief motion)
REVIEW_SECONDS = 4.0    # the photo is shown as a thumbnail so the user can check it
GRAIN_FRAMES = 6        # grain variants: the frozen image looks like a live feed
GRAIN_SIGMA = 1.5
FACE_MEMORY = 1.0       # hands often hide the face: remember the last one seen this long
MASK_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def refine_mask(raw: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    """Generous and stable person mask: better erase a bit of the room than a bit of you."""
    mask = cv2.resize(raw, DETECT_SIZE, interpolation=cv2.INTER_LINEAR)
    mask = np.clip((mask - 0.1) / 0.4, 0.0, 1.0)
    mask = cv2.dilate(mask, MASK_KERNEL)
    mask = cv2.GaussianBlur(mask, (0, 0), 3)
    if previous is not None:
        mask = np.maximum(mask, previous * 0.75)
    return mask


class GokuCam:
    def __init__(self, detectors, frame_size: tuple[int, int], jutsus: list[Jutsu] | None = None) -> None:
        self.detectors = detectors
        self.width, self.height = frame_size
        self.jutsus = {jutsu.name: jutsu for jutsu in (jutsus if jutsus is not None else default_jutsus())}
        self.trackers = {name: GestureTracker() for name in self.jutsus}
        self.checks: dict[str, dict[str, bool]] = {}  # last criteria of every sign, for --debug
        self.plate: np.ndarray | None = None          # photo of the empty room (uint8)
        self.hidden_image: np.ndarray | None = None   # what the call sees while you are invisible
        self.grain: list[np.ndarray] = []             # the same image with a little sensor grain
        self.plate_frames: list[np.ndarray] = []
        self.review_until = -1.0
        self.output_index = 0
        self.rng = np.random.default_rng()
        self.effect_seed = 0
        self.active: Jutsu | None = None              # jutsu whose effect is playing
        self.leftover: Box | None = None              # what the last departure left in hidden_image
        self.last_frame: np.ndarray | None = None
        self.prev_mask: np.ndarray | None = None      # person mask, detection resolution
        self.face: Face | None = None
        self.hands: list[np.ndarray] = []
        self.state = "plate"
        self.state_since = 0.0

    # ------------------------------------------------------------------ room photo

    def recapture(self, now: float) -> None:
        self.state, self.state_since = "plate", now
        self.plate_frames = []

    def set_plate(self, plate: np.ndarray, seed: int = 0) -> None:
        self.plate = plate
        self.set_hidden(plate, seed)

    def set_hidden(self, image: np.ndarray, seed: int = 0) -> None:
        self.hidden_image = image
        rng = np.random.default_rng(seed)
        self.grain = [np.clip(image + rng.normal(0, GRAIN_SIGMA, image.shape), 0, 255).astype(np.uint8)
                      for _ in range(GRAIN_FRAMES)]

    def capture_plate(self, frame: np.ndarray, now: float) -> None:
        """Countdown, then the photo is the median of the last frames. No detector decides
        when the room is empty: a painting or a chair can look like a person."""
        self.plate_frames = (self.plate_frames + [frame.copy()])[-PLATE_FRAMES:]
        if now - self.state_since >= PLATE_COUNTDOWN:
            self.set_plate(np.median(np.stack(self.plate_frames), axis=0).astype(np.uint8))
            self.plate_frames = []
            self.state, self.state_since = "visible", now
            self.review_until = now + REVIEW_SECONDS
            print("[GokuCam] Room photo taken (check the thumbnail; B to retake).")

    # ------------------------------------------------------------------ effects

    def full_mask(self) -> np.ndarray:
        if self.prev_mask is None:
            return np.zeros((self.height, self.width), np.float32)
        return cv2.resize(self.prev_mask, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def scene(self, frame: np.ndarray) -> Scene:
        return Scene(frame=frame, plate=self.plate, hidden=self.hidden_image, mask=self.full_mask(),
                     seed=self.effect_seed, rng=self.rng, leftover=self.leftover)

    def trigger(self, now: float, name: str) -> None:
        """Starts the effect of jutsu `name` (departure if visible, arrival if hidden)."""
        if self.state not in ("visible", "hidden") or self.last_frame is None:
            return
        arriving = self.state == "hidden"
        self.state = "appearing" if arriving else "vanishing"
        self.state_since = now
        self.active = self.jutsus[name]
        self.effect_seed = int(self.rng.integers(1 << 30))  # stable random layout for this effect
        self.active.start(self.scene(self.last_frame), arriving)

    def effect_seconds(self) -> float:
        return self.active.arrival_seconds if self.state == "appearing" else self.active.vanish_seconds

    # ------------------------------------------------------------------ main loop

    def detect(self, frame: np.ndarray, now: float) -> SignInput:
        small = cv2.resize(frame, DETECT_SIZE, interpolation=cv2.INTER_AREA)
        raw, self.hands, detection = self.detectors.process(small)
        self.prev_mask = refine_mask(raw, self.prev_mask)
        if detection is not None:
            self.face = face_from_detection(detection, DETECT_SIZE, now)
        face = self.face if self.face and now - self.face.seen_at < FACE_MEMORY else None
        return SignInput(self.hands, face, self.detectors.hands_on_still_image)

    def step(self, frame: np.ndarray, now: float) -> np.ndarray:
        """Processes one webcam frame and returns the frame to send to the call."""
        self.last_frame = frame
        signs = self.detect(frame, now)
        for name, jutsu in self.jutsus.items():
            self.checks[name] = jutsu.checks(signs)
            if self.trackers[name].update(all(self.checks[name].values()), now):
                self.trigger(now, name)

        if self.state == "plate":
            self.capture_plate(frame, now)
            return frame

        if self.state in ("vanishing", "appearing"):
            elapsed = (now - self.state_since) / self.effect_seconds()
            if elapsed < 1.0:
                return self.active.render(self.scene(frame), elapsed, self.state == "appearing")
            if self.state == "vanishing":
                self.set_hidden(self.active.hidden_image(self.scene(frame)))
                self.leftover = self.active.leftover_box()
                self.state = "hidden"
            else:
                self.state, self.leftover = "visible", None
        if self.state == "hidden":
            self.output_index += 1
            return self.grain[self.output_index % len(self.grain)]
        return frame

    # ------------------------------------------------------------------ preview overlay

    def gesture_progress(self, now: float) -> float:
        return max(tracker.progress(now) for tracker in self.trackers.values())

    def hud_lines(self, now: float) -> list[str]:
        if self.state == "plate":
            remaining = PLATE_COUNTDOWN - (now - self.state_since)
            return [f"Step out of the frame: room photo in {max(remaining, 0):0.0f} s"]
        if now < self.review_until:
            return ["Room photo (top right): are you in it? B to retake"]
        if self.state == "hidden":
            return ["INVISIBLE (frozen image) - make a sign to come back"]
        if self.state == "visible":
            return [f"{jutsu.title}: {jutsu.hint}" for jutsu in self.jutsus.values()]
        return []

    def debug_lines(self) -> list[str]:
        """One line per sign: OK, or the criteria still missing."""
        lines = []
        for name, jutsu in self.jutsus.items():
            missing = [label for label, ok in self.checks.get(name, {}).items() if not ok]
            lines.append(f"{jutsu.short}: OK" if not missing else f"{jutsu.short}: missing {', '.join(missing)}")
        return lines

    def debug_points(self) -> list[tuple[tuple[int, int], int, tuple[int, int, int]]]:
        sx, sy = self.width / DETECT_SIZE[0], self.height / DETECT_SIZE[1]
        points = []
        if self.face is not None:
            fx, fy = self.face.forehead
            points.append(((int(fx * sx), int(fy * sy)), int(0.55 * self.face.width * sx), (0, 200, 255)))
        for hand in self.hands:
            for x, y in hand[[INDEX_TIP, MIDDLE_TIP]]:
                points.append(((int(x * sx), int(y * sy)), 6, (80, 255, 80)))
        return points
