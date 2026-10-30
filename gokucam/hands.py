"""Hand and face geometry shared by every sign.

Hands are MediaPipe's 21 landmarks, as a (21, 2) float32 array of pixel coordinates in
the detection image (see `models.DETECT_SIZE`). y grows downwards.

    4 thumb tip        8 index tip      12 middle tip      16 ring tip      20 pinky tip
    3                  7                11                 15               19
    2                  6 index PIP      10 middle PIP      14 ring PIP      18 pinky PIP
    1                  5 index MCP       9 middle MCP      13 ring MCP      17 pinky MCP
                                    0 wrist
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# finger -> (PIP joint, tip)
FINGERS = {
    "index": (INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_PIP, RING_TIP),
    "pinky": (PINKY_PIP, PINKY_TIP),
}

HOLD_SECONDS = 0.35     # how long a sign must be held before it fires
DROP_TOLERANCE = 0.15   # detection gaps tolerated while holding a sign
RELEASE_SECONDS = 0.5   # the sign must be released before it can fire again


@dataclass
class Face:
    forehead: np.ndarray  # (x, y) in detection-image pixels
    width: float          # face bounding-box width, the natural unit for distances to the face
    seen_at: float


def face_from_detection(detection, size: tuple[int, int], now: float) -> Face:
    """BlazeFace keypoints: right eye, left eye, nose tip, mouth, right ear, left ear.
    The forehead is extrapolated above the eyes along the mouth -> eyes axis, so it
    follows head tilt."""
    w, h = size
    kp = np.array([[k.x * w, k.y * h] for k in detection.keypoints[:4]], np.float32)
    eyes = (kp[0] + kp[1]) / 2
    eye_distance = float(np.linalg.norm(kp[0] - kp[1]))
    up = eyes - kp[3]
    norm = float(np.linalg.norm(up))
    up = up / norm if norm > 1e-6 else np.array([0.0, -1.0], np.float32)
    return Face(eyes + up * eye_distance * 0.8, float(detection.bounding_box.width), now)


def dist(hand: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(hand[a] - hand[b]))


def hand_scale(hand: np.ndarray) -> float:
    """Wrist -> middle-finger knuckle: a size reference that does not depend on the pose.
    Every threshold on a hand is a multiple of it, so signs work near or far from the camera."""
    return max(dist(hand, WRIST, MIDDLE_MCP), 1.0)


def finger_extended(hand: np.ndarray, finger: str) -> bool:
    """A straight finger puts its tip clearly farther from the wrist than its middle joint."""
    pip, tip = FINGERS[finger]
    return dist(hand, WRIST, tip) > 1.1 * dist(hand, WRIST, pip)


def finger_folded(hand: np.ndarray, finger: str) -> bool:
    """A folded finger brings its tip back to (or below) the level of its middle joint."""
    pip, tip = FINGERS[finger]
    return dist(hand, WRIST, tip) < 1.05 * dist(hand, WRIST, pip)


def index_up(hand: np.ndarray) -> bool:
    """Index extended, its tip clearly above the wrist."""
    return finger_extended(hand, "index") and hand[WRIST][1] - hand[INDEX_TIP][1] > hand_scale(hand)


def best_hand(hands: list[np.ndarray], checks: Callable[[np.ndarray], dict[str, bool]]) -> dict[str, bool]:
    """Checks of the hand that meets the most criteria (the sign is made if one hand meets them all)."""
    if not hands:
        return {"a hand in view": False}
    return max((checks(hand) for hand in hands), key=lambda result: sum(result.values()))


class GestureTracker:
    """Fires once per held sign, then re-arms when the sign is released."""

    def __init__(self) -> None:
        self.started: float | None = None
        self.last_seen = -1e9
        self.armed = True

    def update(self, detected: bool, now: float) -> bool:
        if detected:
            if self.started is None or now - self.last_seen > DROP_TOLERANCE:
                self.started = now
            self.last_seen = now
            if self.armed and now - self.started >= HOLD_SECONDS:
                self.armed = False
                return True
        elif now - self.last_seen > RELEASE_SECONDS:
            self.armed = True
            self.started = None
        return False

    def progress(self, now: float) -> float:
        if not self.armed or self.started is None or now - self.last_seen > DROP_TOLERANCE:
            return 0.0
        return min((now - self.started) / HOLD_SECONDS, 1.0)
