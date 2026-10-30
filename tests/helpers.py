"""Synthetic hands and faces for the tests (detection-image pixels, y grows downwards)."""

import numpy as np

from gokucam.hands import Face
from gokucam.jutsus import SignInput

FACE_ON_HAND = Face(forehead=np.float32([295, 185]), width=120.0, seen_at=0.0)  # fingertips on the forehead
FACE_ABOVE = Face(forehead=np.float32([300, 40]), width=120.0, seen_at=0.0)     # face well above the hands
FACE_BELOW = Face(forehead=np.float32([300, 260]), width=120.0, seen_at=0.0)    # hand raised above the head


def make_hand(fingers_apart: bool = False, open_palm: bool = False, offset=(0.0, 0.0),
              index_only: bool = False) -> np.ndarray:
    """Upright hand, wrist at (300, 300), scale (wrist -> middle knuckle) = 50 px.
    Default: index + middle extended and together, ring + pinky folded."""
    hand = np.zeros((21, 2), np.float32)
    hand[0] = (300, 300)
    hand[1:5] = [(280, 290), (270, 275), (265, 262), (262, 250)]  # thumb
    for finger, x in enumerate((290, 300, 310, 320)):
        mcp = 5 + 4 * finger
        extended = finger < (1 if index_only else 2) or open_palm
        if fingers_apart and finger == 1:
            x += 40
        hand[mcp] = (x, 250)
        if extended:
            hand[mcp + 1:mcp + 4] = [(x, 220), (x, 205), (x, 190)]
        else:
            hand[mcp + 1:mcp + 4] = [(x, 235), (x, 250), (x, 262)]
    return hand + np.float32(offset)


def joined_hands(**kwargs) -> list[np.ndarray]:
    """Two hands with their index fingers up and touching (the substitution sign)."""
    return [make_hand(index_only=True, **kwargs), make_hand(index_only=True, offset=(10, 3), **kwargs)]


def signs(hands, face=None, still=None) -> SignInput:
    return SignInput(hands, face, (lambda: still) if still is not None else (lambda: []))
