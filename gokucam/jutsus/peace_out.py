"""Peace out: a V sign, and you vanish like a badly compressed video (the "bye" meme).

The person turns into a translucent ghost made of big compression blocks while the whole
picture takes a hit of pixelation; then only a few blocks remain, smearing sideways near
the head, and nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..fx import pixelate, smoothstep
from ..hands import INDEX_TIP, MIDDLE_TIP, best_hand, dist, finger_extended, finger_folded, hand_scale
from .base import Jutsu, Scene, SignInput

BLOCK = 20  # compression block size, in px for a 1280-wide frame


@dataclass
class Ghost:
    """The person frozen at the moment of the sign (the ghost no longer moves, as in the meme)."""
    frame: np.ndarray
    mask: np.ndarray   # float32 0..1, full resolution
    seed: int


def render_peace_out(frame: np.ndarray, plate: np.ndarray, ghost: Ghost, t: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Departure, t from 0 to 1. At t = 1 the frame is exactly `plate`."""
    if t <= 0.0:
        return frame
    if t >= 1.0:
        return plate
    h, w = plate.shape[:2]
    block = max(int(BLOCK * w / 1280), 4)
    gh, gw = max(h // block, 1), max(w // block, 1)

    # the whole call takes a hit of compression during the effect
    crunch = smoothstep(0.0, 0.06, t) * (1 - smoothstep(0.6, 0.95, t))
    background = plate
    if crunch > 0.01:
        background = cv2.addWeighted(plate, 1 - 0.5 * crunch, pixelate(plate, max(block // 2, 2)), 0.5 * crunch, 0)

    mask = cv2.resize(ghost.mask, (gw, gh), interpolation=cv2.INTER_AREA)
    color = cv2.resize(ghost.frame, (gw, gh), interpolation=cv2.INTER_AREA)
    # the ghost gets more and more transparent
    level = 1.0 - 0.4 * smoothstep(0.0, 0.08, t) - 0.25 * smoothstep(0.08, 0.45, t)
    # leftovers: every block dies at its own time, the top ones (the head) last longer
    order = np.random.default_rng(ghost.seed).random((gh, gw)) * (0.4 + 0.9 * np.linspace(0, 1, gh)[:, None])
    alive = order < 1.3 * (1 - smoothstep(0.45, 0.85, t))
    alpha = mask * level * alive * rng.uniform(0.85, 1.15, (gh, gw))  # blocks flicker
    smear = smoothstep(0.4, 0.8, t)
    if smear > 0.01:  # leftovers smear sideways, one row of blocks at a time
        shifts = np.round(rng.normal(0, 2.5 * smear, gh)).astype(int)
        color = np.stack([np.roll(row, s, axis=0) for row, s in zip(color, shifts)])
        alpha = np.stack([np.roll(row, s) for row, s in zip(alpha, shifts)])
    a = cv2.resize(np.clip(alpha, 0, 1).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)[..., None]
    c = cv2.resize(color, (w, h), interpolation=cv2.INTER_NEAREST)
    composite = (background * (1 - a) + c * a).astype(np.uint8)
    snap = smoothstep(0.0, 0.08, t)  # the live person turns into the ghost very quickly
    return cv2.addWeighted(frame, 1 - snap, composite, snap, 0) if snap < 1 else composite


def render_peace_in(frame: np.ndarray, hidden: np.ndarray, mask: np.ndarray, t: float, seed: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Arrival: blocks pop up, form a ghost, then the live person. For a "bad video"
    meme, playing the departure backwards is what looks right."""
    return render_peace_out(frame, hidden, Ghost(frame, mask, seed), 1.0 - t, rng)


def peace_hand_checks(hand: np.ndarray, face) -> dict[str, bool]:
    tips = (hand[INDEX_TIP] + hand[MIDDLE_TIP]) / 2
    return {
        "index extended": finger_extended(hand, "index"),
        "middle extended": finger_extended(hand, "middle"),
        "ring folded": finger_folded(hand, "ring"),
        "pinky folded": finger_folded(hand, "pinky"),
        "fingers spread": dist(hand, INDEX_TIP, MIDDLE_TIP) > 0.55 * hand_scale(hand),
        # two fingers *together on the forehead* is Instant Transmission
        "away from the forehead": face is None or float(np.linalg.norm(tips - face.forehead)) > 0.8 * face.width,
    }


class PeaceOut(Jutsu):
    name = "peace_out"
    title = "Peace out"
    short = "Peace"
    key = "p"
    hint = "V sign, away from the face"
    vanish_seconds = 0.8
    arrival_seconds = 0.6

    def __init__(self) -> None:
        self.ghost: Ghost | None = None

    def checks(self, signs: SignInput) -> dict[str, bool]:
        return best_hand(signs.hands, lambda hand: peace_hand_checks(hand, signs.face))

    def start(self, scene: Scene, arriving: bool) -> None:
        if not arriving:
            self.ghost = Ghost(scene.frame.copy(), scene.mask, scene.seed)

    def render(self, scene: Scene, t: float, arriving: bool) -> np.ndarray:
        if arriving:
            return render_peace_in(scene.frame, scene.hidden, scene.mask, t, scene.seed, scene.rng)
        return render_peace_out(scene.frame, scene.plate, self.ghost, t, scene.rng)
