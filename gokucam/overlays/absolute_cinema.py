"""Absolute cinema: raise both open palms above your head, as in the meme.

The picture turns black and white, red stage curtains close behind you from both sides
while a valance with a gold fringe drops in at the top, and "ABSOLUTE CINEMA" appears
once the curtains are closed. Lower your hands and it all plays backwards, back to the
live video: every step depends only on the overlay's fade level.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..fx import blend, fractal_noise, paste, smoothstep, text_sprite, tight_mask
from ..hands import FINGERS, INDEX_MCP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP, RING_MCP, WRIST, finger_extended, hand_scale
from ..jutsus.base import Scene, SignInput
from .base import Overlay

TITLE = "ABSOLUTE CINEMA"
CURTAIN_LIGHT = np.array([62, 52, 205], np.float32)   # BGR: velvet catching the light
CURTAIN_DARK = np.array([12, 8, 62], np.float32)      # BGR: deep in the folds
GOLD = (40, 170, 225)                                 # BGR: valance fringe
FONTS = (
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "/Library/Fonts/Georgia Bold.ttf",
    "C:/Windows/Fonts/georgiab.ttf",
    "C:/Windows/Fonts/timesbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
)


# --------------------------------------------------------------------------- sign

def palm_center(hand: np.ndarray) -> np.ndarray:
    return hand[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0)


def raised_palms_checks(signs: SignInput) -> dict[str, bool]:
    """Two open hands (all four fingers extended), fingertips above the forehead, one on
    each side."""
    if len(signs.hands) < 2:
        return {"two hands": False}
    a, b = sorted(signs.hands, key=lambda hand: float(hand[MIDDLE_TIP][1]))[:2]  # the two highest
    above = signs.face.forehead[1] if signs.face is not None else 0.4 * 360
    apart = abs(float(palm_center(a)[0] - palm_center(b)[0])) > (hand_scale(a) + hand_scale(b))
    return {
        "two hands": True,
        "palms open (x2)": all(finger_extended(hand, f) for hand in (a, b) for f in FINGERS),
        "hands above the head": bool(a[MIDDLE_TIP][1] < above and b[MIDDLE_TIP][1] < above),
        "one on each side": apart,
    }


# --------------------------------------------------------------------------- stage

def velvet(h: int, w: int, rng: np.random.Generator, flip: bool) -> np.ndarray:
    """A red velvet curtain panel (float32 BGR): deep vertical folds, a soft sheen, darker
    near the top and the bottom, and a dark hem on its inner edge (the side that moves)."""
    x = np.arange(w, dtype=np.float32)
    folds = np.zeros(w, np.float32)
    for period, weight in ((0.11, 0.5), (0.06, 0.3), (0.035, 0.2)):
        folds += weight * np.cos(2 * np.pi * x / (period * w * 2 * rng.uniform(0.85, 1.15)) + rng.uniform(0, 6.3))
    sheen = 0.5 + 0.5 * folds                                       # 0..1 across the folds
    y = np.arange(h, dtype=np.float32)[:, None] / h
    height_shade = 0.75 + 0.25 * np.sin(np.pi * np.clip(y * 1.1, 0, 1))
    streaks = fractal_noise((h, w), rng, ((30, 0.6), (80, 0.4)), stretch=(12, 1)) - 0.5
    shade = np.clip(sheen[None, :] ** 1.4 * height_shade + 0.12 * streaks, 0, 1)
    hem = np.clip(1 - (w - 1 - x) / (0.025 * w), 0, 1)              # inner edge, darker
    shade = shade * (1 - 0.6 * hem[None, :])
    panel = CURTAIN_DARK + (CURTAIN_LIGHT - CURTAIN_DARK) * shade[..., None]
    return panel[:, ::-1] if flip else panel


class Stage:
    """Curtains and valance generated once for a frame size."""

    def __init__(self, w: int, h: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.panel_w = int(0.56 * w)                                 # the two panels overlap when closed
        self.left = velvet(h, self.panel_w, rng, flip=False).astype(np.uint8)
        self.right = velvet(h, self.panel_w, rng, flip=True).astype(np.uint8)
        rows = np.arange(h, dtype=np.float32)
        self.wave = (0.012 * w * np.sin(rows / h * 2 * np.pi * 1.5 + rng.uniform(0, 6.3))).astype(np.float32)
        self.valance_h = int(0.11 * h)
        self.valance = self.make_valance(w, rng)
        self.vignette = self.make_vignette(w, h)

    def make_valance(self, w: int, rng: np.random.Generator) -> np.ndarray:
        """Top drape (BGRA): velvet with a row of swags along its bottom and a gold fringe."""
        h = self.valance_h
        sprite = np.zeros((h + h // 3, w, 4), np.uint8)
        sprite[:h, :, :3] = velvet(h, w, rng, flip=False)[:, ::-1].astype(np.uint8)
        sprite[:h, :, 3] = 255
        swag = max(w // 7, 1)
        for x in range(-swag // 2, w + swag, swag):                  # swags hanging below the band
            cv2.ellipse(sprite, (x + swag // 2, h - 2), (swag // 2, h // 3), 0, 0, 180, (24, 18, 120, 255), -1, cv2.LINE_AA)
            cv2.ellipse(sprite, (x + swag // 2, h - 2), (swag // 2, h // 3), 0, 0, 180, (*GOLD, 255), max(h // 14, 2), cv2.LINE_AA)
        cv2.line(sprite, (0, h - 2), (w, h - 2), (*GOLD, 255), max(h // 14, 2), cv2.LINE_AA)
        return sprite

    @staticmethod
    def make_vignette(w: int, h: int) -> np.ndarray:
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
        return np.clip((r - 0.75) / 0.6, 0, 1).astype(np.float32) * 0.55

    def curtains(self, cover: float) -> tuple[np.ndarray, np.ndarray]:
        """Both panels drawn `cover` (0..1) of the way to the centre: (BGR layer, alpha)."""
        h, panel_w = self.left.shape[:2]
        w = self.valance.shape[1]
        reach = cover * self.panel_w                                 # how far each panel has come in
        cols = np.arange(w, dtype=np.float32)[None, :]
        edge = reach + self.wave[:, None] * cover                     # a slightly wavy leading edge
        left_alpha = np.clip(edge - cols + 0.5, 0, 1)
        right_alpha = np.clip(cols - (w - 1 - edge) + 0.5, 0, 1)
        layer = np.zeros((h, w, 3), np.uint8)
        width = int(min(np.ceil(reach + np.abs(self.wave).max()) + 1, panel_w, w))
        if width > 0:
            layer[:, :width] = self.left[:, panel_w - width:]        # the panel slides in with its folds
            layer[:, w - width:] = np.where(right_alpha[:, w - width:, None] >= left_alpha[:, w - width:, None],
                                            self.right[:, :width], layer[:, w - width:])
        return layer, np.maximum(left_alpha, right_alpha).astype(np.float32)


def black_and_white(image: np.ndarray) -> np.ndarray:
    """Slightly contrasted grey, like an old film still."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.15, beta=-16)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- overlay

class AbsoluteCinema(Overlay):
    name = "absolute_cinema"
    title = "Absolute cinema"
    short = "Cinema"
    key = "c"
    hint = "both open palms raised above your head"
    fade_in_seconds = 1.6
    fade_out_seconds = 1.2
    linger_seconds = 0.3     # lower your hands and it plays backwards right away

    def __init__(self) -> None:
        self.stage: Stage | None = None
        self.seed = 0

    def checks(self, signs: SignInput) -> dict[str, bool]:
        return raised_palms_checks(signs)

    def start(self, scene: Scene) -> None:
        w, h = scene.size
        if self.stage is None or self.stage.valance.shape[1] != w:
            self.stage = Stage(w, h, scene.seed)

    def render(self, image: np.ndarray, scene: Scene, elapsed: float, level: float) -> np.ndarray:
        if level <= 0.001:
            return image
        if self.stage is None:
            self.start(scene)
        h, w = image.shape[:2]
        gray = smoothstep(0.0, 0.4, level)
        cover = smoothstep(0.15, 0.85, level)
        drop = smoothstep(0.05, 0.6, level)
        title = smoothstep(0.8, 1.0, level)

        out = cv2.addWeighted(image, 1 - gray, black_and_white(image), gray, 0)
        behind = 1.0 - tight_mask(scene.mask)                        # curtains close right behind you
        if cover > 0.001:
            layer, alpha = self.stage.curtains(cover)
            out = blend(out, layer, alpha * behind)
        valance = self.stage.valance
        top = int(-valance.shape[0] + drop * valance.shape[0])      # drops in from above the frame
        if drop > 0.001:
            region = out[max(top, 0):top + valance.shape[0]]
            part = valance[valance.shape[0] - region.shape[0]:]
            a = part[:, :, 3].astype(np.float32) / 255 * behind[max(top, 0):top + valance.shape[0]]
            out[max(top, 0):top + valance.shape[0]] = blend(region, np.ascontiguousarray(part[:, :, :3]), a)
        out = blend(out, np.zeros_like(out), self.stage.vignette * gray)
        if title > 0.001:
            sprite = text_sprite(TITLE, TITLE, max(int(h * 0.09), 12), FONTS, spacing=0.12)
            paste(out, sprite, (w - sprite.shape[1]) // 2, int(h * 0.93) - sprite.shape[0], title)
        return out
