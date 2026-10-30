"""This is fine: pick up your mug, and the room catches fire around you (the meme).

Cartoon flames appear one after the other at different places around you, always behind
you; a ceiling of grey smoke rolls in at the top of the screen; the fire casts a warm,
flickering light; and "This is fine." is typed letter by letter. It lasts as long as
you hold the mug.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ..fx import smoothstep
from ..jutsus.base import Box, Scene, SignInput
from .base import Overlay

CAPTION = "This is fine."
CAPTION_START = 1.4       # seconds before typing starts
CAPTION_CHAR = 0.08       # seconds per letter
FLAME_SPOTS = 9
FLAMES_SPREAD = 2.4       # the flames appear one after the other over this many seconds
FLAME_GROW = 0.5          # seconds for one flame to grow
FLAME_LAYERS = (          # (BGR colour, width scale, height scale): orange rim, yellow body, pale core
    ((20, 130, 240), 1.0, 1.0),
    ((60, 215, 255), 0.72, 0.84),
    ((170, 245, 255), 0.38, 0.5),
)
SMOKE_DARK = (58, 60, 64)     # BGR
SMOKE_MID = (80, 82, 86)
SMOKE_LIGHT = (104, 106, 110)
SMOKE_EDGE = (40, 41, 44)
FONTS = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


# --------------------------------------------------------------------------- sign

def hand_on(box: Box, hand: np.ndarray) -> bool:
    """At least 3 landmarks of the hand on the mug or right around it."""
    x0, y0, x1, y1 = box
    margin = 0.35 * max(x1 - x0, y1 - y0)
    inside = ((hand[:, 0] > x0 - margin) & (hand[:, 0] < x1 + margin)
              & (hand[:, 1] > y0 - margin) & (hand[:, 1] < y1 + margin))
    return int(inside.sum()) >= 3


def mug_checks(signs: SignInput) -> dict[str, bool]:
    cups = [obj for obj in signs.objects if obj.label == "cup"]
    if not cups:
        return {"mug in view": False}
    return {"mug in view": True,
            "hand on the mug": any(hand_on(cup.box, hand) for cup in cups for hand in signs.hands)}


# --------------------------------------------------------------------------- flames

@dataclass
class Flame:
    x: float
    base: float
    width: float
    height: float
    appear: float          # seconds after the start
    tongues: np.ndarray    # rows: x offset (in widths), height ratio, width ratio, phase, frequency


def place_flames(person: Box, size: tuple[int, int], seed: int) -> list[Flame]:
    """Spots spread across the width, appearing in random order. Those behind the person
    are taller, so that they rise above the shoulders."""
    rng = np.random.default_rng(seed)
    w, h = size
    n = FLAME_SPOTS
    xs = (np.linspace(0.03, 0.97, n) + rng.uniform(-0.04, 0.04, n)) * w
    behind = (xs > person[0]) & (xs < person[2])
    heights = np.where(behind, rng.uniform(0.5, 0.75, n), rng.uniform(0.35, 0.65, n)) * h
    widths = rng.uniform(0.11, 0.17, n) * w
    appear = rng.permutation(n) / n * FLAMES_SPREAD
    flames = []
    for x, height, width, start in zip(xs, heights, widths, appear):
        k = int(rng.integers(3, 6))
        ratios = rng.uniform(0.45, 1.0, k)
        ratios[0] = 1.0
        tongues = np.stack([rng.uniform(-0.35, 0.35, k), ratios, rng.uniform(0.35, 0.6, k),
                            rng.uniform(0, 2 * np.pi, k), rng.uniform(1.2, 2.2, k)], axis=1)
        flames.append(Flame(float(x), h * float(rng.uniform(1.0, 1.04)), float(width), float(height),
                            float(start), tongues))
    return flames


def tongue(cx: float, base: float, width: float, height: float, t: float, phase: float, freq: float,
           samples: int = 12) -> np.ndarray:
    """One wavy flame tongue: wide at the base, pointed at the tip, swaying more near the top."""
    s = np.linspace(0, 1, samples)
    half = 0.5 * width * (1 - s) ** 0.7 * (1 + 0.12 * np.sin(2 * np.pi * 2.3 * freq * t + phase + 6 * s))
    sway = 0.3 * width * s ** 1.6 * np.sin(2 * np.pi * freq * t + phase + 2.5 * s)
    y = base - height * s * (1 + 0.1 * np.sin(2 * np.pi * 1.7 * freq * t + 2 * phase))
    left = np.stack([cx + sway - half, y], axis=1)
    right = np.stack([cx + sway + half, y], axis=1)[::-1]
    return np.concatenate([left, right]).astype(np.int32)


def draw_flames(layer: np.ndarray, alpha: np.ndarray, flames: list[Flame], t: float, level: float) -> None:
    for color, width_scale, height_scale in FLAME_LAYERS:  # every rim, then every body, then every core
        for flame in flames:
            grow = smoothstep(flame.appear, flame.appear + FLAME_GROW, t) * level
            if grow <= 0.01:
                continue
            for dx, height_ratio, width_ratio, phase, freq in flame.tongues:
                points = tongue(flame.x + dx * flame.width, flame.base, flame.width * width_ratio * width_scale,
                                flame.height * height_ratio * height_scale * grow, t, phase, freq)
                cv2.fillPoly(layer, [points], color, cv2.LINE_AA)
                cv2.fillPoly(alpha, [points], 255, cv2.LINE_AA)


# --------------------------------------------------------------------------- smoke

SMOKE_ROWS = (  # (vertical position in band depths, puff size, colour, outlined, drift speed)
    (0.55, 1.0, SMOKE_DARK, False, 0.6),
    (0.72, 0.8, SMOKE_MID, True, 0.8),
    (0.86, 0.6, SMOKE_LIGHT, True, 1.0),
)


def draw_smoke_ceiling(layer: np.ndarray, alpha: np.ndarray, t: float, level: float, seed: int) -> None:
    """A band of grey smoke at the top: a dark mass with rows of irregular puffs along its
    lower edge (lighter and outlined in front, as in the cartoon) that drift and bob."""
    h, w = layer.shape[:2]
    depth = 0.17 * h * level
    if depth < 2:
        return
    rng = np.random.default_rng(seed + 1)  # same puffs on every frame
    for image, color in ((layer, SMOKE_DARK), (alpha, 255)):
        cv2.rectangle(image, (0, 0), (w, int(depth * 0.5)), color, -1)
    for row, (height, size, color, outlined, speed) in enumerate(SMOKE_ROWS):
        gaps = rng.uniform(0.035, 0.08, 40) * w          # irregular spacing
        xs = np.cumsum(gaps) - rng.uniform(0, 0.08) * w
        period = float(xs[-1])
        radii = rng.uniform(0.45, 1.0, xs.size) * 0.07 * w * size
        bob = rng.uniform(0, 2 * np.pi, xs.size)
        drift = t * 0.015 * w * speed * (1 if row % 2 == 0 else -1)
        for x, r, b in zip(xs, radii, bob):
            cx = (x + drift) % period - 0.05 * w
            if cx < -r or cx > w + r:
                continue
            cy = depth * (height + 0.05 * np.sin(1.2 * t + b))
            radius = int(r * level)
            if radius < 2:
                continue
            if outlined:
                cv2.circle(layer, (int(cx), int(cy)), radius + 3, SMOKE_EDGE, -1, cv2.LINE_AA)
            cv2.circle(layer, (int(cx), int(cy)), radius, color, -1, cv2.LINE_AA)
            cv2.circle(alpha, (int(cx), int(cy)), radius + 3, 255, -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- caption

@lru_cache(maxsize=4)
def caption_font(size: int):
    """A bold TrueType font through Pillow, or None (then OpenCV's built-in font is used)."""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in FONTS:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return None


@lru_cache(maxsize=32)
def caption_sprite(shown: str, full: str, size: int) -> np.ndarray:
    """BGRA image of the first letters of the caption, laid out as the full caption would
    be (so the text does not move while it is being typed): white with a black outline."""
    font = caption_font(size)
    stroke = max(size // 12, 2)
    if font is not None:
        from PIL import Image, ImageDraw

        x0, y0, x1, y1 = font.getbbox(full, stroke_width=stroke)
        image = Image.new("RGBA", (x1 - x0 + 2 * stroke, y1 - y0 + 2 * stroke), (0, 0, 0, 0))
        ImageDraw.Draw(image).text((stroke - x0, stroke - y0), shown, font=font, fill=(255, 255, 255, 255),
                                   stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        rgba = np.array(image)
        return np.dstack([rgba[:, :, 2::-1], rgba[:, :, 3]])
    scale = size / 30
    (tw, th), base = cv2.getTextSize(full, cv2.FONT_HERSHEY_DUPLEX, scale, 2 + stroke)
    sprite = np.zeros((th + base + 2 * stroke, tw + 2 * stroke, 4), np.uint8)
    for thickness, color in ((2 + 2 * stroke, (0, 0, 0, 255)), (2, (255, 255, 255, 255))):
        cv2.putText(sprite, shown, (stroke, th + stroke), cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)
    return sprite


def blend(base: np.ndarray, layer: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """uint8 alpha blend (alpha: float32 0..1, same height and width)."""
    return cv2.blendLinear(layer, base, alpha, 1.0 - alpha)


def paste(image: np.ndarray, sprite: np.ndarray, x: int, y: int, opacity: float) -> None:
    """Alpha-blends a BGRA sprite onto a uint8 image, in place, clipped to the image."""
    h, w = image.shape[:2]
    sh, sw = sprite.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + sw, w), min(y + sh, h)
    if x0 >= x1 or y0 >= y1:
        return
    part = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = part[:, :, 3].astype(np.float32) / 255 * opacity
    image[y0:y1, x0:x1] = blend(image[y0:y1, x0:x1], np.ascontiguousarray(part[:, :, :3]), a)


# --------------------------------------------------------------------------- overlay

class ThisIsFine(Overlay):
    name = "this_is_fine"
    title = "This is fine"
    short = "Fine"
    key = "f"
    hint = "hold your mug"
    fade_in_seconds = 1.0
    fade_out_seconds = 1.2
    linger_seconds = 1.5

    def __init__(self) -> None:
        self.flames: list[Flame] = []
        self.seed = 0

    def checks(self, signs: SignInput) -> dict[str, bool]:
        return mug_checks(signs)

    def start(self, scene: Scene) -> None:
        self.seed = scene.seed
        self.flames = place_flames(scene.person_box, scene.size, scene.seed)

    def render(self, image: np.ndarray, scene: Scene, elapsed: float, level: float) -> np.ndarray:
        if level <= 0.001:
            return image
        h, w = image.shape[:2]
        layer = np.zeros_like(image)
        alpha = np.zeros((h, w), np.uint8)
        draw_flames(layer, alpha, self.flames, elapsed, level)
        flames = alpha.astype(np.float32) * (1 / 255)
        flames *= 1 - scene.mask  # flames stay behind you
        out = blend(image, layer, flames)

        # the fire lights the room (and you) with a warm flicker
        warmth = 0.12 * level * (1 + 0.04 * np.sin(23 * elapsed) + 0.03 * np.sin(37 * elapsed + 1))
        out = cv2.transform(out, np.diag([1 - 0.6 * warmth, 1 + 0.1 * warmth, 1 + 0.5 * warmth]).astype(np.float32))

        band = int(0.2 * h)  # the smoke ceiling only covers the top of the frame
        smoke, smoke_alpha = np.zeros((band, w, 3), np.uint8), np.zeros((band, w), np.uint8)
        draw_smoke_ceiling(smoke, smoke_alpha, elapsed, level, self.seed)
        out[:band] = blend(out[:band], smoke, smoke_alpha.astype(np.float32) * (0.95 / 255))

        letters = int((elapsed - CAPTION_START) / CAPTION_CHAR) + 1
        if letters > 0:
            size = max(int(h * 0.07), 12)
            sprite = caption_sprite(CAPTION[:letters], CAPTION, size)
            paste(out, sprite, (w - sprite.shape[1]) // 2, int(h * 0.9) - sprite.shape[0], level)
        return out
