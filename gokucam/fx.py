"""Small image helpers shared by the effects."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """0 before edge0, 1 after edge1, smooth in between. Effects use it to schedule phases."""
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def remove_person(frame: np.ndarray, plate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """uint8 image where the masked area is replaced by the empty-room photo."""
    return cv2.blendLinear(frame, plate, 1.0 - mask, mask)


def pixelate(image: np.ndarray, block: int) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(w // block, 1), max(h // block, 1)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def fractal_noise(shape: tuple[int, int], rng: np.random.Generator,
                  octaves: tuple[tuple[int, float], ...], stretch: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Fractal noise in [0, 1]: random grids of `cells` cells upscaled and summed with
    their weights. `stretch` (vertical, horizontal) elongates the patterns."""
    h, w = shape
    out = np.zeros(shape, np.float32)
    for cells, weight in octaves:
        gh = max(int(cells * h / max(h, w) / stretch[0]) + 2, 2)
        gw = max(int(cells * w / max(h, w) / stretch[1]) + 2, 2)
        out += weight * cv2.resize(rng.random((gh, gw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    out -= out.min()
    return out / max(float(out.max()), 1e-6)


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


@lru_cache(maxsize=16)
def truetype_font(candidates: tuple[str, ...], size: int):
    """The first available TrueType font through Pillow, or None (OpenCV's font is then used)."""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return None


@lru_cache(maxsize=64)
def text_sprite(shown: str, full: str, size: int, fonts: tuple[str, ...], spacing: float = 0.0) -> np.ndarray:
    """BGRA image of `shown` (a prefix of `full`, for text typed letter by letter), laid out
    as `full` would be so that it does not move: white with a black outline. `spacing`
    adds letter spacing, as a fraction of the size."""
    font = truetype_font(fonts, size)
    stroke = max(size // 12, 2)
    if font is None:
        scale = size / 30
        (tw, th), base = cv2.getTextSize(full, cv2.FONT_HERSHEY_DUPLEX, scale, 2 + stroke)
        sprite = np.zeros((th + base + 2 * stroke, tw + 2 * stroke, 4), np.uint8)
        for thickness, color in ((2 + 2 * stroke, (0, 0, 0, 255)), (2, (255, 255, 255, 255))):
            cv2.putText(sprite, shown, (stroke, th + stroke), cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness,
                        cv2.LINE_AA)
        return sprite

    from PIL import Image, ImageDraw

    gap = spacing * size
    _, y0, _, y1 = font.getbbox(full, stroke_width=stroke)
    width = sum(font.getlength(c) for c in full) + gap * (len(full) - 1) + 2 * stroke
    image = Image.new("RGBA", (int(width) + 2, y1 - y0 + 2 * stroke), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x = stroke
    for c in shown:  # letter by letter, for the spacing
        draw.text((x, stroke - y0), c, font=font, fill=(255, 255, 255, 255),
                  stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        x += font.getlength(c) + gap
    rgba = np.array(image)
    return np.dstack([rgba[:, :, 2::-1], rgba[:, :, 3]])


TIGHT_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))


def tight_mask(mask: np.ndarray) -> np.ndarray:
    """The app's person mask is padded (so that effects never miss a bit of you); this
    brings it back to the real outline, for effects drawn right behind the person."""
    h, w = mask.shape
    small = cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    body = cv2.erode((small > 0.9).astype(np.uint8), TIGHT_KERNEL).astype(np.float32)
    return cv2.resize(cv2.GaussianBlur(body, (0, 0), 1.0), (w, h), interpolation=cv2.INTER_LINEAR)
