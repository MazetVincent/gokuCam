"""Small image helpers shared by the effects."""

from __future__ import annotations

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
