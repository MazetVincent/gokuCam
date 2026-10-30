"""Instant Transmission (Dragon Ball Z): two fingers on the forehead.

Departure: blue energy lines -> some vertical strips of the body tear -> the whole body
tears, gathers onto its axis and disappears. Arrival plays in chronological order (it is
not a rewind): blue lines -> the body unfolds from a thin column -> a few strips settle.
Only the person is torn, never the room.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..fx import remove_person, smoothstep
from ..hands import INDEX_TIP, MIDDLE_TIP, best_hand, dist, finger_extended, finger_folded, hand_scale
from .base import Jutsu, Scene, SignInput

STRAND_WIDTH = 4                                   # px; strands are 1 to 3 times this wide
ENERGY_LINES = 16
LINE_CORE = np.array([255, 245, 225], np.float32)  # BGR: bluish white core
LINE_GLOW = np.array([255, 110, 20], np.float32)   # BGR: electric blue glow


def strand_field(h: int, w: int, strength: float, fraction: float, rng: np.random.Generator,
                 hole_ratio: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Vertical tearing, as a cv2.remap y-map.

    A `fraction` of the strands (a few columns each) is stretched around a random anchor
    and shifted vertically, more so as `strength` grows: dark details become long sharp
    streaks. Among them, a `hole_ratio` share becomes a hole showing the room through the
    body. Purely vertical: every column stays in place. New random draw on each call.
    Returns map_y and, per column, 1 = keep / 0 = hole."""
    widths = rng.integers(1, 4, size=w // STRAND_WIDTH + 1) * STRAND_WIDTH
    strand_of_column = np.repeat(np.arange(widths.size), widths)[:w]
    n = int(strand_of_column[-1]) + 1

    def per_column(values: np.ndarray) -> np.ndarray:
        return values.astype(np.float32)[strand_of_column]

    torn_strands = rng.random(n) < fraction
    torn = per_column(torn_strands)
    anchor = per_column(rng.random(n)) * h
    # moderate stretch (the picture stays readable) with a few very long streaks
    amount = np.where(rng.random(n) < 0.25, rng.uniform(0.6, 0.85, n), rng.uniform(0.1, 0.5, n))
    stretch = 1.0 - strength * torn * per_column(amount)  # < 1: the column stretches
    shift = strength * torn * per_column(rng.normal(0.0, 0.05, n)) * h
    rows = np.arange(h, dtype=np.float32)[:, None]
    map_y = (anchor + (rows - anchor) * stretch + shift).astype(np.float32)
    keep = per_column(~(torn_strands & (rng.random(n) < hole_ratio)))
    return map_y, keep


def tear_into_strands(img: np.ndarray, s: float, rng: np.random.Generator) -> np.ndarray:
    """Tears the whole image: a few strands when s is small, everything at s = 1."""
    if s <= 0.01:
        return img
    h, w = img.shape[:2]
    map_y, _ = strand_field(h, w, s, 0.1 + 0.9 * s, rng)
    map_x = np.ascontiguousarray(np.broadcast_to(np.arange(w, dtype=np.float32), (h, w)))
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def energy_lines(h: int, w: int, span: tuple[float, float], dash: float, seed: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Vertical energy lines across `span`. Positions are fixed for the whole effect
    (seed); dash 0 -> 1 breaks them into dashes, then dots. Returns the thin core and the
    wide glow as float32 (h, w) maps."""
    lines = np.random.default_rng(seed)
    n = ENERGY_LINES
    xs = lines.uniform(*span, n)
    tops = np.where(lines.random(n) < 0.6, 0.0, lines.uniform(0.0, 0.5, n) * h)
    bottoms = np.where(lines.random(n) < 0.6, float(h), lines.uniform(0.5, 1.0, n) * h)
    widths = lines.integers(2, 5, n)
    phases = lines.random(n)
    period = 0.11 * h
    core = np.zeros((h, w), np.float32)
    for x, top, bottom, width, phase in zip(xs, tops, bottoms, widths, phases):
        level = float(rng.uniform(0.55, 1.0))  # flicker from frame to frame
        if dash <= 0.01:
            cv2.line(core, (int(x), int(top)), (int(x), int(bottom)), level, int(width))
            continue
        piece = period * (1.0 - 0.9 * dash)
        y = top - phase * period
        while y < bottom:
            y0, y1 = max(y, top), min(y + piece, bottom)
            if y1 > y0:
                cv2.line(core, (int(x), int(y0)), (int(x), int(y1)), level, int(width))
            y += period
    glow = cv2.resize(cv2.GaussianBlur(cv2.resize(core, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
                                       (0, 0), 4), (w, h), interpolation=cv2.INTER_LINEAR)
    return core, glow


def keyframes(t: float, arriving: bool) -> dict[str, float]:
    """Schedule of every phase for t in [0, 1] (see the module docstring)."""
    if not arriving:
        partial, full = smoothstep(0.3, 0.38, t), smoothstep(0.55, 0.7, t)
        return {
            "lines": smoothstep(0.0, 0.06, t) * (1.0 - smoothstep(0.32, 0.45, t)),
            "dash": smoothstep(0.2, 0.4, t),
            "settle": smoothstep(0.05, 0.35, t),  # room: live -> photo
            "fraction": 0.3 * partial + 0.7 * full,
            "tear": 0.7 * partial + 0.3 * full,
            "holes": 0.4 * (1.0 - full),
            "gather": smoothstep(0.7, 0.95, t),
            "presence": 1.0 - smoothstep(0.88, 1.0, t),
        }
    settled = smoothstep(0.55, 0.7, t)
    fraction = 1.0 - 0.7 * settled - 0.3 * smoothstep(0.82, 0.95, t)
    return {
        "lines": smoothstep(0.0, 0.06, t) * (1.0 - smoothstep(0.3, 0.42, t)),
        "dash": smoothstep(0.18, 0.38, t),
        "settle": 1.0 - smoothstep(0.6, 0.95, t),  # room: photo -> live
        "fraction": fraction,
        "tear": fraction,
        "holes": 0.4 * settled * (1.0 - smoothstep(0.82, 0.95, t)),
        "gather": 1.0 - smoothstep(0.3, 0.55, t),
        "presence": smoothstep(0.28, 0.4, t),
    }


def render_teleport(frame: np.ndarray, absent: np.ndarray, mask: np.ndarray, t: float, arriving: bool,
                    seed: int, rng: np.random.Generator) -> np.ndarray:
    """One frame of the effect. The person image and its mask go through the same
    deformation, so only the person tears. `absent` is what is shown without the person."""
    if t <= 0.0:
        return absent if arriving else frame
    if t >= 1.0:
        return frame if arriving else absent
    k = keyframes(t, arriving)
    h, w = mask.shape
    out = cv2.addWeighted(remove_person(frame, absent, mask), 1.0 - k["settle"], absent, k["settle"], 0)
    columns = np.nonzero(mask.max(axis=0) > 0.3)[0]
    x0, x1 = (float(columns[0]), float(columns[-1])) if columns.size else (0.35 * w, 0.65 * w)

    if k["presence"] > 0.01:
        cx = (x0 + x1) / 2
        squeeze = 1.0 - 0.98 * k["gather"]  # the body gathers onto its axis
        map_x = np.ascontiguousarray(np.broadcast_to(
            cx + (np.arange(w, dtype=np.float32) - cx) / squeeze, (h, w)), dtype=np.float32)
        map_y, keep = strand_field(h, w, k["tear"], k["fraction"], rng, k["holes"])
        person = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        alpha = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        alpha = (alpha * keep[None, :] * k["presence"]).astype(np.float32)
        if k["lines"] > 0.01:  # the person brightens a little under the energy
            person = cv2.addWeighted(person, 1.0 - 0.12 * k["lines"],
                                     np.full_like(person, (255, 225, 205)), 0.12 * k["lines"], 0)
        out = cv2.blendLinear(out, person, 1.0 - alpha, alpha)

    if k["lines"] > 0.01:
        # tint rather than add light: the lines stay blue even on a bright wall
        core, glow = energy_lines(h, w, (x0, x1), k["dash"], seed, rng)
        halo = np.clip(glow * 2.5 * k["lines"], 0.0, 0.9)[..., None]
        center = np.clip(core * k["lines"], 0.0, 1.0)[..., None]
        lit = out * (1.0 - halo) + LINE_GLOW * halo
        out = np.clip(lit * (1.0 - center) + LINE_CORE * center, 0, 255).astype(np.uint8)
    return out


def goku_hand_checks(hand: np.ndarray, signs: SignInput) -> dict[str, bool]:
    face = signs.face
    tips = (hand[INDEX_TIP] + hand[MIDDLE_TIP]) / 2
    return {
        "index extended": finger_extended(hand, "index"),
        "middle extended": finger_extended(hand, "middle"),
        "ring folded": finger_folded(hand, "ring"),
        "pinky folded": finger_folded(hand, "pinky"),
        "fingers together": dist(hand, INDEX_TIP, MIDDLE_TIP) < 0.5 * hand_scale(hand),
        "on the forehead": float(np.linalg.norm(tips - face.forehead)) < 0.55 * face.width,
    }


class InstantTransmission(Jutsu):
    name = "instant_transmission"
    title = "Instant Transmission (Goku)"
    short = "Goku"
    key = "t"
    hint = "index + middle together on the forehead"
    vanish_seconds = 1.0
    arrival_seconds = 1.0

    def checks(self, signs: SignInput) -> dict[str, bool]:
        if signs.face is None:
            return {"face in view": False}
        return best_hand(signs.hands, lambda hand: goku_hand_checks(hand, signs))

    def render(self, scene: Scene, t: float, arriving: bool) -> np.ndarray:
        # on arrival the "absent" image is whatever the call sees (it may contain a log)
        absent = scene.hidden if arriving else scene.plate
        return render_teleport(scene.frame, absent, scene.mask, t, arriving, scene.seed, scene.rng)
