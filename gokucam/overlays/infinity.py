"""Infinity: press your two fists together, and a blazing infinity sign appears between
them, as in the One Punch Man scene.

Everything but you turns dark grey, the two loops of the sign trace themselves out from
the crossing, then burn white-hot with an orange glow while sparks shoot out of them all
over the screen. It lasts as long as your fists stay together, and the sign follows them.

The background, the loops and the glow depend only on the fade level (so the fade out is
the fade in backwards); the sparks are a stateless particle system driven by the time:
each spark is a function of its birth time, so no per-frame state has to be kept.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..fx import blend, smoothstep, tight_mask
from ..hands import INDEX_MCP, MIDDLE_MCP, PINKY_MCP, RING_MCP, WRIST, finger_folded, hand_scale
from ..jutsus.base import Scene, SignInput
from ..models import DETECT_SIZE
from .base import Overlay

CURVE_POINTS = 240
LOBE_HEIGHT = 1.35          # rounder loops than a plain lemniscate of Bernoulli
SIZE_PER_HAND = 2.6         # half-width of the sign, in hand scales
SIZE_RANGE = (0.10, 0.24)   # ... clamped to this fraction of the frame width
DEFAULT_SIZE = 0.17
FOLLOW = 0.35               # how fast the sign follows the fists (per frame)
CORE = (235, 245, 255)      # BGR: white-hot line
GLOW = np.array([0.30, 0.55, 1.0], np.float32)  # BGR gains: orange-red halo
SPARK_RATE = 260.0          # sparks per second at full level
SPARK_POOL = 4096
SPARK_COLORS = np.array([   # BGR along a spark's life: white-hot, yellow, orange, deep red
    (240, 248, 255), (110, 210, 255), (40, 130, 255), (30, 40, 190)], np.float32)
SPARK_BLUE = np.array((255, 200, 150), np.float32)
BACKGROUND_GAIN = 0.32      # how dark the grey background gets


# --------------------------------------------------------------------------- sign

def palm_center(hand: np.ndarray) -> np.ndarray:
    return hand[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0)


def is_fist(hand: np.ndarray) -> bool:
    """Index folded, and at least two of the three other fingers (one is often half hidden
    by the other fist). The index is required: the Substitution's clasped hands have
    their other fingers folded too, but their index fingers up."""
    return finger_folded(hand, "index") and sum(finger_folded(hand, f) for f in ("middle", "ring", "pinky")) >= 2


def fists_together(a: np.ndarray, b: np.ndarray) -> bool:
    scale = (hand_scale(a) + hand_scale(b)) / 2
    return float(np.linalg.norm(palm_center(a) - palm_center(b))) < 2.2 * scale


def joined_fists_checks(hands: list[np.ndarray]) -> dict[str, bool]:
    if len(hands) < 2:
        return {"two hands": False}
    fists = [hand for hand in hands if is_fist(hand)]
    pairs = [(a, b) for i, a in enumerate(fists) for b in fists[i + 1:]]
    return {
        "two hands": True,
        "two fists": len(fists) >= 2,
        "fists together": any(fists_together(a, b) for a, b in pairs),
    }


# --------------------------------------------------------------------------- drawing

def lemniscate(n: int = CURVE_POINTS) -> np.ndarray:
    """Unit infinity sign, (n, 2), centred on 0, x in [-1, 1]. t = 0 is the right tip,
    pi/2 and 3pi/2 the crossing, pi the left tip."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    d = 1 + np.sin(t) ** 2
    return np.stack([np.cos(t) / d, LOBE_HEIGHT * np.sin(t) * np.cos(t) / d], axis=1).astype(np.float32)


def revealed_lobes(reveal: float, n: int = CURVE_POINTS // 2) -> list[np.ndarray]:
    """Both loops traced from the crossing outwards, `reveal` (0..1) of the way round."""
    lobes = []
    for sign in (-1, 1):                                  # right loop goes down from pi/2, left goes up
        t = np.pi / 2 + sign * np.linspace(0, reveal * np.pi, max(int(n * reveal), 2))
        d = 1 + np.sin(t) ** 2
        lobes.append(np.stack([np.cos(t) / d, LOBE_HEIGHT * np.sin(t) * np.cos(t) / d], axis=1).astype(np.float32))
    return lobes


class Sparks:
    """Random parameters of a pool of sparks, drawn once: spark i is born at i / SPARK_RATE
    seconds and reuses the parameters of slot i % SPARK_POOL."""

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = SPARK_POOL
        self.t = rng.uniform(0, 2 * np.pi, n).astype(np.float32)          # where on the curve
        self.angle = rng.normal(0, 0.55, n).astype(np.float32)             # deviation from straight out
        self.speed = rng.uniform(0.35, 1.5, n).astype(np.float32)          # frame widths per second
        self.life = rng.uniform(0.45, 1.3, n).astype(np.float32)
        self.drag = rng.uniform(0.6, 2.2, n).astype(np.float32)
        self.keep = rng.uniform(0, 1, n).astype(np.float32)                # fewer sparks at low level
        self.blue = rng.uniform(0, 1, n) < 0.12
        self.width = rng.choice([1, 1, 2], n)

    def draw(self, layer: np.ndarray, center: np.ndarray, size: float, elapsed: float,
             level: float, reveal: float, scale: float) -> None:
        """Streaks on `layer` (BGR uint8, `scale` = layer width / frame width)."""
        if level <= 0.01:
            return
        h, w = layer.shape[:2]
        first = int(np.floor(max(elapsed - 1.3, 0) * SPARK_RATE))
        last = int(np.floor(elapsed * SPARK_RATE))
        if last < first:
            return
        index = np.arange(first, last + 1)
        slot = index % SPARK_POOL
        age = elapsed - index / SPARK_RATE
        life = self.life[slot]
        alive = (age >= 0) & (age < life) & (self.keep[slot] < level)
        # only from the part of the loops already traced
        along = np.abs(((self.t[slot] - np.pi / 2 + np.pi) % (2 * np.pi)) - np.pi)
        alive &= along <= reveal * np.pi
        if not alive.any():
            return
        slot, age, life = slot[alive], age[alive], life[alive]
        t = self.t[slot]
        d = 1 + np.sin(t) ** 2
        start = np.stack([np.cos(t) / d, LOBE_HEIGHT * np.sin(t) * np.cos(t) / d], axis=1)
        out = start + np.float32(0.0001)
        out[:, 1] *= 1.6                                   # spread vertically too, not just sideways
        heading = np.arctan2(out[:, 1], out[:, 0]) + self.angle[slot]
        direction = np.stack([np.cos(heading), np.sin(heading)], axis=1)
        speed = self.speed[slot][:, None] * w / scale       # pixels (frame) per second
        drag = self.drag[slot][:, None]

        def travel(a: np.ndarray) -> np.ndarray:
            return (1 - np.exp(-drag * a[:, None])) / drag

        origin = center + start * size
        head = (origin + direction * speed * travel(age)) * scale
        tail = (origin + direction * speed * travel(np.maximum(age - 0.05, 0))) * scale
        fade = age / life
        stage = np.clip(fade * (len(SPARK_COLORS) - 1), 0, len(SPARK_COLORS) - 1.001)
        low = stage.astype(int)
        frac = (stage - low)[:, None]
        color = SPARK_COLORS[low] * (1 - frac) + SPARK_COLORS[low + 1] * frac
        color[self.blue[slot]] = SPARK_BLUE
        color *= ((1 - fade ** 2) * min(level * 1.5, 1.0))[:, None]
        for (x0, y0), (x1, y1), c, width in zip(tail.astype(int), head.astype(int), color, self.width[slot]):
            if -20 < x1 < w + 20 and -20 < y1 < h + 20:
                cv2.line(layer, (int(x0), int(y0)), (int(x1), int(y1)), c.tolist(), int(width), cv2.LINE_AA)


def dark_background(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=BACKGROUND_GAIN, beta=0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- overlay

class Infinity(Overlay):
    name = "infinity"
    title = "Infinity (One Punch Man)"
    short = "Infinity"
    key = "i"
    hint = "press your two fists together"
    fade_in_seconds = 1.4
    fade_out_seconds = 0.8
    linger_seconds = 0.4

    def __init__(self) -> None:
        self.target: tuple[float, float, float] | None = None   # fists: x, y (0..1), hand scale / width
        self.position: np.ndarray | None = None                  # smoothed (x, y, half-width) in pixels
        self.sparks = Sparks(0)

    def checks(self, signs: SignInput) -> dict[str, bool]:
        hands = signs.hands
        # video tracking may merge two fists pressed together into one hand: then look
        # at the still image, which finds both
        if len(hands) < 2 and any(is_fist(hand) for hand in hands):
            hands = signs.still_hands
        result = joined_fists_checks(hands)
        if all(result.values()):
            fists = [hand for hand in hands if is_fist(hand)]
            a, b = next((a, b) for i, a in enumerate(fists) for b in fists[i + 1:] if fists_together(a, b))
            mid = (palm_center(a) + palm_center(b)) / 2
            scale = (hand_scale(a) + hand_scale(b)) / 2
            self.target = (float(mid[0] / DETECT_SIZE[0]), float(mid[1] / DETECT_SIZE[1]),
                           float(scale / DETECT_SIZE[0]))
        return result

    def start(self, scene: Scene) -> None:
        self.sparks = Sparks(scene.seed)
        self.position = None

    def place(self, scene: Scene) -> np.ndarray:
        w, h = scene.size
        if self.target is not None:
            x, y, scale = self.target
            size = float(np.clip(SIZE_PER_HAND * scale, *SIZE_RANGE)) * w
            goal = np.array([x * w, y * h, size], np.float32)
        else:                                                     # toggled by hand: in front of the chest
            x0, y0, x1, y1 = scene.person_box
            goal = np.array([(x0 + x1) / 2, y0 + 0.75 * (h - y0), DEFAULT_SIZE * w], np.float32)
        if self.position is None:
            self.position = goal
        else:
            self.position = self.position + (goal - self.position) * FOLLOW
        return self.position

    def render(self, image: np.ndarray, scene: Scene, elapsed: float, level: float) -> np.ndarray:
        if level <= 0.001:
            return image
        h, w = image.shape[:2]
        x, y, size = self.place(scene)
        center = np.array([x, y], np.float32)
        dark = smoothstep(0.0, 0.45, level)
        reveal = smoothstep(0.15, 0.75, level)
        burn = smoothstep(0.3, 1.0, level)

        behind = (1.0 - tight_mask(scene.mask)) * dark
        out = blend(image, dark_background(image), behind)

        # the light: drawn at half resolution, glow at a quarter, added onto the image
        sw, sh = w // 2, h // 2
        core = np.zeros((sh, sw, 3), np.uint8)
        holes = np.zeros((h // 4, w // 4), np.uint8)
        thickness = max(int(size * 0.045 * (0.6 + 0.4 * burn)), 2)
        for lobe in revealed_lobes(reveal):
            points = ((center + lobe * size) * 0.5).astype(np.int32)
            cv2.polylines(core, [points], False, CORE, thickness, cv2.LINE_AA)
            cv2.circle(core, tuple(points[-1]), thickness + 2, CORE, -1, cv2.LINE_AA)   # tracing head
        if reveal > 0.98:                                         # closed loops: dark eyes inside
            full = ((center + lemniscate() * size) * 0.25).astype(np.int32)
            cv2.fillPoly(holes, [full], 255, cv2.LINE_AA)
        # sparkles running round the loops
        n = 14
        t = (np.arange(n) / n * 2 * np.pi + elapsed * 2.2) % (2 * np.pi)
        d = 1 + np.sin(t) ** 2
        run = np.stack([np.cos(t) / d, LOBE_HEIGHT * np.sin(t) * np.cos(t) / d], axis=1)
        along = np.abs(((t - np.pi / 2 + np.pi) % (2 * np.pi)) - np.pi)
        for (px, py), visible in zip(((center + run * size) * 0.5).astype(int), along <= reveal * np.pi):
            if visible:
                cv2.circle(core, (int(px), int(py)), max(thickness // 2 + 1, 2), CORE, -1, cv2.LINE_AA)

        small = cv2.resize(core, (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
        heat = small.max(axis=2)
        glow = (cv2.GaussianBlur(heat, (0, 0), size / 4 * 0.10) * 2.2
                + cv2.GaussianBlur(heat, (0, 0), size / 4 * 0.35) * 1.4)
        flicker = 0.9 + 0.1 * np.sin(elapsed * 23.0) * np.sin(elapsed * 7.0)
        glow = glow[..., None] * GLOW * (burn * flicker)

        if holes.any():                                           # dark holes inside the loops
            eyes = cv2.GaussianBlur(holes.astype(np.float32) / 255, (0, 0), size / 4 * 0.06)
            eyes = cv2.resize(eyes, (w, h), interpolation=cv2.INTER_LINEAR) * (0.75 * burn)
            out = blend(out, np.zeros_like(out), eyes)

        light = cv2.resize(np.clip(glow, 0, 255).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_LINEAR)
        light = cv2.add(light, cv2.convertScaleAbs(core, alpha=0.4 + 0.6 * reveal))
        self.sparks.draw(light, center, size, elapsed, burn, reveal, 0.5)
        light = cv2.resize(light, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.add(out, light)
