"""Substitution jutsu (Kawarimi no jutsu, Naruto): clasped hands, both index fingers up.

The user disappears in a cloud of smoke; under the smoke an upright tree log takes their
place, bottom centre, where they were sitting (cut by the bottom of the frame). Everything
is generated in code: volumetric smoke and procedural bark, no external image.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..fx import fractal_noise, smoothstep
from ..hands import INDEX_TIP, RING_PIP, RING_TIP, WRIST, dist, finger_folded, hand_scale, index_up
from .base import Box, Jutsu, Scene, SignInput

VANISH_FADE = (0.3, 0.45)    # when each smoke puff starts to dissipate (departure)
ARRIVAL_FADE = (0.28, 0.4)   # same on arrival
SMOKE_BLOBS = 44
SMOKE_SCALE = 4              # smoke is computed at 1/4 resolution
SMOKE_LIGHT = np.array([246, 246, 245], np.float32)  # BGR: lit top
SMOKE_DARK = np.array([168, 171, 177], np.float32)   # BGR: shaded hollows

BARK = np.array([72, 96, 124], np.float32)    # BGR: bark brown
WOOD = np.array([140, 182, 216], np.float32)  # BGR: light wood of the cut


# --------------------------------------------------------------------------- sign

def single_index_up(hand: np.ndarray) -> bool:
    """What video tracking sees when the hands are clasped: one hand, index up."""
    return index_up(hand) and dist(hand, WRIST, RING_TIP) < 1.05 * dist(hand, WRIST, RING_PIP)


def fingers_interlocked(hand: np.ndarray) -> bool:
    return finger_folded(hand, "middle") and finger_folded(hand, "ring")


def seal_checks(hands: list[np.ndarray], face) -> dict[str, bool]:
    """Clasped hands with interlocked fingers, both index fingers extended and touching,
    pointing up in front of the chin. The rear hand is partly hidden, so interlocked
    fingers are only required on one hand."""
    if len(hands) < 2:
        return {"two hands": False}
    pairs = [(a, b) for i, a in enumerate(hands) for b in hands[i + 1:]]
    a, b = min(pairs, key=lambda pair: float(np.linalg.norm(pair[0][INDEX_TIP] - pair[1][INDEX_TIP])))
    scale = (hand_scale(a) + hand_scale(b)) / 2
    center = (a[INDEX_TIP] + b[INDEX_TIP]) / 2
    return {
        "two hands": True,
        "both index up": index_up(a) and index_up(b),
        "fingers interlocked": fingers_interlocked(a) or fingers_interlocked(b),
        "index tips touching": float(np.linalg.norm(a[INDEX_TIP] - b[INDEX_TIP])) < 0.6 * scale,
        "below the eyes": face is None or center[1] > face.forehead[1] + 0.3 * face.width,
    }


# --------------------------------------------------------------------------- log

def make_log_sprite(width: int, height: int, tint: np.ndarray | None = None, seed: int = 7) -> np.ndarray:
    """Upright log (BGRA): cut seen slightly from above (rings, cracks), bark with
    vertical plates and crevices, cylindrical shading (light from the left)."""
    rng = np.random.default_rng(seed)
    cap = max(int(width * 0.17), 4)  # half-height of the cut ellipse
    pad = 2
    h, w = height + cap + 2 * pad, width + 2 * pad
    cx, top = w / 2, cap + pad
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xs - cx) / (width / 2)      # -1..1 across the width
    v = (ys - top) / cap
    body = (np.abs(u) <= 1) & (ys >= top)
    cut = u ** 2 + v ** 2 <= 1

    # bark: vertical plates separated by crevices, fine grain, cylinder shading
    normal = np.sqrt(np.clip(1 - u ** 2, 0, 1))
    shade = 0.45 + 0.65 * np.clip(-0.45 * u + 0.85 * normal, 0, 1)
    plates = fractal_noise((h, w), rng, ((22, 0.5), (48, 0.3), (100, 0.2)), stretch=(10, 1))
    grain = fractal_noise((h, w), rng, ((60, 0.5), (140, 0.5)), stretch=(3, 1))
    texture = 0.5 + 0.75 * plates + 0.25 * (grain - 0.5)
    crevice = np.clip((plates - 0.18) / 0.2, 0, 1)
    texture = texture * (0.5 + 0.5 * crevice * crevice * (3 - 2 * crevice))
    color = BARK * (texture * shade)[..., None]

    # cut: irregular rings, darker heart, thin radial cracks, bark rim
    r = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)
    rings = 0.5 + 0.5 * np.sin((r + 0.05 * (grain - 0.5) + 0.03 * np.sin(angle * 3)) * 2 * np.pi * 12)
    wood = WOOD * (0.82 + 0.1 * rings - 0.12 * r - 0.15 * np.exp(-(r / 0.08) ** 2))[..., None]
    for a0 in rng.uniform(-np.pi, np.pi, 2):
        gap = np.abs((angle - a0 + np.pi) % (2 * np.pi) - np.pi)
        wood[(gap < 0.03 * (1 - r)) & (r < 0.7)] *= 0.6
    wood[r > 0.9] = BARK * 0.85
    color = np.where(cut[..., None], wood, color)

    if tint is not None:
        color = color * tint
    alpha = cv2.GaussianBlur((body | cut).astype(np.float32), (0, 0), 0.8)  # anti-aliased edges
    return np.dstack([np.clip(color, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])


def scene_tint(region: np.ndarray) -> np.ndarray:
    """Matches the log to the room light (brightness and dominant colour)."""
    if region.size == 0:
        return np.ones(3, np.float32)
    mean = region.reshape(-1, 3).mean(axis=0).astype(np.float32)
    brightness = float(np.clip(mean.mean() / 150, 0.55, 1.25))
    cast = mean / max(float(mean.mean()), 1.0)
    return (brightness * (0.75 + 0.25 * cast)).astype(np.float32)


def draw_sprite(image: np.ndarray, sprite: np.ndarray, center: tuple[float, float], angle: float,
                shadow: bool = False) -> np.ndarray:
    """Pastes a BGRA sprite rotated by `angle` degrees, centred on `center`. With `shadow`,
    a soft shadow behind the sprite blends it into the room."""
    h, w = image.shape[:2]
    sh, sw = sprite.shape[:2]
    matrix = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, 1.0)
    matrix[0, 2] += center[0] - sw / 2
    matrix[1, 2] += center[1] - sh / 2
    layer = cv2.warpAffine(sprite, matrix, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = layer[:, :, 3].astype(np.float32) / 255
    out = image.astype(np.float32)
    if shadow:
        small = cv2.resize(a, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (0, 0), max(sw / 40, 1.0))
        blur = cv2.warpAffine(blur, np.float32([[1, 0, sw / 60], [0, 1, 0]]), (w // 4, h // 4))
        out *= 1 - 0.35 * cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    a3 = a[..., None]
    return np.clip(out * (1 - a3) + layer[:, :, :3] * a3, 0, 255).astype(np.uint8)


@dataclass
class LogLayout:
    """Upright log in place of the person; it appears a little higher and settles."""
    sprite: np.ndarray
    rest: tuple[float, float]   # sprite centre once settled
    width: float

    def pose(self, settle: float) -> tuple[tuple[float, float], float]:
        """settle: 0 = just appeared, 1 = settled."""
        lift = 0.12 * self.width * (1 - settle) ** 2
        return (self.rest[0], self.rest[1] - lift), 3.0 * (1 - settle)

    def box(self) -> Box:
        x, y = self.rest
        sh, sw = self.sprite.shape[:2]
        return x - sw / 2, y - sh / 2, x + sw / 2, y + sh / 2


def log_layout(person: Box, size: tuple[int, int], plate: np.ndarray | None = None) -> LogLayout:
    """About half the shoulders wide, standing bottom centre of the person, top at chest
    height, bottom cut by the edge of the frame."""
    w, h = size
    x0, y0, x1, y1 = person
    width = int(np.clip(0.55 * (x1 - x0), 0.22 * w, 0.38 * w))
    top = float(np.clip(y0 + 0.42 * (y1 - y0), 0.38 * h, 0.62 * h))  # centre of the cut
    height = int(h - top + 0.1 * width)                               # overflows the frame
    cx = (x0 + x1) / 2
    tint = None
    if plate is not None:
        tint = scene_tint(plate[int(top):h, int(max(cx - width / 2, 0)):int(min(cx + width / 2, w))])
    sprite = make_log_sprite(width, height, tint)
    cap = max(int(width * 0.17), 4)
    return LogLayout(sprite, (cx, top - cap - 2 + sprite.shape[0] / 2), float(width))


def log_at_rest(plate: np.ndarray, layout: LogLayout) -> np.ndarray:
    center, angle = layout.pose(1.0)
    return draw_sprite(plate, layout.sprite, center, angle, shadow=True)


# --------------------------------------------------------------------------- smoke

@dataclass
class Smoke:
    x: np.ndarray
    y: np.ndarray
    r: np.ndarray
    start: np.ndarray
    fade: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    noise: np.ndarray            # turbulence, twice the grid height (it scrolls upwards)
    grid: tuple[int, int]        # computation grid (1/SMOKE_SCALE)


def make_smoke(box: Box, seed: int, fade_window: tuple[float, float], size: tuple[int, int]) -> Smoke:
    """Puffs spread over `box` (denser in the centre) that rise and spread out."""
    rng = np.random.default_rng(seed)
    w, h = size
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    n = SMOKE_BLOBS
    x = cx + rng.normal(0, 0.28, n) * bw
    y = cy + rng.normal(0, 0.25, n) * bh
    grid = (h // SMOKE_SCALE, w // SMOKE_SCALE)
    # "cauliflower" noise: rounded bumps rather than a uniform fog
    noise = 1 - np.abs(2 * fractal_noise((grid[0] * 2, grid[1]), rng, ((8, 0.45), (18, 0.33), (40, 0.22))) - 1)
    return Smoke(x=x, y=y, r=rng.uniform(0.11, 0.22, n) * min(bw, bh),
                 start=rng.uniform(0.0, 0.05, n), fade=rng.uniform(*fade_window, n),
                 vx=(x - cx) * rng.uniform(0.3, 0.7, n), vy=-rng.uniform(0.15, 0.35, n) * bh,
                 noise=noise, grid=grid)


def render_smoke(image: np.ndarray, smoke: Smoke, t: float) -> np.ndarray:
    """Volumetric smoke: soft density, rising turbulence, lit tops and shaded hollows.
    It bursts, covers the person, then rises and frays."""
    h, w = image.shape[:2]
    gh, gw = smoke.grid
    q = SMOKE_SCALE
    gx = (np.arange(gw, dtype=np.float32) + 0.5) * q
    gy = (np.arange(gh, dtype=np.float32) + 0.5) * q
    density = np.zeros((gh, gw), np.float32)
    for x, y, r, start, fade, vx, vy in zip(smoke.x, smoke.y, smoke.r, smoke.start, smoke.fade, smoke.vx, smoke.vy):
        grow = smoothstep(start, start + 0.12, t)
        amp = 1.6 * grow * (1 - smoothstep(fade, fade + 0.4, t))
        if amp <= 0.01:
            continue
        drift = max(t - 0.1, 0.0)
        px, py = x + vx * drift, y + vy * drift
        sigma = 0.55 * r * (0.35 + 0.65 * grow) * (1 + 0.9 * max(t - start, 0.0))
        cols = slice(max(int((px - 3 * sigma) / q), 0), min(int((px + 3 * sigma) / q) + 1, gw))
        rows = slice(max(int((py - 3 * sigma) / q), 0), min(int((py + 3 * sigma) / q) + 1, gh))
        if cols.start >= cols.stop or rows.start >= rows.stop:
            continue
        dx = (gx[cols] - px) ** 2
        dy = (gy[rows] - py) ** 2
        density[rows, cols] += amp * np.exp(-(dy[:, None] + dx[None, :]) / (2 * sigma * sigma))
    if float(density.max()) < 0.02:
        return image

    scroll = int(t * 0.6 * gh) % gh  # the turbulence rises
    turbulence = smoke.noise[scroll:scroll + gh]
    # thin areas (little turbulence) dissipate first: the smoke frays
    erosion = 1.2 * smoothstep(float(smoke.fade.min()), 0.95, t) * (1.3 - turbulence)
    density = np.clip(density * (0.25 + 1.5 * turbulence) - erosion, 0, None)
    alpha = 1 - np.exp(-3.0 * density)
    # lighting: the more smoke right above a point, the darker it is; bumps catch the light
    k = max(gh // 60, 2)
    above = np.zeros_like(density)
    above[k:] = cv2.GaussianBlur(density, (0, 0), 2)[:-k]
    light = np.clip(1.05 - 0.45 * np.tanh(0.7 * above) + 0.35 * (turbulence - 0.5), 0.0, 1.0)
    light = cv2.GaussianBlur(light, (0, 0), 0.8)
    color = SMOKE_DARK + (SMOKE_LIGHT - SMOKE_DARK) * light[..., None]
    a = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    c = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(image * (1 - a) + c * a, 0, 255).astype(np.uint8)


def render_substitution(frame: np.ndarray, plate: np.ndarray, layout: LogLayout, smoke: Smoke,
                        t: float) -> np.ndarray:
    """Departure: smoke covers the person, a log takes their place underneath.
    At t = 1 the frame is exactly log_at_rest(plate, layout)."""
    if t <= 0.0:
        return frame
    if t >= 1.0:
        return log_at_rest(plate, layout)
    swap = smoothstep(0.08, 0.2, t)  # under the smoke: live -> empty-room photo
    out = cv2.addWeighted(frame, 1 - swap, plate, swap, 0)
    if t > 0.15:
        center, angle = layout.pose(smoothstep(0.15, 0.45, t))
        out = draw_sprite(out, layout.sprite, center, angle, shadow=True)
    return render_smoke(out, smoke, t)


def render_smoke_arrival(frame: np.ndarray, hidden: np.ndarray, smoke: Smoke, t: float) -> np.ndarray:
    """Arrival: smoke covers the log; underneath, the hidden image gives way to live video."""
    if t <= 0.0:
        return hidden
    if t >= 1.0:
        return frame
    swap = smoothstep(0.1, 0.25, t)
    return render_smoke(cv2.addWeighted(hidden, 1 - swap, frame, swap, 0), smoke, t)


# --------------------------------------------------------------------------- jutsu

class Substitution(Jutsu):
    name = "substitution"
    title = "Substitution (Naruto)"
    short = "Naruto"
    key = "n"
    hint = "clasped hands, both index fingers up"
    vanish_seconds = 1.8
    arrival_seconds = 1.1

    def __init__(self) -> None:
        self.log: LogLayout | None = None
        self.smoke: Smoke | None = None

    def checks(self, signs: SignInput) -> dict[str, bool]:
        hands = signs.hands
        # Video tracking merges two clasped hands into one hand with its index up. Only
        # then do we pay for a detection on the still image, which finds both hands (a
        # simple "shush" gesture shows a single hand and stays rejected).
        if len(hands) < 2 and any(single_index_up(hand) for hand in hands):
            hands = signs.still_hands
        return seal_checks(hands, signs.face)

    def start(self, scene: Scene, arriving: bool) -> None:
        box = scene.person_box
        if arriving and scene.leftover is not None:  # the smoke also covers the log
            lx0, ly0, lx1, ly1 = scene.leftover
            box = (min(box[0], lx0), min(box[1], ly0), max(box[2], lx1), max(box[3], ly1))
        if not arriving:
            self.log = log_layout(box, scene.size, scene.plate)
        self.smoke = make_smoke(box, scene.seed, ARRIVAL_FADE if arriving else VANISH_FADE, scene.size)

    def render(self, scene: Scene, t: float, arriving: bool) -> np.ndarray:
        if arriving:
            return render_smoke_arrival(scene.frame, scene.hidden, self.smoke, t)
        return render_substitution(scene.frame, scene.plate, self.log, self.smoke, t)

    def hidden_image(self, scene: Scene) -> np.ndarray:
        return log_at_rest(scene.plate, self.log)

    def leftover_box(self):
        return self.log.box() if self.log is not None else None
