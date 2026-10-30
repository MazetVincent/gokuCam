"""Into the bush (The Simpsons): raise one finger in the air, and back away into a hedge.

Departure: a cartoon hedge grows from the bottom of the frame, around you and above your
head; then its leafy edge closes in on you from the outline of your body towards its
thickest parts, your face last, as Homer peeking out of the bushes. Arrival is the same
animation played backwards. While you are away, the call sees the room behind the hedge.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..fx import fractal_noise, smoothstep
from ..hands import INDEX_TIP, best_hand, finger_folded, index_up
from .base import Box, Jutsu, Scene, SignInput

HEDGE = (32, 139, 58)          # BGR, measured on the reference cartoon
HEDGE_DARK = (24, 118, 46)     # a few slightly darker patches
LEAF_MARKS = (0, 77, 23)       # little dark leaf strokes
OUTLINE = (10, 60, 20)
GROW_END = 0.35                # the hedge has grown at this point of the departure
SCALE = 2                      # masks are computed at half resolution
BODY_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))


# --------------------------------------------------------------------------- sign

def raised_finger_checks(hand: np.ndarray, face) -> dict[str, bool]:
    """Index up, the other fingers folded, and the hand raised above the forehead
    (a finger at the mouth is a "shush", a finger at the chin is the Substitution)."""
    above = (hand[INDEX_TIP][1] < face.forehead[1]) if face is not None else hand[INDEX_TIP][1] < 0.4 * 360
    return {
        "index up": index_up(hand),
        "middle folded": finger_folded(hand, "middle"),
        "ring folded": finger_folded(hand, "ring"),
        "pinky folded": finger_folded(hand, "pinky"),
        "hand above the forehead": bool(above),
    }


# --------------------------------------------------------------------------- hedge

def hedge_texture(size: tuple[int, int], seed: int) -> np.ndarray:
    """Flat cartoon green with darker patches and little leaf strokes, as in the cartoon."""
    w, h = size
    rng = np.random.default_rng(seed)
    texture = np.empty((h, w, 3), np.uint8)
    texture[:] = HEDGE
    patches = fractal_noise((h // 4, w // 4), rng, ((8, 0.6), (20, 0.4)))
    patches = cv2.resize((patches > 0.86).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    texture[patches > 0] = HEDGE_DARK
    for _ in range(int(w * h / 5000)):  # leaf strokes: small "w" and "~" marks
        x, y = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        r = int(rng.uniform(0.006, 0.012) * w)
        start = rng.uniform(0, 360)
        cv2.ellipse(texture, (x, y), (r, r // 2 + 1), rng.uniform(-30, 30), start, start + 160, LEAF_MARKS, 2, cv2.LINE_AA)
        cv2.ellipse(texture, (x + r, y), (r, r // 2 + 1), rng.uniform(-30, 30), start, start + 160, LEAF_MARKS, 2, cv2.LINE_AA)
    return texture


@dataclass
class HedgeShape:
    top: np.ndarray         # (w/SCALE,) y of the hedge top for every column, half resolution
    leaves: np.ndarray      # (h/SCALE, w/SCALE) noise in [-0.5, 0.5]: jagged, leafy edges
    texture: np.ndarray     # full-resolution hedge picture


def hedge_shape(person: Box, size: tuple[int, int], seed: int) -> HedgeShape:
    """The hedge covers the whole width, from above the person's head to the bottom, with
    a top edge made of round bumps."""
    w, h = size
    gw, gh = w // SCALE, h // SCALE
    rng = np.random.default_rng(seed)
    base = max(person[1] / SCALE - 0.14 * gh, 0.05 * gh)
    xs = np.arange(gw, dtype=np.float32)
    top = np.full(gw, base + 0.1 * gh, np.float32)
    x = -0.05 * gw
    while x < 1.05 * gw:  # round bumps along the top
        r = rng.uniform(0.06, 0.12) * gw
        inside = np.abs(xs - x) < r
        bump = base + 0.1 * gh - np.sqrt(np.maximum(r * r - (xs - x) ** 2, 0)) * rng.uniform(0.7, 1.0)
        top[inside] = np.minimum(top[inside], bump[inside])
        x += rng.uniform(0.7, 1.2) * r
    leaves = fractal_noise((gh, gw), rng, ((40, 0.5), (90, 0.5))) - 0.5
    return HedgeShape(np.maximum(top, 0), leaves, hedge_texture(size, seed))


def face_spot(body: np.ndarray) -> np.ndarray:
    """0..1 bump around where the eyes usually are on a webcam silhouette (a fifth of the
    way down from the top of the head), so that the face peeks out of the hedge last."""
    ys, xs = np.nonzero(body)
    top, height = ys.min(), ys.max() - ys.min() + 1
    eyes_y = top + 0.2 * height
    band = np.abs(ys - eyes_y) < 0.08 * height
    eyes_x = xs[band].mean() if band.any() else xs.mean()
    rows = np.arange(body.shape[0], dtype=np.float32)[:, None]
    cols = np.arange(body.shape[1], dtype=np.float32)[None, :]
    return np.exp(-((rows - eyes_y) / (0.12 * height)) ** 2 - ((cols - eyes_x) / (0.1 * body.shape[1])) ** 2)


def hedge_alpha(shape: HedgeShape, person_small: np.ndarray, grow: float, sink: float) -> np.ndarray:
    """Half-resolution coverage of the hedge (0..1).
    grow 0 -> 1: the hedge rises from the bottom to its full height.
    sink 0 -> 1: it closes in on the person, from the outline of the body to its
    thickest parts; the face gets a head start so that it disappears last."""
    gh, gw = shape.leaves.shape
    rows = np.arange(gh, dtype=np.float32)[:, None]
    top_now = gh - (gh - shape.top[None, :]) * grow
    region = rows > top_now + 6 * shape.leaves                     # leafy top edge
    # the app's person mask is generous (it pads the outline); the hedge hugs the real body
    body = cv2.erode((person_small > 0.9).astype(np.uint8), BODY_KERNEL) > 0
    if sink <= 0.0 or not body.any():
        visible = body
    else:
        depth = cv2.distanceTransform(body.astype(np.uint8), cv2.DIST_L2, 5)
        depth /= max(float(depth.max()), 1.0)
        score = 0.7 * depth + 0.6 * face_spot(body) + 0.25 * shape.leaves
        # by rank: the visible area shrinks steadily, the face (highest score) goes last
        threshold = np.quantile(score[body], min(sink * 1.05, 1.0))
        visible = body & (score > threshold)
    alpha = (region & ~visible).astype(np.float32)
    return cv2.GaussianBlur(alpha, (0, 0), 0.7)


def render_bush(frame: np.ndarray, absent: np.ndarray, mask: np.ndarray, shape: HedgeShape, t: float) -> np.ndarray:
    """Departure frame, t from 0 to 1 (play it with 1 - t for the arrival). At t = 1 the
    frame is exactly hedge_over(absent, shape)."""
    if t <= 0.0:
        return frame
    if t >= 1.0:
        return hedge_over(absent, shape)
    h, w = frame.shape[:2]
    grow = smoothstep(0.0, GROW_END, t)
    sink = smoothstep(GROW_END, 1.0, t)
    person = cv2.resize(mask, (w // SCALE, h // SCALE), interpolation=cv2.INTER_AREA)
    alpha = hedge_alpha(shape, person, grow, sink)
    # the room above the hedge switches to the photo only once the person is fully hidden
    settle = smoothstep(0.92, 1.0, t)
    base = cv2.addWeighted(frame, 1 - settle, absent, settle, 0) if settle > 0 else frame
    return composite(base, shape, alpha)


def hedge_over(image: np.ndarray, shape: HedgeShape) -> np.ndarray:
    h, w = image.shape[:2]
    return composite(image, shape, hedge_alpha(shape, np.zeros((h // SCALE, w // SCALE), np.float32), 1.0, 1.0))


def composite(image: np.ndarray, shape: HedgeShape, alpha_small: np.ndarray) -> np.ndarray:
    """Hedge picture with a dark outline along every edge, blended over `image`."""
    h, w = image.shape[:2]
    edge = alpha_small - cv2.erode(alpha_small, np.ones((3, 3), np.uint8))
    alpha = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
    outline = cv2.resize(np.clip(edge * 1.5, 0, 1), (w, h), interpolation=cv2.INTER_LINEAR)
    hedge = cv2.blendLinear(np.full_like(image, OUTLINE), shape.texture, outline, 1.0 - outline)
    return cv2.blendLinear(hedge, image, alpha, 1.0 - alpha)


# --------------------------------------------------------------------------- jutsu

class IntoTheBush(Jutsu):
    name = "into_the_bush"
    title = "Into the bush (Homer)"
    short = "Homer"
    key = "h"
    hint = "one finger raised above your head"
    vanish_seconds = 2.4
    arrival_seconds = 2.0

    def __init__(self) -> None:
        self.shape: HedgeShape | None = None
        self.hidden: np.ndarray | None = None   # the frozen image our last departure produced
        self.from_our_hedge = False

    def checks(self, signs: SignInput) -> dict[str, bool]:
        return best_hand(signs.hands, lambda hand: raised_finger_checks(hand, signs.face))

    def start(self, scene: Scene, arriving: bool) -> None:
        # coming back from our own hedge: reuse it, so the reverse starts from what the call sees
        self.from_our_hedge = arriving and self.shape is not None and scene.hidden is self.hidden
        if not self.from_our_hedge:
            self.shape = hedge_shape(scene.person_box, scene.size, scene.seed)

    def render(self, scene: Scene, t: float, arriving: bool) -> np.ndarray:
        if not arriving:
            return render_bush(scene.frame, scene.plate, scene.mask, self.shape, t)
        if self.from_our_hedge:  # the departure played backwards
            return render_bush(scene.frame, scene.plate, scene.mask, self.shape, 1.0 - t)
        # after another jutsu's departure (e.g. a log): the hedge fades in over it first
        out = render_bush(scene.frame, scene.hidden, scene.mask, self.shape, 1.0 - t)
        fade = smoothstep(0.0, 0.15, t)
        return cv2.addWeighted(scene.hidden, 1 - fade, out, fade, 0) if fade < 1 else out

    def hidden_image(self, scene: Scene) -> np.ndarray:
        self.hidden = hedge_over(scene.plate, self.shape)
        return self.hidden

    def leftover_box(self) -> Box | None:
        if self.shape is None:
            return None
        h, w = self.shape.texture.shape[:2]
        return 0.0, float(self.shape.top.min() * SCALE), float(w), float(h)
