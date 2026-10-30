"""The contract every jutsu (sign + vanishing effect) implements.

A jutsu answers two questions:

1. Is the user making my sign right now?  -> `checks()` returns named criteria; the sign
   is made when they are all True. The names are shown live by `--debug`.
2. What does frame t of my effect look like? -> `render()` draws the departure
   (vanishing) or the arrival (coming back) for t going from 0 to 1.

The application takes care of everything else: holding the sign for a moment, playing
the effect over `vanish_seconds` / `arrival_seconds`, and sending the frozen
`hidden_image()` to the call while the user is invisible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import numpy as np

from ..hands import Face

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


@dataclass
class SignInput:
    """What the detectors saw on the current frame (detection-image coordinates)."""
    hands: list[np.ndarray]                # landmarks of each tracked hand, (21, 2)
    face: Face | None                      # None when no face was seen recently
    detect_still: Callable[[], list[np.ndarray]] = field(default=lambda: [], repr=False)

    @cached_property
    def still_hands(self) -> list[np.ndarray]:
        """Hands re-detected on the still image, without video tracking. Slower: only ask
        for it when needed (video tracking merges two clasped hands into one)."""
        return self.detect_still()


@dataclass
class Scene:
    """Everything an effect needs to draw one frame (full-resolution pixels)."""
    frame: np.ndarray            # live webcam frame
    plate: np.ndarray            # photo of the empty room
    hidden: np.ndarray           # what the call currently sees while the user is invisible
    mask: np.ndarray             # person mask, float32 0..1
    seed: int                    # fixed for the whole effect: keeps random layouts stable
    rng: np.random.Generator     # changes every frame: flicker, jitter
    leftover: Box | None = None  # what the previous departure left in `hidden` (e.g. a log)

    @property
    def size(self) -> tuple[int, int]:
        return self.frame.shape[1], self.frame.shape[0]

    @cached_property
    def person_box(self) -> Box:
        ys, xs = np.nonzero(self.mask > 0.5)
        w, h = self.size
        if xs.size < 50:
            return 0.3 * w, 0.2 * h, 0.7 * w, float(h)
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


class Jutsu(ABC):
    name: str               # identifier, e.g. "substitution"
    title: str              # shown in the preview, e.g. "Substitution (Naruto)"
    short: str              # short label for the keyboard help line, e.g. "Naruto"
    key: str                # keyboard shortcut that plays the effect without the sign
    hint: str               # one-line description of the sign
    vanish_seconds: float
    arrival_seconds: float

    @abstractmethod
    def checks(self, signs: SignInput) -> dict[str, bool]:
        """Named criteria of the sign; the sign is made when all are True."""

    def detected(self, signs: SignInput) -> bool:
        return all(self.checks(signs).values())

    def start(self, scene: Scene, arriving: bool) -> None:
        """Called once when the effect starts: prepare per-effect state here."""

    @abstractmethod
    def render(self, scene: Scene, t: float, arriving: bool) -> np.ndarray:
        """Frame of the effect, t from 0 to 1.
        Departure: t = 0 -> scene.frame, t = 1 -> exactly hidden_image(scene).
        Arrival:   t = 0 -> scene.hidden, t = 1 -> scene.frame."""

    def hidden_image(self, scene: Scene) -> np.ndarray:
        """What the call sees once the user has vanished. Default: the empty room."""
        return scene.plate

    def leftover_box(self) -> Box | None:
        """Region where hidden_image() differs from the empty room, if any."""
        return None
