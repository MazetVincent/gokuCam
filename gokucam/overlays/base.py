"""The contract of an overlay: an effect drawn on top of the image while a sign lasts.

Unlike a jutsu, an overlay does not make you vanish. It fades in when its sign has been
held for a moment, stays while the sign is visible (plus `linger_seconds`, so hiding the
sign for an instant does not stop it), then fades out. Its key toggles it by hand.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..jutsus.base import Scene, SignInput


class Overlay(ABC):
    name: str                # identifier, e.g. "this_is_fine"
    title: str               # shown in the preview
    short: str               # keyboard help line
    key: str                 # toggles the overlay by hand
    hint: str                # how to trigger it
    fade_in_seconds: float = 1.0
    fade_out_seconds: float = 1.0
    linger_seconds: float = 1.5

    @abstractmethod
    def checks(self, signs: SignInput) -> dict[str, bool]:
        """Named criteria of the sign; the overlay starts when all are True."""

    def start(self, scene: Scene) -> None:
        """Called once each time the overlay starts: prepare per-run state here."""

    @abstractmethod
    def render(self, image: np.ndarray, scene: Scene, elapsed: float, level: float) -> np.ndarray:
        """Draws the overlay on `image` (the frame about to be sent to the call).
        `elapsed`: seconds since the overlay started; `level`: 0..1, its fade in / out.
        `scene.mask` is the person in `image` (all zeros when the user is invisible), so
        the overlay can stay behind the person. Must return `image` unchanged at level 0."""
