"""Registry of overlays (effects drawn on top of the image while a sign lasts)."""

from .absolute_cinema import AbsoluteCinema
from .base import Overlay
from .this_is_fine import ThisIsFine


def default_overlays() -> list[Overlay]:
    return [ThisIsFine(), AbsoluteCinema()]


__all__ = ["AbsoluteCinema", "Overlay", "ThisIsFine", "default_overlays"]
