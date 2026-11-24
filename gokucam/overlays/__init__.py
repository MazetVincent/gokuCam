"""Registry of overlays (effects drawn on top of the image while a sign lasts)."""

from .absolute_cinema import AbsoluteCinema
from .base import Overlay
from .infinity import Infinity
from .this_is_fine import ThisIsFine


def default_overlays() -> list[Overlay]:
    return [ThisIsFine(), AbsoluteCinema(), Infinity()]


__all__ = ["AbsoluteCinema", "Infinity", "Overlay", "ThisIsFine", "default_overlays"]
