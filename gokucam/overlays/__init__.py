"""Registry of overlays (effects drawn on top of the image while a sign lasts)."""

from .base import Overlay
from .this_is_fine import ThisIsFine


def default_overlays() -> list[Overlay]:
    return [ThisIsFine()]


__all__ = ["Overlay", "ThisIsFine", "default_overlays"]
