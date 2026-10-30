"""Registry of jutsus. To add a sign, write a Jutsu subclass and list it here
(see docs/ADDING_A_SIGN.md)."""

from .base import Box, Jutsu, Scene, SignInput
from .instant_transmission import InstantTransmission
from .into_the_bush import IntoTheBush
from .peace_out import PeaceOut
from .substitution import Substitution


def default_jutsus() -> list[Jutsu]:
    """Fresh instances (jutsus keep per-effect state, such as the log layout)."""
    return [InstantTransmission(), Substitution(), PeaceOut(), IntoTheBush()]


__all__ = ["Box", "Jutsu", "Scene", "SignInput", "InstantTransmission", "IntoTheBush", "PeaceOut",
           "Substitution", "default_jutsus"]
