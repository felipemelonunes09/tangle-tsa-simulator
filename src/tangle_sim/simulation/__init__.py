"""Discrete-event simulation package.

Concrete classes live in submodules; keeping this initializer light avoids an
algorithm/RNG import cycle.
"""

from .rng import SeedManager

__all__ = ["SeedManager"]
