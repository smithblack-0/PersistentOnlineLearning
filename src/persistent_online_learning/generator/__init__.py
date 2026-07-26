"""Composable token generators and reusable process primitives.

Use :func:`build_generator` for plain-dictionary construction, or inject
primitives directly into a concrete flavor when experimenting with a mechanism.
Batching, process replacement, random-state authority, telemetry, checkpoints,
and storage remain outside this package.
"""

from .base import TokenGenerator, build_generator, register_generator
from .flavors import SimpleEpsilonMachine
from .primitives import GenerativeDecoder, HashReduction, StateCore

__all__ = [
    "GenerativeDecoder",
    "HashReduction",
    "SimpleEpsilonMachine",
    "StateCore",
    "TokenGenerator",
    "build_generator",
    "register_generator",
]
