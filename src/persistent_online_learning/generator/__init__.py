"""Composable runtime token generators and reusable process primitives.

Use :func:`build_generator` for dictionary-driven construction of executable token
processes. Static context-free languages live in ``persistent_online_learning.cfg``;
this package does not own one-shot language construction.
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
