"""Static grammar generation and composable token-process components.

``generate_unold_cfg`` produces one fixed context-free grammar. ``build_generator``
constructs runtime token generators from plain dictionaries. Batching, process
replacement, random-state authority, telemetry, checkpoints, and storage remain
outside this package.
"""

from .base import TokenGenerator, build_generator, register_generator
from .flavors import SimpleEpsilonMachine
from .grammar import CFG, Alternative, Nonterminal, Symbol, Terminal
from .primitives import GenerativeDecoder, HashReduction, StateCore
from .unold_cfg import UnoldCFGParameters, generate_unold_cfg

__all__ = [
    "Alternative",
    "CFG",
    "GenerativeDecoder",
    "HashReduction",
    "Nonterminal",
    "SimpleEpsilonMachine",
    "StateCore",
    "Symbol",
    "Terminal",
    "TokenGenerator",
    "UnoldCFGParameters",
    "build_generator",
    "generate_unold_cfg",
    "register_generator",
]
