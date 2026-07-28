"""Static language construction and composable token-process components.

Use :func:`generate_unold_cfg` to construct one fixed vocabulary-bearing CFG, or
:func:`build_generator` for dictionary-driven runtime token-process flavors.
Batching, process replacement, telemetry, checkpoints, and storage remain
outside this package.
"""

from .base import TokenGenerator, build_generator, register_generator
from .cfg_config import CFGSpawnConfig, GrammarConfig, TerminalVocabularyConfig
from .flavors import SimpleEpsilonMachine
from .grammar import CFG, Node, Nonterminal, Production, TerminalVocabulary
from .primitives import GenerativeDecoder, HashReduction, StateCore
from .terminal_vocabulary import build_terminal_vocabularies
from .unold_cfg import generate_unold_cfg

__all__ = [
    "CFG",
    "CFGSpawnConfig",
    "GenerativeDecoder",
    "GrammarConfig",
    "HashReduction",
    "Node",
    "Nonterminal",
    "Production",
    "SimpleEpsilonMachine",
    "StateCore",
    "TerminalVocabulary",
    "TerminalVocabularyConfig",
    "TokenGenerator",
    "build_generator",
    "build_terminal_vocabularies",
    "generate_unold_cfg",
    "register_generator",
]
