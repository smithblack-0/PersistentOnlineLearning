"""Static language construction and composable token-process components.

``generate_unold_cfg`` produces one fixed lexicalized context-free grammar.
``build_generator`` constructs runtime token generators from plain dictionaries.
Batching, process replacement, random-state authority, telemetry, checkpoints,
and storage remain outside this package.
"""

from .base import TokenGenerator, build_generator, register_generator
from .flavors import SimpleEpsilonMachine
from .grammar import (
    CFG,
    GrammarSymbol,
    LexicalizedCFG,
    Lexicon,
    LexiconEntry,
    Node,
    Nonterminal,
    Production,
    Terminal,
    Vocabulary,
)
from .primitives import GenerativeDecoder, HashReduction, StateCore
from .unold_cfg import LexiconParameters, UnoldCFGParameters, generate_unold_cfg

__all__ = [
    "CFG",
    "GenerativeDecoder",
    "GrammarSymbol",
    "HashReduction",
    "LexicalizedCFG",
    "Lexicon",
    "LexiconEntry",
    "LexiconParameters",
    "Node",
    "Nonterminal",
    "Production",
    "SimpleEpsilonMachine",
    "StateCore",
    "Terminal",
    "TokenGenerator",
    "UnoldCFGParameters",
    "Vocabulary",
    "build_generator",
    "generate_unold_cfg",
    "register_generator",
]
