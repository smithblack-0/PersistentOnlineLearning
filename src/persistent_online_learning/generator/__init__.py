"""Composable token processes and synthetic-language construction tools.

Use :func:`build_generator` for stateful token-generator flavors. The ``cfg``
exports construct ordinary grammar topology only; production probabilities,
epsilon-machine conditioning, lexical realization, batching, telemetry,
checkpoints, and storage remain separate responsibilities.
"""

from .base import TokenGenerator, build_generator, register_generator
from .cfg import (
    CFG,
    Nonterminal,
    Production,
    RuleFamily,
    Terminal,
    UnoldCFGSpecification,
    generate_unold_cfg,
    validate_cfg,
    validate_unold_cfg,
)
from .flavors import SimpleEpsilonMachine
from .primitives import GenerativeDecoder, HashReduction, StateCore

__all__ = [
    "CFG",
    "GenerativeDecoder",
    "HashReduction",
    "Nonterminal",
    "Production",
    "RuleFamily",
    "SimpleEpsilonMachine",
    "StateCore",
    "Terminal",
    "TokenGenerator",
    "UnoldCFGSpecification",
    "build_generator",
    "generate_unold_cfg",
    "register_generator",
    "validate_cfg",
    "validate_unold_cfg",
]
