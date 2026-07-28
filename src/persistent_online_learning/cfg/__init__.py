"""Fixed context-free language graphs and their one-shot construction API.

The ``cfg`` subsystem is separate from ``persistent_online_learning.generator``:
CFG objects define a static language, while token generators are executable
processes that may later consume such a language.
"""

from .config import CFGSpawnConfig, GrammarConfig, TerminalVocabularyConfig
from .construction import generate_unold_cfg
from .grammar import CFG, Node, Nonterminal, Production, TerminalVocabulary

__all__ = [
    "CFG",
    "CFGSpawnConfig",
    "GrammarConfig",
    "Node",
    "Nonterminal",
    "Production",
    "TerminalVocabulary",
    "TerminalVocabularyConfig",
    "generate_unold_cfg",
]
