"""Random context-free grammar topology construction.

This package owns ordinary CFG structure, exact Unold construction parameters,
construction, and independent consistency validation. Production probabilities,
epsilon-machine conditioning, lexical realization, and sequence generation are
separate downstream responsibilities.
"""

from .model import CFG, Nonterminal, Production, RuleFamily, Terminal
from .specification import UnoldCFGSpecification
from .unold import generate_unold_cfg
from .validation import validate_cfg, validate_unold_cfg

__all__ = [
    "CFG",
    "Nonterminal",
    "Production",
    "RuleFamily",
    "Terminal",
    "UnoldCFGSpecification",
    "generate_unold_cfg",
    "validate_cfg",
    "validate_unold_cfg",
]
