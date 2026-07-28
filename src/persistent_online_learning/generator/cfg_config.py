"""Configuration and feasibility mathematics for fixed CFG construction.

The grammar inequalities implement equation system (10) from:

O. Unold, A. Kaczmarek, and Ł. Culer, "Iterative method of generating
artificial context-free grammars," arXiv:1911.05801, 2019.

Paper notation maps to this module as follows:

``R_P^-`` -> ``terminal_pair_rules``      (A -> a b)
``R_P^+`` -> ``parenthesis_rules``       (A -> a B b)
``R_I``   -> ``iteration_rules``         (A -> a B or B a)
``R_B``   -> ``branch_rules``            (A -> B C)
``S_T``   -> ``terminal_count``
``S_NT``  -> ``max_nonterminals``
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt


def _require_count(name: str, value: int, *, positive: bool = False) -> None:
    """Require one exact integer count at its configuration-owner boundary."""

    if type(value) is not int:
        raise TypeError(f"{name} must use int")
    minimum = 1 if positive else 0
    if value < minimum:
        word = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {word}")


@dataclass(frozen=True, slots=True)
class GrammarConfig:
    """Exact Unold rule counts and the nonterminal-symbol limit."""

    terminal_pair_rules: int
    parenthesis_rules: int
    iteration_rules: int
    branch_rules: int
    max_nonterminals: int

    def __post_init__(self) -> None:
        _require_count("terminal_pair_rules", self.terminal_pair_rules, positive=True)
        _require_count("parenthesis_rules", self.parenthesis_rules)
        _require_count("iteration_rules", self.iteration_rules)
        _require_count("branch_rules", self.branch_rules)
        _require_count("max_nonterminals", self.max_nonterminals, positive=True)


@dataclass(frozen=True, slots=True)
class TerminalVocabularyConfig:
    """Exact construction request for vocabulary-bearing terminal nodes.

    Every terminal receives ``tokens_per_terminal`` distinct IDs.  Across all
    terminals, every ID in ``range(vocabulary_size)`` is used at least once.
    Different terminals may overlap or contain identical token sets.
    """

    terminal_count: int
    vocabulary_size: int
    tokens_per_terminal: int

    def __post_init__(self) -> None:
        _require_count("terminal_count", self.terminal_count, positive=True)
        _require_count("vocabulary_size", self.vocabulary_size, positive=True)
        _require_count("tokens_per_terminal", self.tokens_per_terminal, positive=True)
        if self.tokens_per_terminal > self.vocabulary_size:
            raise ValueError("tokens_per_terminal cannot exceed vocabulary_size")
        if self.terminal_count * self.tokens_per_terminal < self.vocabulary_size:
            raise ValueError(
                "terminal vocabularies do not provide enough slots to cover "
                "the vocabulary"
            )


@dataclass(frozen=True, slots=True)
class CFGSpawnConfig:
    """One compatible syntax request and terminal-vocabulary request."""

    grammar: GrammarConfig
    terminal_vocabularies: TerminalVocabularyConfig

    def __post_init__(self) -> None:
        if not isinstance(self.grammar, GrammarConfig):
            raise TypeError("grammar must be GrammarConfig")
        if not isinstance(self.terminal_vocabularies, TerminalVocabularyConfig):
            raise TypeError(
                "terminal_vocabularies must be TerminalVocabularyConfig"
            )
        _construction_plans(self)


@dataclass(frozen=True, slots=True)
class _ConstructionPlan:
    """One feasible total/initial nonterminal-count combination."""

    target_nonterminals: int
    initial_nonterminal_min: int
    initial_nonterminal_max: int


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return the integer ceiling used by the paper's capacity inequalities."""

    return (numerator + denominator - 1) // denominator


def _ceil_sqrt_ratio(count: int, multiplier: int) -> int:
    """Find the least N satisfying ``count <= multiplier * N**2``."""

    if count == 0:
        return 0
    quotient = _ceil_div(count, multiplier)
    root = isqrt(quotient)
    return root if root * root == quotient else root + 1


def _ceil_cuberoot(count: int) -> int:
    """Find the least N satisfying ``count <= N**3`` without float rounding."""

    if count == 0:
        return 0
    low, high = 0, 1
    while high**3 < count:
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if middle**3 >= count:
            high = middle
        else:
            low = middle
    return high


def _construction_plans(config: CFGSpawnConfig) -> tuple[_ConstructionPlan, ...]:
    """Return every feasible node-count plan or reject the composed request."""

    grammar = config.grammar
    terminal_count = config.terminal_vocabularies.terminal_count
    terminal_square = terminal_count**2

    terminal_positions = (
        2 * grammar.terminal_pair_rules
        + 2 * grammar.parenthesis_rules
        + grammar.iteration_rules
    )
    if terminal_positions < terminal_count:
        raise ValueError(
            "grammar does not contain enough terminal positions to use every "
            "TerminalVocabulary"
        )

    minimum_plain = _ceil_div(grammar.terminal_pair_rules, terminal_square)
    connection_slots = (
        grammar.parenthesis_rules
        + grammar.iteration_rules
        + 2 * grammar.branch_rules
    )
    if minimum_plain > grammar.max_nonterminals:
        raise ValueError("terminal-pair rules exceed nonterminal capacity")
    if minimum_plain > connection_slots + 1:
        raise ValueError(
            "remaining rules cannot connect the initial productive nonterminals"
        )
    if (
        grammar.parenthesis_rules
        > grammar.max_nonterminals**2 * terminal_square
    ):
        raise ValueError("parenthesis rules exceed symbol capacity")
    if (
        grammar.iteration_rules
        > 2 * grammar.max_nonterminals**2 * terminal_count
    ):
        raise ValueError("iteration rules exceed symbol capacity")
    if grammar.branch_rules > grammar.max_nonterminals**3:
        raise ValueError("branch rules exceed symbol capacity")

    minimum_nonterminals = max(
        1,
        minimum_plain,
        _ceil_sqrt_ratio(grammar.parenthesis_rules, terminal_square),
        _ceil_sqrt_ratio(grammar.iteration_rules, 2 * terminal_count),
        _ceil_cuberoot(grammar.branch_rules),
    )
    remaining_rules = (
        grammar.parenthesis_rules
        + grammar.iteration_rules
        + grammar.branch_rules
    )
    total_rules = grammar.terminal_pair_rules + remaining_rules
    maximum_nonterminals = min(
        grammar.max_nonterminals,
        connection_slots + 1,
        total_rules,
    )

    plans: list[_ConstructionPlan] = []
    for target_nonterminals in range(
        minimum_nonterminals, maximum_nonterminals + 1
    ):
        initial_min = max(
            minimum_plain,
            target_nonterminals - remaining_rules,
        )
        initial_max = min(
            grammar.terminal_pair_rules,
            target_nonterminals,
            connection_slots + 1,
        )
        if initial_min <= initial_max:
            plans.append(
                _ConstructionPlan(
                    target_nonterminals=target_nonterminals,
                    initial_nonterminal_min=initial_min,
                    initial_nonterminal_max=initial_max,
                )
            )
    if not plans:
        raise ValueError(
            "rule counts cannot create and connect a feasible nonterminal graph"
        )
    return tuple(plans)
