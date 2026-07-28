"""Public requests and compatibility constraints for fixed CFG construction.

The rule-count inequalities in this module begin with equation system (10) from:

O. Unold, A. Kaczmarek, and Ł. Culer, "Iterative method of generating
artificial context-free grammars," arXiv:1911.05801, 2019.

Paper notation maps to the public fields as follows:

``R_P^-`` -> ``terminal_pair_rules``      (A -> a b)
``R_P^+`` -> ``parenthesis_rules``       (A -> a B b)
``R_I``   -> ``iteration_rules``         (A -> a B or B a)
``R_B``   -> ``branch_rules``            (A -> B C)
``S_T``   -> ``terminal_count``
``S_NT``  -> ``max_nonterminals``

The project adds one composition requirement absent from the paper: every supplied
``TerminalVocabulary`` must appear in the generated syntax. This module validates
request-level compatibility only. Choosing an actual nonterminal count and initial
foundation is a construction decision owned by ``cfg.construction.planning``.
"""

from __future__ import annotations

from dataclasses import dataclass


def _require_positive_int(name: str, value: int) -> None:
    """Require a configuration field to be a strictly positive built-in integer."""

    if type(value) is not int:
        raise TypeError(f"{name} must use int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative_int(name: str, value: int) -> None:
    """Require a configuration field to be a nonnegative built-in integer."""

    if type(value) is not int:
        raise TypeError(f"{name} must use int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return the exact integer ceiling used by request-capacity calculations."""

    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True, slots=True)
class GrammarConfig:
    """Request exact Unold rule-family counts under a nonterminal upper bound.

    The four rule counts are exact outputs requested from construction. The paper's
    ``S_NT`` is a maximum, so ``max_nonterminals`` deliberately leaves the actual
    nonterminal count for the construction planner to choose.

    Attributes:
        terminal_pair_rules: Exact number of terminal-only ``A -> a b`` rules.
            These rules establish the initial productive foundation.
        parenthesis_rules: Exact number of ``A -> a B b`` rules.
        iteration_rules: Exact total number of ``A -> a B`` and ``A -> B a`` rules.
        branch_rules: Exact number of ``A -> B C`` rules.
        max_nonterminals: Maximum number of nonterminal symbols construction may
            create; the actual count is selected from the feasible range.
    """

    terminal_pair_rules: int
    parenthesis_rules: int
    iteration_rules: int
    branch_rules: int
    max_nonterminals: int

    def __post_init__(self) -> None:
        _require_positive_int("terminal_pair_rules", self.terminal_pair_rules)
        _require_nonnegative_int("parenthesis_rules", self.parenthesis_rules)
        _require_nonnegative_int("iteration_rules", self.iteration_rules)
        _require_nonnegative_int("branch_rules", self.branch_rules)
        _require_positive_int("max_nonterminals", self.max_nonterminals)


@dataclass(frozen=True, slots=True)
class TerminalVocabularyConfig:
    """Request the fixed terminal alphabet that syntax construction will consume.

    Every terminal node receives the same number of distinct concrete token IDs.
    Membership is intentionally nonexclusive: different terminals may overlap or
    even contain identical token sets. Across the complete terminal alphabet every
    concrete ID is required to appear at least once.

    Attributes:
        terminal_count: Number of distinct ``TerminalVocabulary`` graph nodes.
        vocabulary_size: Size of the concrete token universe, interpreted as IDs
            ``0`` through ``vocabulary_size - 1``.
        tokens_per_terminal: Number of distinct concrete IDs stored by each
            terminal node. This is per-node uniqueness, not global exclusivity.
    """

    terminal_count: int
    vocabulary_size: int
    tokens_per_terminal: int

    def __post_init__(self) -> None:
        _require_positive_int("terminal_count", self.terminal_count)
        _require_positive_int("vocabulary_size", self.vocabulary_size)
        _require_positive_int("tokens_per_terminal", self.tokens_per_terminal)
        if self.tokens_per_terminal > self.vocabulary_size:
            raise ValueError("tokens_per_terminal cannot exceed vocabulary_size")
        if self.terminal_count * self.tokens_per_terminal < self.vocabulary_size:
            raise ValueError(
                "terminal vocabularies do not provide enough slots to cover "
                "the vocabulary"
            )


@dataclass(frozen=True, slots=True)
class CFGSpawnConfig:
    """Compose one syntax request with the terminal alphabet it must use.

    This object is the public boundary where otherwise-valid grammar and terminal
    requests become one construction request. It rejects only compatibility that
    can be decided from the requested maxima and exact counts. Transient choices,
    such as the actual nonterminal count, remain construction responsibilities.

    Attributes:
        grammar: Exact rule-family counts and nonterminal upper bound.
        terminal_vocabularies: Shape and concrete coverage requested for the
            terminal alphabet used by those rules.
    """

    grammar: GrammarConfig
    terminal_vocabularies: TerminalVocabularyConfig

    def __post_init__(self) -> None:
        if not isinstance(self.grammar, GrammarConfig):
            raise TypeError("grammar must be GrammarConfig")
        if not isinstance(self.terminal_vocabularies, TerminalVocabularyConfig):
            raise TypeError(
                "terminal_vocabularies must be TerminalVocabularyConfig"
            )
        _require_spawn_compatibility(self)


def _terminal_position_count(grammar: GrammarConfig) -> int:
    """Count concrete terminal slots available across the requested rule families."""

    return (
        2 * grammar.terminal_pair_rules
        + 2 * grammar.parenthesis_rules
        + grammar.iteration_rules
    )


def _connection_edge_count(grammar: GrammarConfig) -> int:
    """Count nonterminal RHS edges available to connect productive components."""

    return grammar.parenthesis_rules + grammar.iteration_rules + 2 * grammar.branch_rules


def _minimum_plain_nonterminals(config: CFGSpawnConfig) -> int:
    """Return the fewest LHS nodes able to host all unique terminal-pair rules."""

    terminal_count = config.terminal_vocabularies.terminal_count
    return _ceil_div(config.grammar.terminal_pair_rules, terminal_count**2)


def _require_spawn_compatibility(config: CFGSpawnConfig) -> None:
    """Reject request pairs that no allowed Unold construction could satisfy.

    These are request-level bounds, not a second construction algorithm. The paper
    supplies the unique-rule capacity inequalities; exact use of every terminal is
    the project-specific additional constraint. Planning later chooses one concrete
    nonterminal count inside these bounds.
    """

    grammar = config.grammar
    terminal_count = config.terminal_vocabularies.terminal_count
    terminal_square = terminal_count**2

    if _terminal_position_count(grammar) < terminal_count:
        raise ValueError(
            "grammar does not contain enough terminal positions to use every "
            "TerminalVocabulary"
        )

    minimum_plain = _minimum_plain_nonterminals(config)
    if minimum_plain > grammar.max_nonterminals:
        raise ValueError("terminal-pair rules exceed nonterminal capacity")
    if minimum_plain > _connection_edge_count(grammar) + 1:
        raise ValueError(
            "remaining rules cannot connect the initial productive nonterminals"
        )
    if grammar.parenthesis_rules > grammar.max_nonterminals**2 * terminal_square:
        raise ValueError("parenthesis rules exceed symbol capacity")
    if grammar.iteration_rules > 2 * grammar.max_nonterminals**2 * terminal_count:
        raise ValueError("iteration rules exceed symbol capacity")
    if grammar.branch_rules > grammar.max_nonterminals**3:
        raise ValueError("branch rules exceed symbol capacity")
