"""Exact input contract for the Unold CFG construction.

Range selection and curriculum policy are deliberately absent. A later resolver
may produce this exact specification without changing the constructor's
scientific contract.
"""

from dataclasses import dataclass


def _require_nonnegative_int(name: str, value: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must use int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_positive_int(name: str, value: int) -> int:
    value = _require_nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True, slots=True)
class UnoldCFGSpecification:
    """One feasible exact grammar request for the literature construction.

    Symbol counts are maxima, matching the paper. The constructor may use fewer
    terminal or nonterminal symbols while producing the exact requested rule
    counts.
    """

    parenthesis_without_nonterminal: int
    parenthesis_with_nonterminal: int
    iteration_rules: int
    branch_rules: int
    max_terminals: int
    max_nonterminals: int

    def __post_init__(self) -> None:
        plain = _require_positive_int(
            "parenthesis_without_nonterminal",
            self.parenthesis_without_nonterminal,
        )
        nested = _require_nonnegative_int(
            "parenthesis_with_nonterminal",
            self.parenthesis_with_nonterminal,
        )
        iteration = _require_nonnegative_int("iteration_rules", self.iteration_rules)
        branch = _require_nonnegative_int("branch_rules", self.branch_rules)
        terminals = _require_positive_int("max_terminals", self.max_terminals)
        nonterminals = _require_positive_int(
            "max_nonterminals",
            self.max_nonterminals,
        )

        terminal_pairs = terminals**2
        minimum_plain_lhs = _ceil_div(plain, terminal_pairs)
        if minimum_plain_lhs > nonterminals:
            raise ValueError(
                "plain parenthesis rules require more nonterminals than allowed"
            )

        connection_slots = nested + iteration + 2 * branch
        if minimum_plain_lhs - 1 > connection_slots:
            raise ValueError(
                "remaining rules cannot connect the terminal-rule components"
            )

        if nested > nonterminals**2 * terminal_pairs:
            raise ValueError(
                "parenthesis-with-nonterminal count exceeds the symbol capacity"
            )
        if iteration > 2 * nonterminals**2 * terminals:
            raise ValueError("iteration count exceeds the symbol capacity")
        if branch > nonterminals**3:
            raise ValueError("branch count exceeds the symbol capacity")

    @property
    def total_rules(self) -> int:
        """Total number of productions requested."""

        return (
            self.parenthesis_without_nonterminal
            + self.parenthesis_with_nonterminal
            + self.iteration_rules
            + self.branch_rules
        )
