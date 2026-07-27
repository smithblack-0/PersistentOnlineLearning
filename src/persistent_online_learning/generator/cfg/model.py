"""Ordinary context-free grammars used by the project.

The structures describe grammar topology only. They do not own production
probabilities, epsilon-machine state, lexical realization, derivation state, or
training configuration.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


def _require_symbol_index(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must use int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True, order=True, slots=True)
class Terminal:
    """One abstract terminal category in a generated grammar."""

    index: int

    def __post_init__(self) -> None:
        _require_symbol_index("terminal index", self.index)


@dataclass(frozen=True, order=True, slots=True)
class Nonterminal:
    """One nonterminal symbol in a generated grammar."""

    index: int

    def __post_init__(self) -> None:
        _require_symbol_index("nonterminal index", self.index)


Symbol: TypeAlias = Terminal | Nonterminal


class RuleFamily(StrEnum):
    """The four production families admitted by the Unold construction."""

    PARENTHESIS_WITHOUT_NONTERMINAL = "parenthesis_without_nonterminal"
    PARENTHESIS_WITH_NONTERMINAL = "parenthesis_with_nonterminal"
    ITERATION = "iteration"
    BRANCH = "branch"


@dataclass(frozen=True, slots=True)
class Production:
    """One ordered CFG production."""

    lhs: Nonterminal
    rhs: tuple[Symbol, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.lhs, Nonterminal):
            raise TypeError("production lhs must be a Nonterminal")
        if type(self.rhs) is not tuple or not self.rhs:
            raise TypeError("production rhs must be a nonempty tuple")
        if not all(isinstance(symbol, (Terminal, Nonterminal)) for symbol in self.rhs):
            raise TypeError("production rhs contains an invalid symbol")

    @property
    def family(self) -> RuleFamily:
        """Classify this production under the paper's four rule families."""

        rhs = self.rhs
        if len(rhs) == 2 and all(isinstance(symbol, Terminal) for symbol in rhs):
            return RuleFamily.PARENTHESIS_WITHOUT_NONTERMINAL
        if (
            len(rhs) == 3
            and isinstance(rhs[0], Terminal)
            and isinstance(rhs[1], Nonterminal)
            and isinstance(rhs[2], Terminal)
        ):
            return RuleFamily.PARENTHESIS_WITH_NONTERMINAL
        if len(rhs) == 2 and (
            (
                isinstance(rhs[0], Terminal)
                and isinstance(rhs[1], Nonterminal)
            )
            or (
                isinstance(rhs[0], Nonterminal)
                and isinstance(rhs[1], Terminal)
            )
        ):
            return RuleFamily.ITERATION
        if len(rhs) == 2 and all(isinstance(symbol, Nonterminal) for symbol in rhs):
            return RuleFamily.BRANCH
        raise ValueError("production does not belong to an Unold rule family")


@dataclass(frozen=True, slots=True)
class CFG:
    """An ordinary context-free grammar in construction order."""

    start: Nonterminal
    productions: tuple[Production, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.start, Nonterminal):
            raise TypeError("grammar start must be a Nonterminal")
        if type(self.productions) is not tuple or not self.productions:
            raise TypeError("grammar productions must be a nonempty tuple")
        if not all(isinstance(rule, Production) for rule in self.productions):
            raise TypeError("grammar contains an invalid production")

    @property
    def terminals(self) -> frozenset[Terminal]:
        """Terminal categories appearing on production right-hand sides."""

        return frozenset(
            symbol
            for production in self.productions
            for symbol in production.rhs
            if isinstance(symbol, Terminal)
        )

    @property
    def nonterminals(self) -> frozenset[Nonterminal]:
        """Nonterminals appearing anywhere in the grammar."""

        symbols = {self.start}
        for production in self.productions:
            symbols.add(production.lhs)
            symbols.update(
                symbol
                for symbol in production.rhs
                if isinstance(symbol, Nonterminal)
            )
        return frozenset(symbols)
