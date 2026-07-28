"""Static context-free grammars.

A grammar is a rooted graph of nonterminal nodes. Each nonterminal owns its
right-hand-side alternatives directly; terminals are leaf values. The graph is
mutable only while it is being assembled. Creating ``CFG`` validates and seals
the complete reachable graph, after which it is the fixed grammar produced by a
generator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TypeAlias


def _require_nonnegative_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must use int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True, slots=True)
class Terminal:
    """One abstract terminal category in a generated language."""

    category: int

    def __post_init__(self) -> None:
        _require_nonnegative_int("terminal category", self.category)


class Nonterminal:
    """One CFG nonterminal and the alternatives it can expand into.

    Nodes are created before all graph edges are known so recursive grammars can
    be assembled naturally. ``add_alternative`` is available only until the
    containing ``CFG`` is created; ``CFG`` then seals every reachable node.
    """

    __slots__ = ("name", "_alternatives", "_sealed")

    def __init__(self, name: str) -> None:
        if type(name) is not str or not name:
            raise TypeError("nonterminal name must be a nonempty string")
        self.name = name
        self._alternatives: list[Alternative] | tuple[Alternative, ...] = []
        self._sealed = False

    def add_alternative(self, *symbols: Symbol) -> None:
        """Add one ordered right-hand-side alternative while assembling a CFG."""

        if self._sealed:
            raise RuntimeError(f"nonterminal {self.name!r} belongs to a sealed CFG")
        if not symbols:
            raise ValueError("CFG alternatives must not be empty")
        if not all(isinstance(symbol, (Terminal, Nonterminal)) for symbol in symbols):
            raise TypeError("CFG alternatives may contain only terminals and nonterminals")
        alternative = tuple(symbols)
        assert isinstance(self._alternatives, list)
        if alternative in self._alternatives:
            raise ValueError(f"nonterminal {self.name!r} already owns that alternative")
        self._alternatives.append(alternative)

    @property
    def alternatives(self) -> tuple[Alternative, ...]:
        """Return this nonterminal's alternatives in construction order."""

        return tuple(self._alternatives)

    def _seal(self) -> None:
        if not self._sealed:
            self._alternatives = tuple(self._alternatives)
            self._sealed = True

    def __repr__(self) -> str:
        return f"Nonterminal({self.name!r})"


Symbol: TypeAlias = Terminal | Nonterminal
Alternative: TypeAlias = tuple[Symbol, ...]


class CFG:
    """A complete fixed context-free grammar rooted at one nonterminal.

    The grammar is exactly the graph reachable from ``start``. Construction
    requires every reachable node to own at least one alternative and every node
    to have a finite terminal derivation, then seals the graph against mutation.
    """

    __slots__ = ("start", "nonterminals", "terminals", "rule_count")

    def __init__(self, start: Nonterminal) -> None:
        if not isinstance(start, Nonterminal):
            raise TypeError("CFG start must be a Nonterminal")

        nonterminals = self._reachable_nonterminals(start)
        names: set[str] = set()
        terminals: set[Terminal] = set()
        for node in nonterminals:
            if node.name in names:
                raise ValueError(f"duplicate nonterminal name {node.name!r}")
            names.add(node.name)
            if not node.alternatives:
                raise ValueError(f"nonterminal {node.name!r} has no alternatives")
            for alternative in node.alternatives:
                terminals.update(
                    symbol for symbol in alternative if isinstance(symbol, Terminal)
                )

        unresolved_by_alternative: dict[Nonterminal, list[int]] = {
            node: [
                sum(isinstance(symbol, Nonterminal) for symbol in alternative)
                for alternative in node.alternatives
            ]
            for node in nonterminals
        }
        dependents: dict[Nonterminal, list[tuple[Nonterminal, int]]] = {
            node: [] for node in nonterminals
        }
        productive: set[Nonterminal] = set()
        pending_productive: deque[Nonterminal] = deque()
        for owner in nonterminals:
            for alternative_index, alternative in enumerate(owner.alternatives):
                children = [
                    symbol
                    for symbol in alternative
                    if isinstance(symbol, Nonterminal)
                ]
                if not children and owner not in productive:
                    productive.add(owner)
                    pending_productive.append(owner)
                for child in children:
                    dependents[child].append((owner, alternative_index))

        while pending_productive:
            child = pending_productive.popleft()
            for owner, alternative_index in dependents[child]:
                unresolved_by_alternative[owner][alternative_index] -= 1
                if (
                    unresolved_by_alternative[owner][alternative_index] == 0
                    and owner not in productive
                ):
                    productive.add(owner)
                    pending_productive.append(owner)

        if len(productive) != len(nonterminals):
            missing = next(node for node in nonterminals if node not in productive)
            raise ValueError(
                f"nonterminal {missing.name!r} has no finite terminal derivation"
            )

        for node in nonterminals:
            node._seal()

        self.start = start
        self.nonterminals = tuple(nonterminals)
        self.terminals = frozenset(terminals)
        self.rule_count = sum(len(node.alternatives) for node in nonterminals)

    @staticmethod
    def _reachable_nonterminals(start: Nonterminal) -> list[Nonterminal]:
        ordered: list[Nonterminal] = []
        seen: set[Nonterminal] = {start}
        pending: deque[Nonterminal] = deque([start])
        while pending:
            node = pending.popleft()
            ordered.append(node)
            for alternative in node.alternatives:
                for symbol in alternative:
                    if isinstance(symbol, Nonterminal) and symbol not in seen:
                        seen.add(symbol)
                        pending.append(symbol)
        return ordered
