"""Fixed context-free grammars with node-owned lexical realization.

A grammar is a rooted graph of nonterminals. Each nonterminal owns its ordered
right-hand-side alternatives, and each abstract terminal category owns the
concrete vocabulary indices that can realize it. Nodes are mutable only while a
generator assembles the graph and lexicon. Creating ``CFG`` validates the whole
reachable language structure and seals it.
"""

from __future__ import annotations

from collections import deque
from typing import TypeAlias


def _require_nonnegative_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must use int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_positive_int(name: str, value: int) -> None:
    _require_nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


class Terminal:
    """One abstract terminal category and its concrete vocabulary realizations.

    A generator creates the category while assembling syntax, then assigns its
    complete vocabulary exactly once. Constructing the containing ``CFG`` seals
    the category against later mutation.
    """

    __slots__ = ("category", "_vocabulary", "_sealed")

    def __init__(self, category: int) -> None:
        _require_nonnegative_int("terminal category", category)
        self.category = category
        self._vocabulary: tuple[int, ...] = ()
        self._sealed = False

    def set_vocabulary(self, vocabulary: tuple[int, ...]) -> None:
        """Assign the distinct vocabulary indices that realize this category."""

        if self._sealed:
            raise RuntimeError(
                f"terminal category {self.category} belongs to a sealed CFG"
            )
        if self._vocabulary:
            raise RuntimeError(
                f"terminal category {self.category} already has a vocabulary"
            )
        if type(vocabulary) is not tuple or not vocabulary:
            raise TypeError("terminal vocabulary must be a nonempty tuple")
        for index in vocabulary:
            _require_nonnegative_int("vocabulary index", index)
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("terminal vocabulary must not contain duplicate indices")
        self._vocabulary = vocabulary

    @property
    def vocabulary(self) -> tuple[int, ...]:
        """Concrete vocabulary indices assigned to this category."""

        return self._vocabulary

    def _seal(self) -> None:
        self._sealed = True

    def __repr__(self) -> str:
        return f"Terminal({self.category})"


class Nonterminal:
    """One CFG nonterminal and the alternatives it can expand into.

    Nodes are created before all graph edges are known so recursive grammars can
    be assembled naturally. ``add_alternative`` is available only until the
    containing ``CFG`` validates and seals the complete reachable graph.
    """

    __slots__ = ("name", "_alternatives", "_alternative_set", "_sealed")

    def __init__(self, name: str) -> None:
        if type(name) is not str or not name:
            raise TypeError("nonterminal name must be a nonempty string")
        self.name = name
        self._alternatives: list[Alternative] | tuple[Alternative, ...] = []
        self._alternative_set: set[Alternative] | None = set()
        self._sealed = False

    def add_alternative(self, *symbols: Symbol) -> None:
        """Add one ordered right-hand-side alternative while assembling a CFG."""

        if self._sealed:
            raise RuntimeError(f"nonterminal {self.name!r} belongs to a sealed CFG")
        if not symbols:
            raise ValueError("CFG alternatives must not be empty")
        if not all(isinstance(symbol, (Terminal, Nonterminal)) for symbol in symbols):
            raise TypeError(
                "CFG alternatives may contain only terminals and nonterminals"
            )
        alternative = tuple(symbols)
        assert isinstance(self._alternatives, list)
        assert self._alternative_set is not None
        if alternative in self._alternative_set:
            raise ValueError(f"nonterminal {self.name!r} already owns that alternative")
        self._alternatives.append(alternative)
        self._alternative_set.add(alternative)

    @property
    def alternatives(self) -> tuple[Alternative, ...]:
        """Return this nonterminal's alternatives in construction order."""

        return tuple(self._alternatives)

    def _seal(self) -> None:
        if not self._sealed:
            self._alternatives = tuple(self._alternatives)
            self._alternative_set = None
            self._sealed = True

    def __repr__(self) -> str:
        return f"Nonterminal({self.name!r})"


Symbol: TypeAlias = Terminal | Nonterminal
Alternative: TypeAlias = tuple[Symbol, ...]


class CFG:
    """A complete fixed CFG over abstract categories and one concrete vocabulary.

    The grammar is exactly the graph reachable from ``start``. Every reachable
    nonterminal must own a finite terminal derivation. Every reachable terminal
    category must own a nonempty vocabulary, and their union must cover every
    index in ``range(vocabulary_size)``. Categories may overlap.
    """

    __slots__ = (
        "start",
        "nonterminals",
        "terminals",
        "vocabulary_size",
        "rule_count",
    )

    def __init__(self, start: Nonterminal, vocabulary_size: int) -> None:
        if not isinstance(start, Nonterminal):
            raise TypeError("CFG start must be a Nonterminal")
        _require_positive_int("vocabulary_size", vocabulary_size)

        nonterminals = self._reachable_nonterminals(start)
        names: set[str] = set()
        terminals_by_category: dict[int, Terminal] = {}
        for node in nonterminals:
            if node.name in names:
                raise ValueError(f"duplicate nonterminal name {node.name!r}")
            names.add(node.name)
            if not node.alternatives:
                raise ValueError(f"nonterminal {node.name!r} has no alternatives")
            for alternative in node.alternatives:
                for symbol in alternative:
                    if not isinstance(symbol, Terminal):
                        continue
                    existing = terminals_by_category.get(symbol.category)
                    if existing is not None and existing is not symbol:
                        raise ValueError(
                            f"duplicate terminal category {symbol.category}"
                        )
                    terminals_by_category[symbol.category] = symbol

        terminals = tuple(
            terminals_by_category[category]
            for category in sorted(terminals_by_category)
        )
        used_vocabulary: set[int] = set()
        for terminal in terminals:
            if not terminal.vocabulary:
                raise ValueError(
                    f"terminal category {terminal.category} has no vocabulary"
                )
            for index in terminal.vocabulary:
                if index >= vocabulary_size:
                    raise ValueError(
                        f"vocabulary index {index} is outside configured size "
                        f"{vocabulary_size}"
                    )
                used_vocabulary.add(index)

        expected_vocabulary = set(range(vocabulary_size))
        if used_vocabulary != expected_vocabulary:
            missing = min(expected_vocabulary - used_vocabulary)
            raise ValueError(f"vocabulary index {missing} is not used by the CFG")

        self._require_productive(nonterminals)
        for node in nonterminals:
            node._seal()
        for terminal in terminals:
            terminal._seal()

        self.start = start
        self.nonterminals = tuple(nonterminals)
        self.terminals = terminals
        self.vocabulary_size = vocabulary_size
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

    @staticmethod
    def _require_productive(nonterminals: list[Nonterminal]) -> None:
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
        pending: deque[Nonterminal] = deque()

        for owner in nonterminals:
            for alternative_index, alternative in enumerate(owner.alternatives):
                children = [
                    symbol for symbol in alternative if isinstance(symbol, Nonterminal)
                ]
                if not children and owner not in productive:
                    productive.add(owner)
                    pending.append(owner)
                for child in children:
                    dependents[child].append((owner, alternative_index))

        while pending:
            child = pending.popleft()
            for owner, alternative_index in dependents[child]:
                unresolved_by_alternative[owner][alternative_index] -= 1
                if (
                    unresolved_by_alternative[owner][alternative_index] == 0
                    and owner not in productive
                ):
                    productive.add(owner)
                    pending.append(owner)

        if len(productive) != len(nonterminals):
            missing = next(node for node in nonterminals if node not in productive)
            raise ValueError(
                f"nonterminal {missing.name!r} has no finite terminal derivation"
            )
