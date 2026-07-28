"""Passive data structures for a fixed lexicalized context-free grammar.

The syntax is a directed graph whose nodes are ordinary Python objects. Terminal
nodes are leaves. Nonterminal nodes own ordered production alternatives and may
reference any terminal or nonterminal node, including themselves. Construction
algorithms are responsible for graph-wide properties such as reachability and
productivity before they seal the nodes and publish these containers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


class Node:
    """One named graph node with a shared construction-to-runtime lifecycle."""

    __slots__ = ("name", "_sealed")

    def __init__(self, name: str) -> None:
        if type(name) is not str or not name:
            raise TypeError("node name must be a nonempty string")
        self.name = name
        self._sealed = False

    @property
    def sealed(self) -> bool:
        """Whether construction has finished for this node."""

        return self._sealed

    def _require_open(self) -> None:
        if self._sealed:
            raise RuntimeError(f"node {self.name!r} is sealed")

    def seal(self) -> None:
        """End construction for this node.

        Subclasses may finalize their local storage before delegating here.
        Graph-wide validation does not belong to this method.
        """

        self._sealed = True


class Terminal(Node):
    """One abstract terminal symbol in the grammar graph."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"Terminal({self.name!r})"


class Nonterminal(Node):
    """One nonterminal symbol and its locally owned production alternatives."""

    __slots__ = ("_alternatives",)

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._alternatives: list[Production] | tuple[Production, ...] = []

    def add_alternative(self, *symbols: GrammarSymbol) -> None:
        """Add one nonempty production while this node is under construction."""

        self._require_open()
        if not symbols:
            raise ValueError("a production alternative must not be empty")
        if not all(isinstance(symbol, (Terminal, Nonterminal)) for symbol in symbols):
            raise TypeError(
                "a production alternative may contain only Terminal or Nonterminal nodes"
            )

        production = tuple(symbols)
        if production in self._alternatives:
            raise ValueError(f"nonterminal {self.name!r} already owns that production")

        assert isinstance(self._alternatives, list)
        self._alternatives.append(production)

    @property
    def alternatives(self) -> tuple[Production, ...]:
        """Return the productions in construction order."""

        return tuple(self._alternatives)

    def seal(self) -> None:
        if not self.sealed:
            self._alternatives = tuple(self._alternatives)
            super().seal()

    def __repr__(self) -> str:
        return f"Nonterminal({self.name!r})"


GrammarSymbol: TypeAlias = Terminal | Nonterminal
Production: TypeAlias = tuple[GrammarSymbol, ...]


@dataclass(frozen=True, slots=True)
class CFG:
    """The finished syntax graph.

    This object stores the nodes selected by the constructor. It does not discover,
    validate, seal, or otherwise compute over them.
    """

    start: Nonterminal
    nonterminals: tuple[Nonterminal, ...]
    terminals: tuple[Terminal, ...]


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """The concrete token-index space available to the lexical realization."""

    size: int

    def __post_init__(self) -> None:
        if type(self.size) is not int:
            raise TypeError("vocabulary size must use int")
        if self.size <= 0:
            raise ValueError("vocabulary size must be positive")


@dataclass(frozen=True, slots=True)
class LexiconEntry:
    """The concrete token IDs that may realize one terminal node."""

    terminal: Terminal
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.terminal, Terminal):
            raise TypeError("lexicon entry terminal must be a Terminal")
        if type(self.token_ids) is not tuple or not self.token_ids:
            raise TypeError("lexicon token_ids must be a nonempty tuple")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.token_ids):
            raise ValueError("lexicon token IDs must be nonnegative integers")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError("a lexicon entry must not contain duplicate token IDs")


@dataclass(frozen=True, slots=True)
class Lexicon:
    """The fixed lexical realization attached to the grammar's terminal nodes."""

    vocabulary: Vocabulary
    entries: tuple[LexiconEntry, ...]


@dataclass(frozen=True, slots=True)
class LexicalizedCFG:
    """The fixed syntax graph and its concrete lexical realization."""

    grammar: CFG
    lexicon: Lexicon
