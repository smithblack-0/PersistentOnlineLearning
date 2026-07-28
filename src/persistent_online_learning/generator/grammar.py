"""Passive graph objects for a fixed vocabulary-bearing context-free grammar.

A finished grammar is directly traversable from nonterminal productions to
``TerminalVocabulary`` nodes and from those nodes to concrete token IDs.  The
objects know only their local shape and construction lifecycle.  Reachability,
productivity, feasibility, and global vocabulary coverage belong to the
construction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


class Node:
    """One named graph node that may be sealed after recursive assembly."""

    __slots__ = ("_name", "_sealed")

    def __init__(self, name: str) -> None:
        if type(name) is not str or not name:
            raise TypeError("node name must be a nonempty string")
        self._name = name
        self._sealed = False

    @property
    def name(self) -> str:
        """Human-readable identity used when inspecting the grammar."""

        return self._name

    @property
    def sealed(self) -> bool:
        """Whether recursive construction has ended for this node."""

        return self._sealed

    def _require_open(self) -> None:
        if self._sealed:
            raise RuntimeError(f"node {self.name!r} is sealed")

    def _seal(self) -> None:
        """End construction without performing graph-wide validation."""

        self._sealed = True


class TerminalVocabulary(Node):
    """One terminal symbol and the concrete token IDs that may realize it.

    Different terminal vocabularies may overlap or even contain identical token
    sets.  A terminal vocabulary is complete when created; only its shared node
    lifecycle remains open until the containing CFG is finalized.
    """

    __slots__ = ("_token_ids",)

    def __init__(self, name: str, token_ids: tuple[int, ...]) -> None:
        super().__init__(name)
        if type(token_ids) is not tuple or not token_ids:
            raise TypeError("terminal token_ids must be a nonempty tuple")
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise ValueError("terminal token IDs must be nonnegative integers")
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("one terminal vocabulary must not repeat a token ID")
        self._token_ids = token_ids

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Concrete vocabulary choices represented by this terminal."""

        return self._token_ids

    def __repr__(self) -> str:
        return f"TerminalVocabulary({self.name!r}, token_ids={self.token_ids!r})"


class Nonterminal(Node):
    """One nonterminal and its ordered production alternatives."""

    __slots__ = ("_productions",)

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._productions: list[Production] | tuple[Production, ...] = []

    def add_production(self, *nodes: GrammarNode) -> None:
        """Add one locally valid production while the graph is being assembled."""

        self._require_open()
        if not nodes:
            raise ValueError("a production must not be empty")
        if not all(
            isinstance(node, (Nonterminal, TerminalVocabulary)) for node in nodes
        ):
            raise TypeError(
                "a production may contain only Nonterminal or TerminalVocabulary nodes"
            )
        production = tuple(nodes)
        if production in self._productions:
            raise ValueError(f"nonterminal {self.name!r} already owns that production")
        assert isinstance(self._productions, list)
        self._productions.append(production)

    @property
    def productions(self) -> tuple[Production, ...]:
        """Productions in the order chosen by the construction algorithm."""

        return tuple(self._productions)

    def _seal(self) -> None:
        if not self.sealed:
            self._productions = tuple(self._productions)
            super()._seal()

    def __repr__(self) -> str:
        return f"Nonterminal({self.name!r})"


GrammarNode: TypeAlias = Nonterminal | TerminalVocabulary
Production: TypeAlias = tuple[GrammarNode, ...]


@dataclass(frozen=True, slots=True)
class CFG:
    """A sealed, directly traversable context-free grammar graph."""

    start: Nonterminal
    nonterminals: tuple[Nonterminal, ...]
    terminal_vocabularies: tuple[TerminalVocabulary, ...]
