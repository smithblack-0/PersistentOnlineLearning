"""Finished graph types for a fixed vocabulary-bearing context-free grammar.

This module defines the language *after* construction: named grammar nodes,
nonterminal productions, terminal vocabularies, and the passive ``CFG`` container.
It deliberately does not know how a grammar was sampled. Construction feasibility,
Unold-specific rule selection, and global correctness checks belong outside these
objects; only local shape contracts and the recursive assembly lifecycle live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


class Node:
    """Shared identity and sealing lifecycle for nodes in one CFG graph.

    Recursive nonterminal edges cannot all be supplied when nodes are created, so
    graph assembly needs a short mutable phase. ``Node`` centralizes only that
    lifecycle: construction code may assemble open nodes, then seals them before
    publication so later consumers see a fixed language.
    """

    __slots__ = ("_name", "_sealed")

    def __init__(self, name: str) -> None:
        """Create one open graph node with a stable human-readable identity."""

        if type(name) is not str or not name:
            raise TypeError("node name must be a nonempty string")
        self._name = name
        self._sealed = False

    @property
    def name(self) -> str:
        """Return the stable name used to inspect and report this graph node."""

        return self._name

    @property
    def sealed(self) -> bool:
        """Report whether recursive graph assembly has ended for this node."""

        return self._sealed

    def _require_open(self) -> None:
        """Reject local graph mutation after construction publishes the language."""

        if self._sealed:
            raise RuntimeError(f"node {self.name!r} is sealed")

    def _seal(self) -> None:
        """Close this node after its owning construction pipeline has finished."""

        self._sealed = True


class TerminalVocabulary(Node):
    """One grammar terminal and the concrete token IDs that may realize it.

    A terminal vocabulary is part of the finished CFG, not a side-table lookup.
    Its token tuple therefore exists at node creation and never changes. Different
    terminal nodes may overlap or even contain identical token sets; the node name
    identifies the grammar choice point, not an exclusive vocabulary category.
    """

    __slots__ = ("_token_ids",)

    def __init__(self, name: str, token_ids: tuple[int, ...]) -> None:
        """Create a complete terminal choice point over distinct concrete IDs."""

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
        """Return the concrete vocabulary alternatives defined by this terminal."""

        return self._token_ids

    def __repr__(self) -> str:
        return f"TerminalVocabulary({self.name!r}, token_ids={self.token_ids!r})"


class Nonterminal(Node):
    """One CFG nonterminal and the production alternatives it owns.

    Production ownership sits on the left-hand-side node so the graph directly
    represents recursion and shared substructure through object references. The
    only mutation allowed is adding locally valid productions during recursive
    assembly; global properties such as reachability are construction guarantees
    and test contracts rather than responsibilities of this node.
    """

    __slots__ = ("_productions",)

    def __init__(self, name: str) -> None:
        """Create an open nonterminal whose productions will be assembled later."""

        super().__init__(name)
        self._productions: list[Production] | tuple[Production, ...] = []

    def add_production(self, *nodes: GrammarNode) -> None:
        """Attach one locally valid RHS while recursive graph assembly is open.

        The method protects the nonterminal's own representation: productions are
        nonempty, contain only grammar nodes, and are unique for this LHS. It does
        not attempt to prove graph-wide properties that belong to the constructor.
        """

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
        """Return this nonterminal's alternatives in construction-selected order."""

        return tuple(self._productions)

    def _seal(self) -> None:
        """Freeze the production collection when the containing CFG is published."""

        if not self.sealed:
            self._productions = tuple(self._productions)
            super()._seal()

    def __repr__(self) -> str:
        return f"Nonterminal({self.name!r})"


GrammarNode: TypeAlias = Nonterminal | TerminalVocabulary
Production: TypeAlias = tuple[GrammarNode, ...]


@dataclass(frozen=True, slots=True)
class CFG:
    """Published fixed language graph consumed by later execution systems.

    ``CFG`` intentionally performs no discovery or validation. A constructor owns
    assembling and sealing the supplied nodes; this container simply makes the
    start node and declared node sets explicit for downstream consumers.

    Attributes:
        start: Nonterminal from which derivations of the language begin.
        nonterminals: All nonterminal nodes declared by the construction process.
        terminal_vocabularies: All terminal choice points declared by the language.
    """

    start: Nonterminal
    nonterminals: tuple[Nonterminal, ...]
    terminal_vocabularies: tuple[TerminalVocabulary, ...]
