"""Construction-boundary audits for a completed CFG candidate.

These checks are deliberately absent from the runtime ``CFG`` object.  A builder
uses them once, before sealing and publishing a graph.
"""

from __future__ import annotations

from collections import deque

from .grammar import Nonterminal, TerminalVocabulary


def audit_constructed_cfg(
    *,
    start: Nonterminal,
    nonterminals: list[Nonterminal],
    terminal_vocabularies: tuple[TerminalVocabulary, ...],
    vocabulary_size: int,
) -> None:
    """Require one declared, reachable, productive graph with full vocabulary use."""

    _require_distinct_nodes(nonterminals, terminal_vocabularies)
    ordered_nonterminals, used_terminals = _walk_graph(
        start,
        nonterminals,
        terminal_vocabularies,
    )
    if len(ordered_nonterminals) != len(nonterminals):
        reachable = set(ordered_nonterminals)
        missing = next(node for node in nonterminals if node not in reachable)
        raise RuntimeError(f"constructed nonterminal {missing.name!r} is unreachable")
    if len(used_terminals) != len(terminal_vocabularies):
        missing = next(
            terminal
            for terminal in terminal_vocabularies
            if terminal not in used_terminals
        )
        raise RuntimeError(
            f"constructed TerminalVocabulary {missing.name!r} is unused"
        )
    _require_productive(ordered_nonterminals)
    _require_vocabulary_coverage(
        terminal_vocabularies=terminal_vocabularies,
        vocabulary_size=vocabulary_size,
    )


def _require_distinct_nodes(
    nonterminals: list[Nonterminal],
    terminal_vocabularies: tuple[TerminalVocabulary, ...],
) -> None:
    """Require unique node identities and human-readable names."""

    nodes = [*nonterminals, *terminal_vocabularies]
    if len(set(nodes)) != len(nodes):
        raise RuntimeError("construction declared the same graph node more than once")
    names: set[str] = set()
    for node in nodes:
        if node.name in names:
            raise RuntimeError(f"construction reused node name {node.name!r}")
        names.add(node.name)


def _walk_graph(
    start: Nonterminal,
    nonterminals: list[Nonterminal],
    terminal_vocabularies: tuple[TerminalVocabulary, ...],
) -> tuple[list[Nonterminal], set[TerminalVocabulary]]:
    """Traverse declared edges and reject references outside the candidate graph."""

    declared_nonterminals = set(nonterminals)
    declared_terminals = set(terminal_vocabularies)
    if start not in declared_nonterminals:
        raise RuntimeError("CFG start is not a declared nonterminal")

    ordered: list[Nonterminal] = []
    seen: set[Nonterminal] = {start}
    used_terminals: set[TerminalVocabulary] = set()
    pending: deque[Nonterminal] = deque([start])
    while pending:
        node = pending.popleft()
        ordered.append(node)
        if not node.productions:
            raise RuntimeError(
                f"constructed nonterminal {node.name!r} has no productions"
            )
        for production in node.productions:
            for child in production:
                if isinstance(child, TerminalVocabulary):
                    if child not in declared_terminals:
                        raise RuntimeError(
                            f"production references undeclared terminal {child.name!r}"
                        )
                    used_terminals.add(child)
                else:
                    if child not in declared_nonterminals:
                        raise RuntimeError(
                            "production references undeclared nonterminal "
                            f"{child.name!r}"
                        )
                    if child not in seen:
                        seen.add(child)
                        pending.append(child)
    return ordered, used_terminals


def _require_productive(nonterminals: list[Nonterminal]) -> None:
    """Propagate productivity iteratively from terminal-only productions."""

    unresolved: dict[Nonterminal, list[int]] = {
        node: [
            sum(isinstance(child, Nonterminal) for child in production)
            for production in node.productions
        ]
        for node in nonterminals
    }
    dependents: dict[Nonterminal, list[tuple[Nonterminal, int]]] = {
        node: [] for node in nonterminals
    }
    productive: set[Nonterminal] = set()
    pending: deque[Nonterminal] = deque()

    for owner in nonterminals:
        for index, production in enumerate(owner.productions):
            children = [
                child for child in production if isinstance(child, Nonterminal)
            ]
            if not children and owner not in productive:
                productive.add(owner)
                pending.append(owner)
            for child in children:
                dependents[child].append((owner, index))

    while pending:
        child = pending.popleft()
        for owner, production_index in dependents[child]:
            unresolved[owner][production_index] -= 1
            if unresolved[owner][production_index] == 0 and owner not in productive:
                productive.add(owner)
                pending.append(owner)

    if len(productive) != len(nonterminals):
        missing = next(node for node in nonterminals if node not in productive)
        raise RuntimeError(
            "constructed nonterminal "
            f"{missing.name!r} has no finite terminal derivation"
        )


def _require_vocabulary_coverage(
    *,
    terminal_vocabularies: tuple[TerminalVocabulary, ...],
    vocabulary_size: int,
) -> None:
    """Require every concrete ID exactly within the configured token universe."""

    used: set[int] = set()
    for terminal in terminal_vocabularies:
        for token_id in terminal.token_ids:
            if token_id >= vocabulary_size:
                raise RuntimeError(
                    f"terminal token ID {token_id} is outside vocabulary size "
                    f"{vocabulary_size}"
                )
            used.add(token_id)
    expected = set(range(vocabulary_size))
    if used != expected:
        missing = min(expected - used)
        raise RuntimeError(f"constructed terminals do not use token ID {missing}")
