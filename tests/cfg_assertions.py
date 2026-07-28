"""Independent test oracles for CFG-construction guarantees.

These checks intentionally duplicate guarantees made by production construction.
They are test infrastructure, not reusable production utilities: their purpose is
to catch a constructor that silently stops establishing reachability, productivity,
terminal use, concrete-vocabulary coverage, or production uniqueness.
"""

from __future__ import annotations

from collections import deque

from persistent_online_learning.cfg import (
    CFG,
    CFGSpawnConfig,
    Nonterminal,
    TerminalVocabulary,
)


def assert_generated_cfg_contract(grammar: CFG, config: CFGSpawnConfig) -> None:
    """Independently verify the complete fixed-language contract of one result."""

    assert grammar.nonterminals
    assert grammar.terminal_vocabularies
    assert grammar.start in set(grammar.nonterminals)
    assert len(grammar.terminal_vocabularies) == config.terminal_vocabularies.terminal_count
    assert len(grammar.nonterminals) <= config.grammar.max_nonterminals
    assert all(node.sealed for node in grammar.nonterminals)
    assert all(terminal.sealed for terminal in grammar.terminal_vocabularies)

    _assert_distinct_declarations(grammar)
    used_terminals = _assert_reachable_declared_graph(grammar)
    assert used_terminals == set(grammar.terminal_vocabularies)
    _assert_productive(grammar.nonterminals)
    _assert_terminal_vocabulary_contract(grammar, config)
    _assert_unique_productions(grammar)
    assert rule_family_counts(grammar) == (
        config.grammar.terminal_pair_rules,
        config.grammar.parenthesis_rules,
        config.grammar.iteration_rules,
        config.grammar.branch_rules,
    )


def rule_family_counts(grammar: CFG) -> tuple[int, int, int, int]:
    """Classify final productions independently from construction rule drafts."""

    plain = parenthesis = iteration = branch = 0
    for node in grammar.nonterminals:
        for production in node.productions:
            nonterminal_positions = tuple(
                isinstance(child, Nonterminal) for child in production
            )
            if len(production) == 2 and nonterminal_positions == (False, False):
                plain += 1
            elif len(production) == 3 and nonterminal_positions == (
                False,
                True,
                False,
            ):
                parenthesis += 1
            elif len(production) == 2 and nonterminal_positions in (
                (False, True),
                (True, False),
            ):
                iteration += 1
            elif len(production) == 2 and nonterminal_positions == (True, True):
                branch += 1
            else:
                raise AssertionError(f"unexpected production shape: {production!r}")
    return plain, parenthesis, iteration, branch


def assert_productive(nonterminals: tuple[Nonterminal, ...]) -> None:
    """Expose the iterative productivity oracle for focused adversarial tests."""

    _assert_productive(nonterminals)


def _assert_distinct_declarations(grammar: CFG) -> None:
    """Reject accidental node aliasing or duplicate inspection names in a result."""

    nodes = [*grammar.nonterminals, *grammar.terminal_vocabularies]
    assert len(set(nodes)) == len(nodes)
    assert len({node.name for node in nodes}) == len(nodes)


def _assert_reachable_declared_graph(grammar: CFG) -> set[TerminalVocabulary]:
    """Walk final production edges and require exactly the declared graph."""

    declared_nonterminals = set(grammar.nonterminals)
    declared_terminals = set(grammar.terminal_vocabularies)
    seen: set[Nonterminal] = {grammar.start}
    used_terminals: set[TerminalVocabulary] = set()
    pending: deque[Nonterminal] = deque([grammar.start])

    while pending:
        node = pending.popleft()
        assert node.productions
        for production in node.productions:
            assert production
            for child in production:
                if isinstance(child, TerminalVocabulary):
                    assert child in declared_terminals
                    used_terminals.add(child)
                else:
                    assert isinstance(child, Nonterminal)
                    assert child in declared_nonterminals
                    if child not in seen:
                        seen.add(child)
                        pending.append(child)

    assert seen == declared_nonterminals
    return used_terminals


def _assert_productive(nonterminals: tuple[Nonterminal, ...]) -> None:
    """Propagate finite terminal derivability without recursive Python traversal."""

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

    assert productive == set(nonterminals)


def _assert_terminal_vocabulary_contract(
    grammar: CFG,
    config: CFGSpawnConfig,
) -> None:
    """Verify per-terminal membership and project-wide concrete-token coverage."""

    expected = set(range(config.terminal_vocabularies.vocabulary_size))
    used: set[int] = set()
    for terminal in grammar.terminal_vocabularies:
        assert len(terminal.token_ids) == config.terminal_vocabularies.tokens_per_terminal
        assert len(set(terminal.token_ids)) == len(terminal.token_ids)
        assert all(token_id in expected for token_id in terminal.token_ids)
        used.update(terminal.token_ids)
    assert used == expected


def _assert_unique_productions(grammar: CFG) -> None:
    """Verify the paper's rule-uniqueness guarantee on final graph productions."""

    for node in grammar.nonterminals:
        assert len(node.productions) == len(set(node.productions))
