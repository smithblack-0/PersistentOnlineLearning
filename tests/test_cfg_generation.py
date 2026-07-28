"""Tests for CFG node ownership and Unold construction finalization."""

import itertools

import pytest
import torch

from persistent_online_learning.generator import (
    CFG,
    LexiconEntry,
    LexiconParameters,
    Nonterminal,
    Terminal,
    UnoldCFGParameters,
    generate_unold_cfg,
)
from persistent_online_learning.generator.unold_cfg import _finalize_language


def _family_counts(language) -> tuple[int, int, int, int]:
    terminal_pairs = parenthesis = iteration = branches = 0
    for node in language.grammar.nonterminals:
        for production in node.alternatives:
            positions = tuple(
                isinstance(symbol, Nonterminal) for symbol in production
            )
            if len(production) == 2 and positions == (False, False):
                terminal_pairs += 1
            elif len(production) == 3 and positions == (False, True, False):
                parenthesis += 1
            elif len(production) == 2 and positions in (
                (False, True),
                (True, False),
            ):
                iteration += 1
            elif len(production) == 2 and positions == (True, True):
                branches += 1
            else:
                raise AssertionError(production)
    return terminal_pairs, parenthesis, iteration, branches


def _signature(language) -> tuple[object, ...]:
    grammar = language.grammar
    syntax = tuple(
        (
            node.name,
            tuple(
                tuple(symbol.name for symbol in production)
                for production in node.alternatives
            ),
        )
        for node in grammar.nonterminals
    )
    lexicon = tuple(
        (entry.terminal.name, entry.token_ids) for entry in language.lexicon.entries
    )
    return syntax, lexicon


def _finalize(
    start: Nonterminal,
    nonterminals: list[Nonterminal],
    terminals: list[Terminal],
    entries: tuple[LexiconEntry, ...],
    vocabulary_size: int,
):
    return _finalize_language(
        start=start,
        declared_nonterminals=nonterminals,
        declared_terminals=terminals,
        entries=entries,
        vocabulary_size=vocabulary_size,
    )


def test_specialized_nodes_own_only_local_shape_and_sealing() -> None:
    terminal = Terminal("word")
    node = Nonterminal("phrase")

    with pytest.raises(ValueError, match="must not be empty"):
        node.add_alternative()
    with pytest.raises(TypeError, match="Terminal or Nonterminal"):
        node.add_alternative(object())

    node.add_alternative(terminal)
    with pytest.raises(ValueError, match="already owns"):
        node.add_alternative(terminal)

    node._seal()
    terminal._seal()
    assert node.sealed and terminal.sealed
    with pytest.raises(RuntimeError, match="sealed"):
        node.add_alternative(terminal)


def test_cfg_is_a_passive_container_for_already_finalized_nodes() -> None:
    terminal = Terminal("word")
    root = Nonterminal("root")
    root.add_alternative(terminal)
    grammar = CFG(root, (root,), (terminal,))

    assert grammar.start is root
    assert grammar.nonterminals == (root,)
    assert grammar.terminals == (terminal,)
    assert not root.sealed
    assert not terminal.sealed


def test_finalization_accepts_productive_recursion_and_seals_the_graph() -> None:
    terminal = Terminal("word")
    left = Nonterminal("left")
    right = Nonterminal("right")
    left.add_alternative(right)
    right.add_alternative(left)
    right.add_alternative(terminal)

    language = _finalize(
        left,
        [left, right],
        [terminal],
        (LexiconEntry(terminal, (0,)),),
        1,
    )

    assert language.grammar.nonterminals == (left, right)
    assert language.lexicon.entries[0].terminal is terminal
    assert left.sealed and right.sealed and terminal.sealed


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unreachable", "unreachable"),
        ("undeclared_reference", "undeclared nonterminal"),
        ("unproductive", "finite terminal derivation"),
        ("incomplete_vocabulary", "token ID 1"),
        ("duplicate_declaration", "same node more than once"),
    ],
)
def test_finalization_rejects_invalid_construction_output_without_sealing(
    mutation: str,
    message: str,
) -> None:
    terminal = Terminal("word")
    root = Nonterminal("root")
    root.add_alternative(terminal)
    nonterminals = [root]
    terminals = [terminal]
    entries = (LexiconEntry(terminal, (0,)),)
    vocabulary_size = 1

    if mutation == "unreachable":
        other = Nonterminal("other")
        other.add_alternative(terminal)
        nonterminals.append(other)
    elif mutation == "undeclared_reference":
        undeclared = Nonterminal("undeclared")
        root.add_alternative(undeclared)
    elif mutation == "unproductive":
        left = Nonterminal("left")
        right = Nonterminal("right")
        root.add_alternative(left)
        left.add_alternative(right)
        right.add_alternative(left)
        nonterminals.extend([left, right])
    elif mutation == "incomplete_vocabulary":
        vocabulary_size = 2
    elif mutation == "duplicate_declaration":
        nonterminals.append(root)

    with pytest.raises(RuntimeError, match=message):
        _finalize(
            root,
            nonterminals,
            terminals,
            entries,
            vocabulary_size,
        )

    assert not root.sealed
    assert not terminal.sealed


def test_exact_unold_request_generates_requested_syntax_and_lexicon() -> None:
    parameters = UnoldCFGParameters(
        terminal_pair_rules=5,
        parenthesis_rules=4,
        iteration_rules=3,
        branch_rules=2,
        max_nonterminals=8,
        lexicon=LexiconParameters(5, 10, 4),
    )
    torch.manual_seed(12)
    language = generate_unold_cfg(parameters)

    assert _family_counts(language) == (5, 4, 3, 2)
    assert len(language.grammar.terminals) == 5
    assert all(node.sealed for node in language.grammar.nonterminals)
    assert all(node.sealed for node in language.grammar.terminals)
    assert all(len(entry.token_ids) == 4 for entry in language.lexicon.entries)
    assert set().union(
        *(set(entry.token_ids) for entry in language.lexicon.entries)
    ) == set(range(10))


def test_unold_construction_is_deterministic_under_callers_torch_seed() -> None:
    parameters = UnoldCFGParameters(
        4,
        3,
        2,
        2,
        7,
        LexiconParameters(4, 12, 4),
    )
    torch.manual_seed(91)
    left = generate_unold_cfg(parameters)
    torch.manual_seed(91)
    right = generate_unold_cfg(parameters)
    assert _signature(left) == _signature(right)


def test_small_feasible_parameter_matrix_constructs_for_multiple_seeds() -> None:
    accepted = 0
    for values in itertools.product(
        range(1, 4),
        range(0, 3),
        range(0, 3),
        range(0, 2),
        range(1, 5),
        range(1, 4),
        range(1, 6),
        range(1, 4),
    ):
        (
            pair,
            parenthesis,
            iteration,
            branch,
            max_nonterminals,
            categories,
            vocabulary_size,
            tokens_per_category,
        ) = values
        try:
            parameters = UnoldCFGParameters(
                pair,
                parenthesis,
                iteration,
                branch,
                max_nonterminals,
                LexiconParameters(
                    categories,
                    vocabulary_size,
                    tokens_per_category,
                ),
            )
        except ValueError:
            continue

        accepted += 1
        expected = (pair, parenthesis, iteration, branch)
        for seed in (0, 1):
            torch.manual_seed(seed)
            language = generate_unold_cfg(parameters)
            assert _family_counts(language) == expected
            assert len(language.grammar.terminals) == categories

    assert accepted >= 100


def test_representative_large_vocabulary_is_fully_used() -> None:
    parameters = UnoldCFGParameters(
        100,
        0,
        0,
        0,
        1,
        LexiconParameters(200, 10_000, 200),
    )
    torch.manual_seed(7)
    language = generate_unold_cfg(parameters)

    assert len(language.grammar.terminals) == 200
    assert all(len(entry.token_ids) == 200 for entry in language.lexicon.entries)
    assert set().union(
        *(set(entry.token_ids) for entry in language.lexicon.entries)
    ) == set(range(10_000))


def test_large_productivity_check_is_iterative() -> None:
    terminal = Terminal("word")
    nodes = [Nonterminal(f"N{index}") for index in range(2_000)]
    nodes[-1].add_alternative(terminal)
    for index in range(len(nodes) - 1):
        nodes[index].add_alternative(nodes[index + 1])

    language = _finalize(
        nodes[0],
        nodes,
        [terminal],
        (LexiconEntry(terminal, (0,)),),
        1,
    )
    assert len(language.grammar.nonterminals) == 2_000
