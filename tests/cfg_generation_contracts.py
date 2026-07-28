"""Contract tests for fixed vocabulary-bearing CFG construction."""

from __future__ import annotations

import itertools

import pytest
import torch

from persistent_online_learning.generator import (
    CFG,
    CFGSpawnConfig,
    GrammarConfig,
    Nonterminal,
    TerminalVocabulary,
    TerminalVocabularyConfig,
    build_terminal_vocabularies,
    generate_unold_cfg,
)
from persistent_online_learning.generator.cfg_audit import audit_constructed_cfg


def _config(
    plain: int = 4,
    parenthesis: int = 3,
    iteration: int = 2,
    branch: int = 2,
    max_nonterminals: int = 7,
    terminal_count: int = 4,
    vocabulary_size: int = 12,
    tokens_per_terminal: int = 4,
) -> CFGSpawnConfig:
    return CFGSpawnConfig(
        grammar=GrammarConfig(
            terminal_pair_rules=plain,
            parenthesis_rules=parenthesis,
            iteration_rules=iteration,
            branch_rules=branch,
            max_nonterminals=max_nonterminals,
        ),
        terminal_vocabularies=TerminalVocabularyConfig(
            terminal_count=terminal_count,
            vocabulary_size=vocabulary_size,
            tokens_per_terminal=tokens_per_terminal,
        ),
    )


def _family_counts(grammar: CFG) -> tuple[int, int, int, int]:
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
                raise AssertionError(f"unexpected production: {production!r}")
    return plain, parenthesis, iteration, branch


def _signature(grammar: CFG) -> tuple[object, ...]:
    return (
        tuple(
            (terminal.name, terminal.token_ids)
            for terminal in grammar.terminal_vocabularies
        ),
        tuple(
            (
                node.name,
                tuple(
                    tuple(child.name for child in production)
                    for production in node.productions
                ),
            )
            for node in grammar.nonterminals
        ),
    )


def _syntax_signature(grammar: CFG) -> tuple[object, ...]:
    return tuple(
        (
            node.name,
            tuple(
                tuple(child.name for child in production)
                for production in node.productions
            ),
        )
        for node in grammar.nonterminals
    )


def test_terminal_vocabulary_is_complete_and_locally_valid_at_construction() -> None:
    """A terminal owns an immutable nonempty set of distinct concrete token IDs."""

    terminal = TerminalVocabulary("word", (2, 4, 7))
    assert terminal.token_ids == (2, 4, 7)
    with pytest.raises(ValueError, match="repeat"):
        TerminalVocabulary("duplicate", (1, 1))
    with pytest.raises(ValueError, match="nonnegative"):
        TerminalVocabulary("negative", (-1,))


def test_terminal_vocabularies_may_overlap_or_contain_identical_token_sets() -> None:
    """Token membership is nonexclusive across independently named terminals."""

    left = TerminalVocabulary("left", (0, 1, 2))
    right = TerminalVocabulary("right", (0, 1, 2))
    root = Nonterminal("root")
    root.add_production(left)
    root.add_production(right)
    audit_constructed_cfg(
        start=root,
        nonterminals=[root],
        terminal_vocabularies=(left, right),
        vocabulary_size=3,
    )
    assert left.token_ids == right.token_ids


def test_nonterminal_accepts_only_unique_node_productions_until_sealed() -> None:
    """Nonterminals enforce local production shape and reject later mutation."""

    terminal = TerminalVocabulary("word", (0,))
    node = Nonterminal("phrase")
    with pytest.raises(ValueError, match="must not be empty"):
        node.add_production()
    with pytest.raises(TypeError, match="Nonterminal or TerminalVocabulary"):
        node.add_production(object())
    node.add_production(terminal)
    with pytest.raises(ValueError, match="already owns"):
        node.add_production(terminal)
    node._seal()
    with pytest.raises(RuntimeError, match="sealed"):
        node.add_production(terminal)


def test_cfg_is_a_passive_container_and_does_not_finalize_its_nodes() -> None:
    """Constructing CFG stores supplied graph objects without validation or sealing."""

    terminal = TerminalVocabulary("word", (0,))
    root = Nonterminal("root")
    grammar = CFG(root, (root,), (terminal,))
    assert grammar.start is root
    assert not root.sealed
    assert not terminal.sealed


def test_terminal_vocabulary_builder_guarantees_coverage_and_allows_overlap() -> None:
    """Every concrete ID is used while terminal token sets remain nonexclusive."""

    torch.manual_seed(8)
    generator = torch.Generator().manual_seed(8)
    terminals = build_terminal_vocabularies(
        TerminalVocabularyConfig(
            terminal_count=4, vocabulary_size=7, tokens_per_terminal=4
        ),
        generator,
    )
    assert all(len(terminal.token_ids) == 4 for terminal in terminals)
    assert set().union(*(set(terminal.token_ids) for terminal in terminals)) == set(
        range(7)
    )
    memberships = sum(len(terminal.token_ids) for terminal in terminals)
    assert memberships > 7


def test_spawn_config_separates_local_and_cross_product_feasibility() -> None:
    """Independent configs validate locally; composition rejects incompatibility."""

    with pytest.raises(ValueError, match="enough slots"):
        TerminalVocabularyConfig(
            terminal_count=2, vocabulary_size=5, tokens_per_terminal=2
        )
    with pytest.raises(ValueError, match="terminal positions"):
        _config(
            plain=1,
            parenthesis=0,
            iteration=0,
            branch=0,
            max_nonterminals=1,
            terminal_count=3,
            vocabulary_size=3,
            tokens_per_terminal=1,
        )


def test_generated_cfg_contains_exact_rule_counts_and_direct_terminal_vocabulary(
) -> None:
    """The graph matches the request and reaches IDs through terminal nodes."""

    config = _config(plain=5, parenthesis=4, iteration=3, branch=2, max_nonterminals=8)
    torch.manual_seed(12)
    grammar = generate_unold_cfg(config)
    assert _family_counts(grammar) == (5, 4, 3, 2)
    assert all(node.sealed for node in grammar.nonterminals)
    assert all(terminal.sealed for terminal in grammar.terminal_vocabularies)
    used_terminals = {
        child
        for node in grammar.nonterminals
        for production in node.productions
        for child in production
        if isinstance(child, TerminalVocabulary)
    }
    assert used_terminals == set(grammar.terminal_vocabularies)
    assert all(terminal.token_ids for terminal in used_terminals)


def test_generation_is_deterministic_under_the_callers_torch_seed() -> None:
    """The same config and active PyTorch seed reproduce syntax and vocabulary."""

    config = _config()
    torch.manual_seed(91)
    left = generate_unold_cfg(config)
    torch.manual_seed(91)
    right = generate_unold_cfg(config)
    assert _signature(left) == _signature(right)


def test_vocabulary_density_does_not_shift_syntax_randomness() -> None:
    """Concrete-token sampling does not alter syntax for the same terminal alphabet."""

    grammar_config = GrammarConfig(
        terminal_pair_rules=5,
        parenthesis_rules=4,
        iteration_rules=3,
        branch_rules=2,
        max_nonterminals=8,
    )
    sparse = CFGSpawnConfig(
        grammar=grammar_config,
        terminal_vocabularies=TerminalVocabularyConfig(4, 8, 2),
    )
    dense = CFGSpawnConfig(
        grammar=grammar_config,
        terminal_vocabularies=TerminalVocabularyConfig(4, 20, 5),
    )
    torch.manual_seed(44)
    sparse_grammar = generate_unold_cfg(sparse)
    torch.manual_seed(44)
    dense_grammar = generate_unold_cfg(dense)
    assert _syntax_signature(sparse_grammar) == _syntax_signature(dense_grammar)


def test_feasible_nonterminal_counts_remain_randomized_within_configured_limit(
) -> None:
    """The planner preserves variation instead of always choosing minimum nodes."""

    config = _config(
        plain=5,
        parenthesis=4,
        iteration=3,
        branch=2,
        max_nonterminals=8,
        terminal_count=4,
        vocabulary_size=12,
        tokens_per_terminal=4,
    )
    counts = set()
    for seed in range(20):
        torch.manual_seed(seed)
        counts.add(len(generate_unold_cfg(config).nonterminals))
    assert len(counts) > 1
    assert max(counts) <= config.grammar.max_nonterminals


def test_audit_accepts_productive_recursion_and_rejects_unproductive_cycles() -> None:
    """Finalization distinguishes recursive productive graphs from dead cycles."""

    terminal = TerminalVocabulary("word", (0,))
    left = Nonterminal("left")
    right = Nonterminal("right")
    left.add_production(right)
    right.add_production(left)
    right.add_production(terminal)
    audit_constructed_cfg(
        start=left,
        nonterminals=[left, right],
        terminal_vocabularies=(terminal,),
        vocabulary_size=1,
    )

    bad_left = Nonterminal("bad_left")
    bad_right = Nonterminal("bad_right")
    bad_left.add_production(bad_right)
    bad_right.add_production(bad_left)
    with pytest.raises(RuntimeError, match="finite terminal derivation"):
        audit_constructed_cfg(
            start=bad_left,
            nonterminals=[bad_left, bad_right],
            terminal_vocabularies=(),
            vocabulary_size=0,
        )


def test_failed_audit_does_not_seal_or_repair_construction_state() -> None:
    """An invalid graph fails before publication and leaves construction state open."""

    terminal = TerminalVocabulary("word", (0,))
    root = Nonterminal("root")
    unreachable = Nonterminal("unreachable")
    root.add_production(terminal)
    unreachable.add_production(terminal)
    with pytest.raises(RuntimeError, match="unreachable"):
        audit_constructed_cfg(
            start=root,
            nonterminals=[root, unreachable],
            terminal_vocabularies=(terminal,),
            vocabulary_size=1,
        )
    assert not root.sealed
    assert not unreachable.sealed
    assert not terminal.sealed


def test_small_feasible_matrix_constructs_without_runtime_rescue_failures() -> None:
    """A deterministic matrix of accepted requests succeeds across two random seeds."""

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
            plain,
            parenthesis,
            iteration,
            branch,
            maximum,
            terminals,
            vocabulary,
            per_terminal,
        ) = values
        try:
            config = _config(
                plain=plain,
                parenthesis=parenthesis,
                iteration=iteration,
                branch=branch,
                max_nonterminals=maximum,
                terminal_count=terminals,
                vocabulary_size=vocabulary,
                tokens_per_terminal=per_terminal,
            )
        except ValueError:
            continue
        accepted += 1
        expected = (plain, parenthesis, iteration, branch)
        for seed in (0, 1):
            torch.manual_seed(seed)
            grammar = generate_unold_cfg(config)
            assert _family_counts(grammar) == expected
    assert accepted >= 100


def test_representative_large_vocabulary_is_directly_and_completely_reachable() -> None:
    """Two hundred terminal vocabularies collectively expose all 10,000 token IDs."""

    config = _config(
        plain=100,
        parenthesis=0,
        iteration=0,
        branch=0,
        max_nonterminals=1,
        terminal_count=200,
        vocabulary_size=10_000,
        tokens_per_terminal=200,
    )
    torch.manual_seed(7)
    grammar = generate_unold_cfg(config)
    assert len(grammar.terminal_vocabularies) == 200
    assert set().union(
        *(set(terminal.token_ids) for terminal in grammar.terminal_vocabularies)
    ) == set(range(10_000))


def test_productivity_audit_is_iterative_for_deep_graphs() -> None:
    """A 2,000-node productive chain finalizes without recursive Python traversal."""

    terminal = TerminalVocabulary("word", (0,))
    nodes = [Nonterminal(f"N{index}") for index in range(2_000)]
    nodes[-1].add_production(terminal)
    for index in range(len(nodes) - 1):
        nodes[index].add_production(nodes[index + 1])
    audit_constructed_cfg(
        start=nodes[0],
        nonterminals=nodes,
        terminal_vocabularies=(terminal,),
        vocabulary_size=1,
    )
