"""Tests for common generator contracts and grammar structure."""

import pytest

from persistent_online_learning.generator import (
    BranchRule,
    Grammar,
    IterationRule,
    ParenthesisRule,
    SimpleEpsilonMachine,
    TerminalPairRule,
    TokenGenerator,
    build_generator,
    register_generator,
)


def _specification() -> dict[str, object]:
    return {
        "type": "simple_epsilon",
        "vocab_size": 64,
        "state_count": 5,
        "hash_count": 8,
        "outcomes_per_state": 4,
        "epsilon": 0.05,
    }


def _recursive_grammar() -> Grammar:
    return Grammar(
        root=0,
        nodes=(
            (BranchRule(1, 2),),
            (
                TerminalPairRule(0, 1),
                IterationRule(2, 1, terminal_first=False),
            ),
            (ParenthesisRule(3, 1, 4),),
        ),
    )


def test_registry_dispatches_without_mutating_specification() -> None:
    specification = _specification()
    original = dict(specification)
    generator = build_generator(specification)
    assert isinstance(generator, SimpleEpsilonMachine)
    assert specification == original


def test_registry_reports_missing_and_unknown_type() -> None:
    with pytest.raises(ValueError, match="requires a type"):
        build_generator({"vocab_size": 64})
    with pytest.raises(ValueError, match="unknown generator type"):
        build_generator({"type": "missing"})


def test_registry_accepts_an_external_factory() -> None:
    class ConstantGenerator(TokenGenerator):
        def step(self) -> int:
            return 3

        def state_dict(self) -> dict[str, object]:
            return {}

        def load_state_dict(self, state: dict[str, object]) -> None:
            return None

    def build_constant() -> TokenGenerator:
        return ConstantGenerator()

    register_generator("test_constant", build_constant)
    generator = build_generator({"type": "test_constant"})
    assert generator.step() == 3


def test_default_generate_uses_repeated_transition() -> None:
    generator = build_generator(_specification())
    tokens = generator.generate(12)
    assert tokens.shape == (12,)


def test_grammar_owns_node_alternatives_and_derived_relationships() -> None:
    grammar = _recursive_grammar()

    assert grammar.alternatives(1) == (
        TerminalPairRule(0, 1),
        IterationRule(2, 1, terminal_first=False),
    )
    assert grammar.children(0) == frozenset({1, 2})
    assert grammar.children(1) == frozenset({1})
    assert grammar.terminal_categories == frozenset({0, 1, 2, 3, 4})
    assert grammar.rule_count == 4


def test_grammar_accepts_mutual_recursion_with_a_finite_derivation() -> None:
    grammar = Grammar(
        root=0,
        nodes=(
            (BranchRule(1, 1),),
            (ParenthesisRule(0, 0, 1), TerminalPairRule(2, 3)),
        ),
    )
    assert grammar.children(1) == frozenset({0})


def test_grammar_rejects_a_child_outside_its_node_table() -> None:
    with pytest.raises(ValueError, match="references child 1"):
        Grammar(root=0, nodes=((ParenthesisRule(0, 1, 2),),))


def test_grammar_rejects_an_empty_node() -> None:
    with pytest.raises(TypeError, match="node 0 alternatives"):
        Grammar(root=0, nodes=((),))


def test_grammar_rejects_duplicate_alternatives_owned_by_one_node() -> None:
    rule = TerminalPairRule(0, 1)
    with pytest.raises(ValueError, match="duplicate alternatives"):
        Grammar(root=0, nodes=((rule, rule),))


def test_grammar_rejects_an_unreachable_node() -> None:
    with pytest.raises(ValueError, match="node 1 is unreachable"):
        Grammar(
            root=0,
            nodes=(
                (TerminalPairRule(0, 1),),
                (TerminalPairRule(2, 3),),
            ),
        )


def test_grammar_rejects_an_unproductive_cycle() -> None:
    with pytest.raises(ValueError, match="no finite terminal derivation"):
        Grammar(
            root=0,
            nodes=(
                (IterationRule(0, 1, terminal_first=True),),
                (IterationRule(1, 0, terminal_first=False),),
            ),
        )


def test_rule_indices_reject_booleans_and_negative_values() -> None:
    with pytest.raises(TypeError, match="left terminal must use int"):
        TerminalPairRule(True, 1)
    with pytest.raises(ValueError, match="left child must be nonnegative"):
        BranchRule(-1, 0)
