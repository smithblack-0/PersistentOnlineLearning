"""Tests for common generator dispatch and exact random CFG construction."""

from collections import Counter

import pytest
import torch

from persistent_online_learning.generator import (
    CFG,
    Nonterminal,
    Production,
    RuleFamily,
    SimpleEpsilonMachine,
    Terminal,
    TokenGenerator,
    UnoldCFGSpecification,
    build_generator,
    generate_unold_cfg,
    register_generator,
    validate_cfg,
    validate_unold_cfg,
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


def test_rule_families_are_classified_from_structure() -> None:
    a = Terminal(0)
    b = Terminal(1)
    left = Nonterminal(0)
    right = Nonterminal(1)

    assert Production(left, (a, b)).family is (
        RuleFamily.PARENTHESIS_WITHOUT_NONTERMINAL
    )
    assert Production(left, (a, right, b)).family is (
        RuleFamily.PARENTHESIS_WITH_NONTERMINAL
    )
    assert Production(left, (a, right)).family is RuleFamily.ITERATION
    assert Production(left, (right, a)).family is RuleFamily.ITERATION
    assert Production(left, (left, right)).family is RuleFamily.BRANCH


def test_paper_example_is_consistent() -> None:
    a, b, c = Terminal(0), Terminal(1), Terminal(2)
    A, B, C = Nonterminal(0), Nonterminal(1), Nonterminal(2)
    grammar = CFG(
        start=C,
        productions=(
            Production(A, (a, b)),
            Production(B, (b, c)),
            Production(C, (A, A)),
            Production(A, (C, c)),
            Production(C, (B, c)),
        ),
    )
    specification = UnoldCFGSpecification(
        parenthesis_without_nonterminal=2,
        parenthesis_with_nonterminal=0,
        iteration_rules=2,
        branch_rules=1,
        max_terminals=3,
        max_nonterminals=3,
    )

    validate_unold_cfg(grammar, specification)


def test_specification_rejects_disconnected_terminal_components() -> None:
    with pytest.raises(ValueError, match="cannot connect"):
        UnoldCFGSpecification(
            parenthesis_without_nonterminal=3,
            parenthesis_with_nonterminal=0,
            iteration_rules=0,
            branch_rules=0,
            max_terminals=1,
            max_nonterminals=3,
        )


def test_specification_rejects_rule_capacity_overflow() -> None:
    with pytest.raises(ValueError, match="branch count"):
        UnoldCFGSpecification(
            parenthesis_without_nonterminal=1,
            parenthesis_with_nonterminal=0,
            iteration_rules=0,
            branch_rules=2,
            max_terminals=1,
            max_nonterminals=1,
        )


@pytest.mark.parametrize(
    "specification",
    (
        UnoldCFGSpecification(1, 0, 0, 0, 1, 1),
        UnoldCFGSpecification(4, 0, 0, 0, 2, 1),
        UnoldCFGSpecification(3, 0, 0, 1, 1, 3),
        UnoldCFGSpecification(3, 2, 3, 2, 3, 6),
        UnoldCFGSpecification(8, 5, 6, 4, 4, 8),
    ),
)
def test_generated_grammars_satisfy_exact_contract_across_random_seeds(
    specification: UnoldCFGSpecification,
) -> None:
    for seed in range(50):
        torch.manual_seed(seed)
        grammar = generate_unold_cfg(specification)
        validate_unold_cfg(grammar, specification)
        counts = Counter(production.family for production in grammar.productions)
        assert len(grammar.productions) == specification.total_rules
        assert counts[RuleFamily.PARENTHESIS_WITHOUT_NONTERMINAL] == (
            specification.parenthesis_without_nonterminal
        )


@pytest.mark.parametrize(
    "specification",
    (
        UnoldCFGSpecification(1, 4, 0, 0, 1, 2),
        UnoldCFGSpecification(1, 0, 8, 0, 1, 2),
        UnoldCFGSpecification(1, 0, 0, 8, 1, 2),
        UnoldCFGSpecification(1, 4, 8, 8, 1, 2),
    ),
)
def test_rule_family_capacity_boundaries_remain_constructible(
    specification: UnoldCFGSpecification,
) -> None:
    for seed in range(20):
        torch.manual_seed(seed)
        validate_unold_cfg(generate_unold_cfg(specification), specification)


def test_construction_uses_the_callers_torch_random_stream() -> None:
    specification = UnoldCFGSpecification(3, 2, 2, 1, 3, 5)
    torch.manual_seed(123)
    first = generate_unold_cfg(specification)
    torch.manual_seed(123)
    second = generate_unold_cfg(specification)
    assert first == second


def test_validation_rejects_an_unreachable_rule() -> None:
    a = Terminal(0)
    A, B = Nonterminal(0), Nonterminal(1)
    grammar = CFG(
        start=A,
        productions=(
            Production(A, (a, a)),
            Production(B, (a, a)),
        ),
    )

    with pytest.raises(ValueError, match="unreachable production"):
        validate_cfg(grammar)
