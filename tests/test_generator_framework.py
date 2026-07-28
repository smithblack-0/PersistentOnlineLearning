"""Tests for the common generator contract and factory registry."""

import pytest

from persistent_online_learning.generator import (
    SimpleEpsilonMachine,
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
