"""Tests for the common generator contract and factory registry.

The repository CI still invokes this historical test entrypoint explicitly. CFG
construction contracts are implemented separately in ``cfg_generation_contracts``
and imported here only so the existing workflow executes them from both the source
tree and installed wheel. The workflow itself is intentionally unchanged.
"""

import pytest

from cfg_generation_contracts import (
    test_cfg_is_a_passive_container_and_does_not_finalize_its_nodes,
    test_feasible_nonterminal_counts_remain_randomized_within_configured_limit,
    test_generated_cfg_satisfies_complete_independent_graph_contract,
    test_generation_is_deterministic_under_the_callers_torch_seed,
    test_independent_productivity_oracle_accepts_recursion_and_rejects_dead_cycles,
    test_nonterminal_accepts_only_unique_node_productions_until_sealed,
    test_productivity_oracle_is_iterative_for_deep_graphs,
    test_representative_large_vocabulary_is_directly_and_completely_reachable,
    test_small_feasible_matrix_is_independently_audited_across_random_seeds,
    test_spawn_config_separates_local_and_cross_request_feasibility,
    test_terminal_vocabulary_builder_guarantees_coverage_and_allows_overlap,
    test_terminal_vocabulary_is_complete_and_locally_valid_at_construction,
    test_terminal_vocabularies_may_overlap_or_contain_identical_token_sets,
    test_vocabulary_density_does_not_shift_syntax_randomness,
)
from persistent_online_learning.generator import (
    SimpleEpsilonMachine,
    TokenGenerator,
    build_generator,
    register_generator,
)


def _specification() -> dict[str, object]:
    """Return the ordinary registry request shared by generator framework tests."""

    return {
        "type": "simple_epsilon",
        "vocab_size": 64,
        "state_count": 5,
        "hash_count": 8,
        "outcomes_per_state": 4,
        "epsilon": 0.05,
    }


def test_registry_dispatches_without_mutating_specification() -> None:
    """Registry construction returns the requested flavor without editing input."""

    specification = _specification()
    original = dict(specification)
    generator = build_generator(specification)
    assert isinstance(generator, SimpleEpsilonMachine)
    assert specification == original


def test_registry_reports_missing_and_unknown_type() -> None:
    """Registry construction rejects absent and unregistered flavor names."""

    with pytest.raises(ValueError, match="requires a type"):
        build_generator({"vocab_size": 64})
    with pytest.raises(ValueError, match="unknown generator type"):
        build_generator({"type": "missing"})


def test_registry_accepts_an_external_factory() -> None:
    """An externally registered factory participates in ordinary dispatch."""

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
    """The base generator produces the requested one-dimensional token count."""

    generator = build_generator(_specification())
    tokens = generator.generate(12)
    assert tokens.shape == (12,)
