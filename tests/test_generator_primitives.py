"""Focused behavior and restoration tests for reusable primitives."""

import pytest
import torch

from persistent_online_learning.generator import (
    GenerativeDecoder,
    HashReduction,
    StateCore,
)


def test_hash_reduction_is_balanced_and_exactly_restorable() -> None:
    torch.manual_seed(3)
    reducer = HashReduction(vocab_size=101, hash_count=12)
    state = reducer.state_dict()
    hashes = state["token_hashes"]
    assert isinstance(hashes, torch.Tensor)
    counts = torch.bincount(hashes, minlength=12)
    assert int(counts.max() - counts.min()) <= 1

    restored = HashReduction(vocab_size=101, hash_count=12)
    restored.load_state_dict(state)
    assert [restored(token) for token in range(101)] == [
        reducer(token) for token in range(101)
    ]


def test_hash_reduction_rejects_incompatible_mapping() -> None:
    source = HashReduction(vocab_size=16, hash_count=4)
    target = HashReduction(vocab_size=17, hash_count=4)
    with pytest.raises(ValueError, match="wrong shape"):
        target.load_state_dict(source.state_dict())


def test_state_core_applies_hash_selected_transition() -> None:
    core = StateCore(state_count=3, hash_count=2)
    core.load_state_dict(
        {
            "transitions": torch.tensor(
                [
                    [1, 2],
                    [2, 0],
                    [0, 1],
                ],
                dtype=torch.long,
            ),
            "state": 0,
        }
    )
    assert core.transition(1) == 2
    assert core.transition(1) == 1
    assert core.transition(0) == 2


def test_state_core_restore_preserves_current_state() -> None:
    torch.manual_seed(8)
    core = StateCore(state_count=5, hash_count=4)
    core.transition(2)
    state = core.state_dict()

    restored = StateCore(state_count=5, hash_count=4)
    restored.load_state_dict(state)
    assert restored.state == core.state
    assert restored.transition(1) == core.transition(1)


def test_decoder_distribution_matches_state_support_and_epsilon() -> None:
    decoder = GenerativeDecoder(
        state_count=2,
        vocab_size=6,
        outcomes_per_state=2,
        epsilon=0.2,
    )
    decoder.load_state_dict(
        {
            "outcomes": torch.tensor([[1, 3], [2, 5]], dtype=torch.long),
            "probabilities": torch.tensor(
                [[0.25, 0.75], [0.6, 0.4]],
                dtype=torch.float32,
            ),
        }
    )
    distribution = decoder.distribution(0)
    baseline = 0.2 / 6
    assert torch.isclose(distribution.sum(), torch.tensor(1.0))
    assert torch.isclose(distribution[0], torch.tensor(baseline))
    assert torch.isclose(distribution[1], torch.tensor(baseline + 0.8 * 0.25))
    assert torch.isclose(distribution[3], torch.tensor(baseline + 0.8 * 0.75))


def test_decoder_construction_is_deterministic_under_harness_seed() -> None:
    torch.manual_seed(10)
    left = GenerativeDecoder(
        state_count=4,
        vocab_size=32,
        outcomes_per_state=5,
        epsilon=0.0,
    )
    torch.manual_seed(10)
    right = GenerativeDecoder(
        state_count=4,
        vocab_size=32,
        outcomes_per_state=5,
        epsilon=0.0,
    )
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert torch.equal(left_state["outcomes"], right_state["outcomes"])
    assert torch.equal(left_state["probabilities"], right_state["probabilities"])
    assert not torch.equal(left.distribution(0), left.distribution(1))


def test_decoder_rejects_duplicate_outcomes() -> None:
    decoder = GenerativeDecoder(
        state_count=2,
        vocab_size=8,
        outcomes_per_state=2,
        epsilon=0.0,
    )
    with pytest.raises(ValueError, match="disjoint"):
        decoder.load_state_dict(
            {
                "outcomes": torch.tensor([[1, 1], [2, 3]], dtype=torch.long),
                "probabilities": torch.full((2, 2), 0.5),
            }
        )


def test_decoder_assigns_distinct_structured_supports_to_states() -> None:
    torch.manual_seed(22)
    decoder = GenerativeDecoder(
        state_count=4,
        vocab_size=16,
        outcomes_per_state=4,
        epsilon=0.0,
    )
    outcomes = decoder.state_dict()["outcomes"]
    supports = {tuple(sorted(row.tolist())) for row in outcomes}
    assert len(supports) == 4


def test_primitives_are_valid_immediately_after_construction() -> None:
    reducer = HashReduction(vocab_size=8, hash_count=2)
    core = StateCore(state_count=2, hash_count=2)
    decoder = GenerativeDecoder(
        state_count=2,
        vocab_size=8,
        outcomes_per_state=2,
        epsilon=0.0,
    )
    assert 0 <= reducer(0) < 2
    assert 0 <= core.state < 2
    assert torch.isclose(decoder.distribution(0).sum(), torch.tensor(1.0))


def test_decoder_restore_rejects_cross_state_support_overlap() -> None:
    decoder = GenerativeDecoder(
        state_count=2,
        vocab_size=8,
        outcomes_per_state=2,
        epsilon=0.0,
    )
    with pytest.raises(ValueError, match="disjoint"):
        decoder.load_state_dict(
            {
                "outcomes": torch.tensor([[1, 2], [2, 3]], dtype=torch.long),
                "probabilities": torch.full((2, 2), 0.5),
            }
        )


def test_epsilon_rejects_boolean_values() -> None:
    with pytest.raises(TypeError, match="finite number"):
        GenerativeDecoder(
            state_count=2,
            vocab_size=8,
            outcomes_per_state=2,
            epsilon=True,
        )
