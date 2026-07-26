"""Composition, recurrence, construction, and continuation tests."""

import pytest
import torch

from persistent_online_learning.generator import SimpleEpsilonMachine


def _machine(seed: int = 11, **overrides: object) -> SimpleEpsilonMachine:
    arguments: dict[str, object] = {
        "vocab_size": 64,
        "state_count": 6,
        "hash_count": 8,
        "outcomes_per_state": 4,
        "epsilon": 0.03,
    }
    arguments.update(overrides)
    torch.manual_seed(seed)
    return SimpleEpsilonMachine.create(**arguments)


def test_machine_is_only_orchestration_under_dependency_injection() -> None:
    events: list[tuple[str, int]] = []

    class Reducer:
        def __call__(self, token: int) -> int:
            events.append(("reduce", token))
            return 2

        def state_dict(self) -> dict[str, object]:
            return {}

        def load_state_dict(self, state: dict[str, object]) -> None:
            return None

    class Core:
        state = 3

        def transition(self, hash_code: int) -> int:
            events.append(("transition", hash_code))
            self.state = 4
            return self.state

        def state_dict(self) -> dict[str, object]:
            return {}

        def load_state_dict(self, state: dict[str, object]) -> None:
            return None

    class Decoder:
        def sample(self, state: int) -> int:
            events.append(("decode", state))
            return 7

        def state_dict(self) -> dict[str, object]:
            return {}

        def load_state_dict(self, state: dict[str, object]) -> None:
            return None

    machine = SimpleEpsilonMachine(
        reducer=Reducer(),
        state_core=Core(),
        decoder=Decoder(),
    )
    assert machine.step() == 7
    assert events == [("decode", 3), ("reduce", 7), ("transition", 2)]


def test_factory_constructs_valid_default_components() -> None:
    machine = _machine()
    assert machine.reducer.vocab_size == 64
    assert machine.state_core.state_count == 6
    assert machine.decoder.outcomes_per_state == 4
    assert 0 <= machine.step() < 64


def test_same_harness_seed_produces_same_process_and_sequence() -> None:
    left = _machine(seed=17)
    left_tokens = left.generate(80)
    right = _machine(seed=17)
    right_tokens = right.generate(80)
    assert torch.equal(left_tokens, right_tokens)


def test_new_construction_replaces_the_process() -> None:
    first = _machine(seed=3)
    first_state = first.state_dict()
    second = _machine(seed=4)
    second_state = second.state_dict()
    assert not torch.equal(
        first_state["reducer"]["token_hashes"],
        second_state["reducer"]["token_hashes"],
    )


def test_state_and_harness_rng_restore_exact_continuation() -> None:
    machine = _machine(seed=21)
    machine.generate(19)
    process_state = machine.state_dict()
    random_state = torch.get_rng_state()
    expected = machine.generate(41)

    restored = _machine(seed=999)
    restored.load_state_dict(process_state)
    torch.set_rng_state(random_state)
    assert torch.equal(restored.generate(41), expected)


def test_state_dict_delegates_to_component_owners() -> None:
    state = _machine().state_dict()
    assert set(state) == {"reducer", "state_core", "decoder"}
    assert set(state["reducer"]) == {"token_hashes"}
    assert set(state["state_core"]) == {"transitions", "state"}
    assert set(state["decoder"]) == {"outcomes", "probabilities"}


def test_incompatible_component_state_fails_at_owning_boundary() -> None:
    source = _machine()
    target = _machine(vocab_size=65)
    with pytest.raises(ValueError, match="token_hashes has the wrong shape"):
        target.load_state_dict(source.state_dict())


def test_long_scalar_continuation_stays_in_range() -> None:
    machine = _machine(seed=31)
    tokens = machine.generate(10_000)
    assert int(tokens.min()) >= 0
    assert int(tokens.max()) < 64
