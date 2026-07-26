"""Simple epsilon-machine flavor assembled from reusable primitives.

The machine owns only call ordering, composite process state, and default
component construction. Hash reduction, hidden-state transitions, and
state-conditioned decoding are injected dependencies with independent owners.
Process replacement and random-state authority belong to the harness.
"""

from typing import Protocol

from ..base import TokenGenerator, register_generator
from ..primitives import GenerativeDecoder, HashReduction, StateCore


class _Restorable(Protocol):
    def state_dict(self) -> dict[str, object]:
        ...

    def load_state_dict(self, state: dict[str, object]) -> None:
        ...


class _Reducer(_Restorable, Protocol):
    def __call__(self, token: int) -> int:
        ...


class _StateCore(_Restorable, Protocol):
    @property
    def state(self) -> int:
        ...

    def transition(self, hash_code: int) -> int:
        ...


class _Decoder(_Restorable, Protocol):
    def sample(self, state: int) -> int:
        ...


class SimpleEpsilonMachine(TokenGenerator):
    """Orchestrate one hash reducer, state core, and generative decoder.

    Emission precedes transition: the decoder samples from the current hidden
    state, the emitted token is reduced to a hash code, and that hash code
    advances the state used for the next token.

    Args:
        reducer: Owns token-to-hash reduction and its sampled mapping.
        state_core: Owns categorical recurrent state and hash-indexed transitions.
        decoder: Owns state-conditioned token outcomes and sampling.
    """

    def __init__(
        self,
        *,
        reducer: _Reducer,
        state_core: _StateCore,
        decoder: _Decoder,
    ) -> None:
        self.reducer = reducer
        self.state_core = state_core
        self.decoder = decoder

    @classmethod
    def create(
        cls,
        *,
        vocab_size: int,
        state_count: int,
        hash_count: int,
        outcomes_per_state: int,
        epsilon: float,
    ) -> "SimpleEpsilonMachine":
        """Construct one new process from the default primitive composition."""

        return cls(
            reducer=HashReduction(
                vocab_size=vocab_size,
                hash_count=hash_count,
            ),
            state_core=StateCore(
                state_count=state_count,
                hash_count=hash_count,
            ),
            decoder=GenerativeDecoder(
                state_count=state_count,
                vocab_size=vocab_size,
                outcomes_per_state=outcomes_per_state,
                epsilon=epsilon,
            ),
        )

    def step(self) -> int:
        """Emit one token and apply its hash-selected state transition."""

        token = self.decoder.sample(self.state_core.state)
        hash_code = self.reducer(token)
        self.state_core.transition(hash_code)
        return token

    def state_dict(self) -> dict[str, object]:
        """Compose child process state without interpreting child payloads."""

        return {
            "reducer": self.reducer.state_dict(),
            "state_core": self.state_core.state_dict(),
            "decoder": self.decoder.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Delegate restoration to each component owner.

        Construct a compatible machine, load its process state, and discard it
        if validation fails. Checkpoint transaction and random-state restoration
        belong to the harness.
        """

        if set(state) != {"reducer", "state_core", "decoder"}:
            raise ValueError("invalid simple-epsilon machine state")
        reducer_state = state["reducer"]
        state_core_state = state["state_core"]
        decoder_state = state["decoder"]
        if not isinstance(reducer_state, dict):
            raise TypeError("reducer state must be a dictionary")
        if not isinstance(state_core_state, dict):
            raise TypeError("state_core state must be a dictionary")
        if not isinstance(decoder_state, dict):
            raise TypeError("decoder state must be a dictionary")
        self.reducer.load_state_dict(reducer_state)
        self.state_core.load_state_dict(state_core_state)
        self.decoder.load_state_dict(decoder_state)


register_generator("simple_epsilon", SimpleEpsilonMachine.create)
