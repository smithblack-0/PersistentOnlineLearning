"""Reusable categorical recurrent-state primitive.

``StateCore`` owns the current hidden state and a transition table indexed by
that state and an externally supplied hash code. It does not know tokens,
hashing policy, output decoding, random-state authority, or sequence
orchestration.
"""

import torch

from ..utils import require_positive_int


class StateCore:
    """Categorical hidden state with one next-state lookup per hash code.

    Construction samples one transition system from the active PyTorch random
    stream and starts the process in state zero.

    Args:
        state_count: Number of possible hidden states.
        hash_count: Number of input hash codes accepted by ``transition``.
    """

    def __init__(self, *, state_count: int, hash_count: int) -> None:
        require_positive_int("state_count", state_count)
        require_positive_int("hash_count", hash_count)
        self.state_count = state_count
        self.hash_count = hash_count
        self._transitions = torch.randint(
            state_count,
            (state_count, hash_count),
            dtype=torch.long,
        )
        self._state = 0

    @property
    def state(self) -> int:
        """Current hidden-state index."""

        return self._state

    def transition(self, hash_code: int) -> int:
        """Apply the transition selected by ``hash_code`` and return new state."""

        if type(hash_code) is not int:
            raise TypeError("hash_code must use int")
        if not 0 <= hash_code < self.hash_count:
            raise ValueError("hash_code is outside the transition table")
        self._state = int(self._transitions[self._state, hash_code])
        return self._state

    def state_dict(self) -> dict[str, object]:
        """Return transition rules and current hidden state."""

        return {
            "transitions": self._transitions.clone(),
            "state": self._state,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Validate and restore transition rules and current hidden state."""

        if set(state) != {"transitions", "state"}:
            raise ValueError("invalid state-core state")
        transitions = state["transitions"]
        current_state = state["state"]
        if not isinstance(transitions, torch.Tensor):
            raise TypeError("transitions must be a tensor")
        if transitions.device.type != "cpu":
            raise ValueError("transitions must remain on CPU")
        if transitions.dtype != torch.long:
            raise TypeError("transitions must use torch.long")
        if tuple(transitions.shape) != (self.state_count, self.hash_count):
            raise ValueError("transitions has the wrong shape")
        if bool((transitions < 0).any()) or bool((transitions >= self.state_count).any()):
            raise ValueError("transitions contains an invalid state index")
        if type(current_state) is not int:
            raise TypeError("state must use int")
        if not 0 <= current_state < self.state_count:
            raise ValueError("state contains an invalid state index")
        self._transitions = transitions.clone()
        self._state = current_state
