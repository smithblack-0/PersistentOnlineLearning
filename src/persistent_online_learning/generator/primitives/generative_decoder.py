"""Reusable state-conditioned categorical decoder.

``GenerativeDecoder`` owns the token outcomes, probabilities, and epsilon
sampling behavior associated with each hidden state. It does not own
hidden-state transitions, token reduction, process replacement, random-state
authority, or sequence ordering.
"""

import torch

from ..utils import require_positive_int, require_probability


class GenerativeDecoder:
    """Decode a categorical hidden state into token probabilities and samples.

    Construction samples one decoder from the active PyTorch random stream.
    Each state receives ``outcomes_per_state`` distinct structured outcomes.
    Sampling follows those state-specific probabilities with probability
    ``1 - epsilon`` and is uniform over the vocabulary with probability
    ``epsilon``.

    Args:
        state_count: Number of hidden states accepted by the decoder.
        vocab_size: Number of token outcomes.
        outcomes_per_state: Distinct structured outcomes assigned to each state.
        epsilon: Uniform-random sampling probability.
    """

    def __init__(
        self,
        *,
        state_count: int,
        vocab_size: int,
        outcomes_per_state: int,
        epsilon: float,
    ) -> None:
        require_positive_int("state_count", state_count)
        require_positive_int("vocab_size", vocab_size)
        require_positive_int("outcomes_per_state", outcomes_per_state)
        if state_count * outcomes_per_state > vocab_size:
            raise ValueError(
                "state_count * outcomes_per_state cannot exceed vocab_size"
            )
        self.state_count = state_count
        self.vocab_size = vocab_size
        self.outcomes_per_state = outcomes_per_state
        self.epsilon = require_probability("epsilon", epsilon)
        permutation = torch.randperm(vocab_size)
        support_count = state_count * outcomes_per_state
        self._outcomes = permutation[:support_count].reshape(
            state_count,
            outcomes_per_state,
        )
        weights = torch.rand(
            (state_count, outcomes_per_state),
            dtype=torch.float32,
        ).clamp_min(torch.finfo(torch.float32).tiny)
        self._probabilities = weights / weights.sum(dim=-1, keepdim=True)

    def distribution(self, state: int) -> torch.Tensor:
        """Return the exact vocabulary distribution for one hidden state."""

        self._validate_state(state)
        result = torch.full(
            (self.vocab_size,),
            self.epsilon / self.vocab_size,
            dtype=torch.float32,
        )
        outcomes = self._outcomes[state]
        result[outcomes] += (1.0 - self.epsilon) * self._probabilities[state]
        return result

    def sample(self, state: int) -> int:
        """Sample one token from the exact epsilon-mixture distribution."""

        self._validate_state(state)
        if self.epsilon and float(torch.rand(())) < self.epsilon:
            return int(torch.randint(self.vocab_size, ()))
        support_index = int(torch.multinomial(self._probabilities[state], 1))
        return int(self._outcomes[state, support_index])

    def state_dict(self) -> dict[str, object]:
        """Return state-conditioned outcomes and probabilities."""

        return {
            "outcomes": self._outcomes.clone(),
            "probabilities": self._probabilities.clone(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Validate and restore state-conditioned outcomes and probabilities."""

        if set(state) != {"outcomes", "probabilities"}:
            raise ValueError("invalid generative-decoder state")
        outcomes = state["outcomes"]
        probabilities = state["probabilities"]
        if not isinstance(outcomes, torch.Tensor) or not isinstance(
            probabilities,
            torch.Tensor,
        ):
            raise TypeError("decoder state values must be tensors")
        if outcomes.device.type != "cpu" or probabilities.device.type != "cpu":
            raise ValueError("decoder state tensors must remain on CPU")
        expected_shape = (self.state_count, self.outcomes_per_state)
        if tuple(outcomes.shape) != expected_shape:
            raise ValueError("outcomes has the wrong shape")
        if tuple(probabilities.shape) != expected_shape:
            raise ValueError("probabilities has the wrong shape")
        if outcomes.dtype != torch.long:
            raise TypeError("outcomes must use torch.long")
        if probabilities.dtype != torch.float32:
            raise TypeError("probabilities must use torch.float32")
        if bool((outcomes < 0).any()) or bool((outcomes >= self.vocab_size).any()):
            raise ValueError("outcomes contains an invalid token index")
        if torch.unique(outcomes).numel() != outcomes.numel():
            raise ValueError("outcomes must be disjoint across states")
        if not bool(torch.isfinite(probabilities).all()) or bool(
            (probabilities <= 0).any()
        ):
            raise ValueError("probabilities must be finite and positive")
        sums = probabilities.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
            raise ValueError("probabilities must sum to one per state")
        self._outcomes = outcomes.clone()
        self._probabilities = probabilities.clone()

    def _validate_state(self, state: int) -> None:
        if type(state) is not int:
            raise TypeError("state must use int")
        if not 0 <= state < self.state_count:
            raise ValueError("state is outside the decoder")
