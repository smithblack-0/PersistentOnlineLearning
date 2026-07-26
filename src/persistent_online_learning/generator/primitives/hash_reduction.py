"""Reusable token-to-hash reduction primitive.

``HashReduction`` owns one balanced mapping from vocabulary tokens to shared
hash cores. It does not know hidden states, transition rules, output
probabilities, random-state authority, or sequence orchestration.
"""

import torch

from ..utils import require_positive_int


class HashReduction:
    """Reduce each scalar token to one balanced categorical hash code.

    Construction samples one complete mapping from the active PyTorch random
    stream. Creating another instance creates another reduction system.

    Args:
        vocab_size: Number of valid input token IDs.
        hash_count: Number of shared hash codes. Every code receives either
            ``floor(vocab_size / hash_count)`` or ``ceil(...)`` tokens.
    """

    def __init__(self, *, vocab_size: int, hash_count: int) -> None:
        require_positive_int("vocab_size", vocab_size)
        require_positive_int("hash_count", hash_count)
        if vocab_size < hash_count:
            raise ValueError("vocab_size must be at least hash_count")
        self.vocab_size = vocab_size
        self.hash_count = hash_count
        base = torch.arange(vocab_size, dtype=torch.long).remainder(hash_count)
        permutation = torch.randperm(vocab_size)
        self._token_hashes = base[permutation]

    def __call__(self, token: int) -> int:
        """Return the hash code assigned to one token ID."""

        if type(token) is not int:
            raise TypeError("token must use int")
        if not 0 <= token < self.vocab_size:
            raise ValueError("token is outside the vocabulary")
        return int(self._token_hashes[token])

    def state_dict(self) -> dict[str, object]:
        """Return the sampled token-to-hash mapping."""

        return {"token_hashes": self._token_hashes.clone()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Validate and restore a token-to-hash mapping."""

        if set(state) != {"token_hashes"}:
            raise ValueError("invalid hash-reduction state")
        token_hashes = state["token_hashes"]
        if not isinstance(token_hashes, torch.Tensor):
            raise TypeError("token_hashes must be a tensor")
        if token_hashes.device.type != "cpu":
            raise ValueError("token_hashes must remain on CPU")
        if token_hashes.dtype != torch.long:
            raise TypeError("token_hashes must use torch.long")
        if tuple(token_hashes.shape) != (self.vocab_size,):
            raise ValueError("token_hashes has the wrong shape")
        if bool((token_hashes < 0).any()) or bool((token_hashes >= self.hash_count).any()):
            raise ValueError("token_hashes contains an invalid hash code")
        counts = torch.bincount(token_hashes, minlength=self.hash_count)
        if int(counts.max() - counts.min()) > 1:
            raise ValueError("token_hashes violates the balanced mapping contract")
        self._token_hashes = token_hashes.clone()
