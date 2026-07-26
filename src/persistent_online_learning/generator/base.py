"""Common generator lifecycle and dictionary-driven flavor construction.

This module owns only the token-generator contract and the registry that maps a
plain ``type`` name to a flavor factory. Concrete mechanisms, scientific
parameters, batching, telemetry, checkpoints, random-state authority, and
storage belong elsewhere.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import torch


class TokenGenerator(ABC):
    """Stateful source of discrete tokens with restorable process state."""

    @abstractmethod
    def step(self) -> int:
        """Advance the process by one token and return that token."""

    def generate(self, length: int) -> torch.Tensor:
        """Generate a one-dimensional sequence using repeated ``step`` calls."""

        if type(length) is not int:
            raise TypeError("length must use int")
        if length <= 0:
            raise ValueError("length must be positive")
        tokens = torch.empty(length, dtype=torch.long)
        for index in range(length):
            tokens[index] = self.step()
        return tokens

    @abstractmethod
    def state_dict(self) -> dict[str, object]:
        """Return hidden rules and recurrent state owned by this process."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore process state into a newly constructed compatible generator."""


GeneratorFactory = Callable[..., TokenGenerator]
_GENERATOR_FACTORIES: dict[str, GeneratorFactory] = {}


def register_generator(name: str, factory: GeneratorFactory) -> None:
    """Register one flavor factory under a nonempty construction name."""

    if not isinstance(name, str) or not name:
        raise ValueError("generator names must be nonempty strings")
    if not callable(factory):
        raise TypeError("generator factory must be callable")
    if name in _GENERATOR_FACTORIES:
        raise ValueError(f"generator already registered: {name}")
    _GENERATOR_FACTORIES[name] = factory


def build_generator(specification: Mapping[str, object]) -> TokenGenerator:
    """Dispatch a plain mapping to a registered flavor factory.

    ``type`` selects the flavor. Every other item is forwarded unchanged as a
    keyword argument, leaving validation and meaning with the selected flavor.
    """

    values = dict(specification)
    if "type" not in values:
        raise ValueError("generator specification requires a type")
    name = values.pop("type")
    if not isinstance(name, str):
        raise TypeError("generator type must be a string")
    try:
        factory = _GENERATOR_FACTORIES[name]
    except KeyError as error:
        raise ValueError(f"unknown generator type: {name}") from error
    return factory(**values)
