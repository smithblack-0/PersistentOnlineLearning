"""Standalone recurrent language-model flavors supplied by FLA.

The project accepts FLA's homogeneous DeltaNet and Gated DeltaNet causal language
models directly. This package does not wrap their transformer blocks, recurrent
kernels, cache, generation mixin, configuration, or checkpoint behavior.
Importing these classes also performs FLA's Hugging Face auto-class registration.
"""

from fla.models.delta_net import DeltaNetConfig, DeltaNetForCausalLM
from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

__all__ = [
    "DeltaNetConfig",
    "DeltaNetForCausalLM",
    "GatedDeltaNetConfig",
    "GatedDeltaNetForCausalLM",
]
