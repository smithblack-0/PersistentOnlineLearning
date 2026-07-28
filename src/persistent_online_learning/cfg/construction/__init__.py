"""One-shot construction processes that publish fixed ``cfg`` language graphs.

Construction is organized by lifecycle phase rather than by the paper that
inspired parts of the algorithm. Individual modules document which constraints
come directly from Unold et al. and which are project-specific adaptations.
"""

from .generate import generate_unold_cfg

__all__ = ["generate_unold_cfg"]
