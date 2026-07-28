"""Sequence CFG construction phases and publish one fixed language graph.

The syntax process is based on the productive-first method from:

O. Unold, A. Kaczmarek, and Ł. Culer, "Iterative method of generating
artificial context-free grammars," arXiv:1911.05801, 2019.

This project extends that process with vocabulary-bearing terminal nodes, exact use
of the supplied terminal alphabet, a preselected feasible nonterminal-count plan,
and separate random streams for concrete vocabulary membership and syntax. The
phase modules document those adaptations where their invariants are implemented.

No final production-time graph audit is performed. Each phase establishes the
contract needed by the next phase; independent test oracles re-check reachability,
productivity, terminal use, vocabulary coverage, and rule uniqueness across the
finished result.
"""

from __future__ import annotations

import torch

from ..config import CFGSpawnConfig
from ..grammar import CFG
from .planning import choose_construction_plan
from .terminal_assignment import publish_terminal_productions
from .terminal_vocabularies import build_terminal_vocabularies
from .topology import build_unold_topology


def generate_unold_cfg(config: CFGSpawnConfig) -> CFG:
    """Construct and publish one fixed vocabulary-bearing CFG.

    This function is intentionally orchestration-only. It separates the caller's
    random authority into independent terminal-membership and syntax streams,
    delegates each construction calculation to its phase owner, seals the complete
    graph after terminal assignment succeeds, and returns the passive ``CFG``.
    """

    if not isinstance(config, CFGSpawnConfig):
        raise TypeError("config must be CFGSpawnConfig")

    terminal_generator, syntax_generator = _split_random_stream()
    terminals = build_terminal_vocabularies(
        config.terminal_vocabularies,
        terminal_generator,
    )
    plan = choose_construction_plan(config, syntax_generator)
    topology = build_unold_topology(
        config.grammar,
        plan,
        len(terminals),
        syntax_generator,
    )
    publish_terminal_productions(
        topology.drafts,
        terminals,
        syntax_generator,
    )

    for node in (*topology.nonterminals, *terminals):
        node._seal()
    return CFG(
        start=topology.start,
        nonterminals=topology.nonterminals,
        terminal_vocabularies=terminals,
    )


def _split_random_stream() -> tuple[torch.Generator, torch.Generator]:
    """Decouple syntax randomness from concrete vocabulary-density sampling.

    The caller's active PyTorch RNG remains the single external random authority.
    Two derived seeds create one-shot child streams so changing
    ``tokens_per_terminal`` cannot shift syntax choices when the terminal alphabet
    size and grammar configuration are unchanged.
    """

    seeds = torch.randint(0, 2**63 - 1, (2,), dtype=torch.int64).tolist()
    terminal_generator = torch.Generator()
    terminal_generator.manual_seed(seeds[0])
    syntax_generator = torch.Generator()
    syntax_generator.manual_seed(seeds[1])
    return terminal_generator, syntax_generator
