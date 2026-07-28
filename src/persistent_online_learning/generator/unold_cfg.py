"""Construct and publish a fixed CFG using the Unold productive-first method.

Reference
---------
O. Unold, A. Kaczmarek, and Ł. Culer, "Iterative method of generating
artificial context-free grammars," arXiv:1911.05801, 2019, sections 2.3-2.5.

Pipeline
--------

1. Construct complete ``TerminalVocabulary`` nodes.
2. Build productive nonterminal topology from a feasible count plan.
3. Assign terminal nodes to topology drafts.
4. Audit declarations, reachability, productivity, terminal use, and vocabulary.
5. Seal and publish the passive ``CFG``.

The paper treats terminal and nonterminal counts as maxima and decides symbol
creation while inserting rules. This implementation receives an exact terminal
alphabet and randomly selects a feasible total nonterminal count before rule
insertion. Existing-LHS extensions are restricted to the reachable component;
hanging productive roots enter that component through RHS edges. Those deliberate
restrictions make the construction lifecycle and remaining edge budget explicit
while preserving the paper's productive-first and final-consistency guarantees.
"""

from __future__ import annotations

import torch

from .cfg_audit import audit_constructed_cfg
from .cfg_config import CFGSpawnConfig
from .grammar import CFG
from .terminal_vocabulary import build_terminal_vocabularies
from .unold_terminal_assignment import publish_terminal_productions
from .unold_topology import build_unold_topology


def generate_unold_cfg(config: CFGSpawnConfig) -> CFG:
    """Construct, audit, seal, and return one fixed vocabulary-bearing CFG."""

    if not isinstance(config, CFGSpawnConfig):
        raise TypeError("config must be CFGSpawnConfig")

    terminal_generator, syntax_generator = _split_random_stream()
    terminals = build_terminal_vocabularies(
        config.terminal_vocabularies,
        terminal_generator,
    )
    topology = build_unold_topology(
        config,
        len(terminals),
        syntax_generator,
    )
    publish_terminal_productions(
        topology.drafts,
        terminals,
        syntax_generator,
    )

    nonterminals = list(topology.nonterminals)
    audit_constructed_cfg(
        start=topology.start,
        nonterminals=nonterminals,
        terminal_vocabularies=terminals,
        vocabulary_size=config.terminal_vocabularies.vocabulary_size,
    )
    for node in (*nonterminals, *terminals):
        node._seal()
    return CFG(
        start=topology.start,
        nonterminals=topology.nonterminals,
        terminal_vocabularies=terminals,
    )


def _split_random_stream() -> tuple[torch.Generator, torch.Generator]:
    """Derive independent vocabulary and syntax streams from caller-owned state."""

    seeds = torch.randint(0, 2**63 - 1, (2,), dtype=torch.int64).tolist()
    terminal_generator = torch.Generator()
    terminal_generator.manual_seed(seeds[0])
    syntax_generator = torch.Generator()
    syntax_generator.manual_seed(seeds[1])
    return terminal_generator, syntax_generator
