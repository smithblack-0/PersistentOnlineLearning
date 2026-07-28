"""Construct complete terminal vocabularies before CFG syntax is generated.

This phase is project-specific rather than part of Unold et al.'s CFG algorithm.
The paper assumes an already-existing terminal alphabet. Here each terminal symbol
also defines the concrete vocabulary IDs it may realize, so construction first
builds those complete ``TerminalVocabulary`` nodes and then gives the resulting
alphabet to the syntax constructor.

Global concrete-vocabulary coverage is established constructively: every token ID
is assigned once before overlap fills remaining per-terminal capacity. The syntax
phase never needs to inspect concrete token membership.
"""

from __future__ import annotations

import torch

from ..config import TerminalVocabularyConfig
from ..grammar import TerminalVocabulary


def build_terminal_vocabularies(
    config: TerminalVocabularyConfig,
    generator: torch.Generator,
) -> tuple[TerminalVocabulary, ...]:
    """Create the terminal alphabet consumed by later CFG construction.

    The function exists because terminal membership is a separate random decision
    from syntax topology. It guarantees the terminal-config contract up front:
    every node contains exactly ``tokens_per_terminal`` distinct IDs, every ID in
    the configured universe appears globally, and overlap between nodes remains
    legal. Returned nodes are complete but still unsealed so the final CFG
    publisher can close the shared graph lifecycle atomically.
    """

    assignments: list[list[int]] = [[] for _ in range(config.terminal_count)]
    membership: list[set[int]] = [set() for _ in range(config.terminal_count)]
    terminal_order = torch.randperm(config.terminal_count, generator=generator).tolist()
    token_order = torch.randperm(config.vocabulary_size, generator=generator).tolist()

    # First use every concrete ID exactly once. Round-robin assignment keeps the
    # unavoidable first-use load balanced before overlap is introduced.
    for position, token_id in enumerate(token_order):
        terminal_index = terminal_order[position % config.terminal_count]
        assignments[terminal_index].append(token_id)
        membership[terminal_index].add(token_id)

    # Fill each terminal independently. Repeated IDs across terminals are valid;
    # only duplicate membership inside one terminal would erase a real choice.
    for terminal_index in range(config.terminal_count):
        if len(assignments[terminal_index]) == config.tokens_per_terminal:
            continue
        token_order = torch.randperm(
            config.vocabulary_size, generator=generator
        ).tolist()
        for token_id in token_order:
            if token_id in membership[terminal_index]:
                continue
            assignments[terminal_index].append(token_id)
            membership[terminal_index].add(token_id)
            if len(assignments[terminal_index]) == config.tokens_per_terminal:
                break

    return tuple(
        TerminalVocabulary(f"T{index}", tuple(token_ids))
        for index, token_ids in enumerate(assignments)
    )
