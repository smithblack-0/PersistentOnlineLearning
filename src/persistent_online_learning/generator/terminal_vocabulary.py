"""Construct complete ``TerminalVocabulary`` nodes before syntax generation."""

from __future__ import annotations

import torch

from .cfg_config import TerminalVocabularyConfig
from .grammar import TerminalVocabulary


def build_terminal_vocabularies(
    config: TerminalVocabularyConfig,
    generator: torch.Generator,
) -> tuple[TerminalVocabulary, ...]:
    """Build terminal nodes with balanced first-use coverage and allowed overlap.

    Every concrete token ID is assigned once before any terminal is filled with
    overlapping samples.  This makes global coverage constructive rather than a
    repair performed after syntax generation.
    """

    if not isinstance(config, TerminalVocabularyConfig):
        raise TypeError("config must be TerminalVocabularyConfig")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be torch.Generator")

    assignments: list[list[int]] = [[] for _ in range(config.terminal_count)]
    membership: list[set[int]] = [set() for _ in range(config.terminal_count)]
    terminal_order = torch.randperm(config.terminal_count, generator=generator).tolist()
    token_order = torch.randperm(config.vocabulary_size, generator=generator).tolist()

    for position, token_id in enumerate(token_order):
        terminal_index = terminal_order[position % config.terminal_count]
        assignments[terminal_index].append(token_id)
        membership[terminal_index].add(token_id)

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
