"""Assign complete terminal nodes to an Unold syntax topology.

Topology construction has already fixed nonterminal connectivity. This phase
chooses terminal nodes for each open terminal position, guarantees that every
supplied ``TerminalVocabulary`` is used, and preserves exact production
uniqueness within each finite topology bucket.
"""

from __future__ import annotations

from itertools import product
from typing import TypeVar

import torch

from .grammar import TerminalVocabulary
from .unold_rules import RuleDraft, bucket_key, realize_production, terminal_arity

_RANDOM_ATTEMPTS = 128
_Value = TypeVar("_Value")


def publish_terminal_productions(
    drafts: tuple[RuleDraft, ...],
    terminals: tuple[TerminalVocabulary, ...],
    generator: torch.Generator,
) -> None:
    """Label every draft, use every terminal, and populate its LHS node."""

    remaining_slots = sum(terminal_arity(draft.family) for draft in drafts)
    missing = _shuffle(list(terminals), generator)
    used_by_bucket: dict[
        tuple[object, ...], set[tuple[TerminalVocabulary, ...]]
    ] = {}

    for draft in drafts:
        arity = terminal_arity(draft.family)
        selected: tuple[TerminalVocabulary, ...] = ()
        if arity:
            remaining_slots -= arity
            required_new = max(0, len(missing) - remaining_slots)
            used = used_by_bucket.setdefault(bucket_key(draft), set())
            selected = _choose_terminal_tuple(
                terminals=terminals,
                missing=missing,
                arity=arity,
                required_new=required_new,
                used=used,
                generator=generator,
            )
            used.add(selected)
            selected_set = set(selected)
            missing = [terminal for terminal in missing if terminal not in selected_set]
        draft.lhs.add_production(*realize_production(draft, selected))

    if missing:
        raise RuntimeError("terminal assignment left a TerminalVocabulary unused")


def _choose_terminal_tuple(
    *,
    terminals: tuple[TerminalVocabulary, ...],
    missing: list[TerminalVocabulary],
    arity: int,
    required_new: int,
    used: set[tuple[TerminalVocabulary, ...]],
    generator: torch.Generator,
) -> tuple[TerminalVocabulary, ...]:
    """Choose an unused tuple while reserving enough future terminal coverage."""

    preferred_new = min(arity, len(missing))
    for _ in range(_RANDOM_ATTEMPTS):
        chosen = _sample_without_replacement(missing, preferred_new, generator)
        while len(chosen) < arity:
            chosen.append(terminals[_random_index(len(terminals), generator)])
        candidate = tuple(_shuffle(chosen, generator))
        if candidate not in used:
            return candidate

    missing_set = set(missing)
    for new_count in range(preferred_new, required_new - 1, -1):
        for candidate in product(terminals, repeat=arity):
            if candidate in used:
                continue
            if len(set(candidate) & missing_set) >= new_count:
                return tuple(candidate)
    raise RuntimeError("no unique terminal assignment remains for a rule topology")


def _random_index(size: int, generator: torch.Generator) -> int:
    """Draw one index from the explicit terminal-label random stream."""

    if size <= 0:
        raise RuntimeError("cannot sample an empty choice set")
    return int(torch.randint(size, (), generator=generator))


def _shuffle(
    values: list[_Value],
    generator: torch.Generator,
) -> list[_Value]:
    """Return a random ordering without mutating the supplied list."""

    if len(values) < 2:
        return values.copy()
    order = torch.randperm(len(values), generator=generator).tolist()
    return [values[index] for index in order]


def _sample_without_replacement(
    values: list[_Value],
    count: int,
    generator: torch.Generator,
) -> list[_Value]:
    """Select distinct terminal nodes from the explicit label stream."""

    if count == 0:
        return []
    if count > len(values):
        raise RuntimeError("cannot sample more unique values than are available")
    order = torch.randperm(len(values), generator=generator)[:count].tolist()
    return [values[index] for index in order]
